#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <camera_queue.hpp>
#include <libcamera/libcamera.h>
#include <sys/mman.h>
#include "cam_downsampler.hpp"

class Camera {
public:
    Camera(libcamera::CameraManager* camera_manager, uint8_t id, CameraQueue& queue)
        : cm_(camera_manager), id_(id), queue_(queue)
    {
    }

    ~Camera()
    {
        if (running_)
            stop();
    }

    void start(){
        camera_ = cm_->cameras()[id_];
        camera_->acquire();

        auto config = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
        config->at(0).pixelFormat = libcamera::formats::BGR888;
        camera_->configure(config.get());
        stream_ = config->at(0).stream();

        allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
        allocator_->allocate(stream_);
        const auto& buffers = allocator_->buffers(stream_);

        for (const auto& buffer : buffers){
            std::unique_ptr<libcamera::Request> request = camera_->createRequest();
            request->addBuffer(stream_, buffer.get());
            requests_.push_back(std::move(request));
        }

        camera_->requestCompleted.connect(this, &Camera::on_request_completed);

        running_ = true;
        camera_->start();
        for (auto& request : requests_)
            camera_->queueRequest(request.get());
    }

    void stop()
    {
        // Set running_ false first so the callback stops re-queuing.
        running_ = false;

        // Wait for any in-progress callback to finish before calling
        // camera_->stop(). Without this, libcamera can segfault in
        // doCancelRequest() while our callback is still reading the request.
        std::lock_guard<std::mutex> lk(callback_mutex_);

        camera_->stop();
        camera_->requestCompleted.disconnect(this, &Camera::on_request_completed);
        allocator_->free(stream_);
        camera_->release();
    }

private:
    std::atomic<bool> running_{false};
    std::mutex        callback_mutex_;   // held by callback while running, acquired by stop()

    CameraQueue& queue_;
    uint8_t id_;
    libcamera::CameraManager* cm_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    libcamera::Stream* stream_;
    CamDownsampler downsampler_;

    void on_request_completed(libcamera::Request* request)
    {
        // Check cancellation BEFORE acquiring the mutex.
        // Cancelled callbacks are triggered by camera_->stop() which is called
        // while holding callback_mutex_ — if we tried to acquire it here we
        // would deadlock.
        if (request->status() == libcamera::Request::RequestCancelled)
            return;

        // Hold the mutex for the rest of the callback so stop() cannot call
        // camera_->stop() while we are still reading the request object.
        std::lock_guard<std::mutex> lk(callback_mutex_);

        const libcamera::FrameBuffer* buffer = request->buffers().begin()->second;
        const libcamera::FrameBuffer::Plane& plane = buffer->planes()[0];

        void* mapped = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED,
                            plane.fd.get(), plane.offset);

        if (mapped == MAP_FAILED) {
            if (running_) {
                request->reuse(libcamera::Request::ReuseBuffers);
                camera_->queueRequest(request);
            }
            return;
        }

        const uint8_t* data_ptr = static_cast<const uint8_t*>(mapped);

        std::vector<uint8_t> downsampled_frame(640 * 640 * 3);
        bool ok = downsampler_.process(
            data_ptr, 1920, 1080, 1920 * 3,
            downsampled_frame.data(), 640 * 3
        );

        munmap(mapped, plane.length);

        if (ok) {
            FramePacket packet;
            packet.camera_id    = id_;
            packet.timestamp_us = buffer->metadata().timestamp / 1000;
            packet.data         = std::move(downsampled_frame);
            queue_.push(std::move(packet));
        }

        if (running_) {
            request->reuse(libcamera::Request::ReuseBuffers);
            camera_->queueRequest(request);
        }
    }
};
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include <libcamera/libcamera.h>
#include <sys/mman.h>

#include "cam_downsampler.hpp"
#include "camera_queue.hpp"
#include "camera_config.hpp"
#include "frame_packet.hpp"
#include "inference_config.hpp"



class Camera {
public:
    Camera(libcamera::CameraManager* camera_manager,
           uint8_t                   id,
           queue<FramePacket>&       q,
           int                       fps = CAMERA_FPS)
        : cm_(camera_manager), id_(id), queue_(q),
          fps_(fps), downsampler_(CAMERA_WIDTH, CAMERA_HEIGHT,
                                  INPUT_WIDTH,  INPUT_HEIGHT)
    {}

    bool start()
    {
        camera_ = cm_->cameras()[id_];
        if (camera_->acquire() != 0) return false;

        auto config = camera_->generateConfiguration(
            { libcamera::StreamRole::VideoRecording });

        // Request RGB888 — the YOLOv8n HEF expects RGB input.
        // The IMX708 ISP handles the Bayer→RGB conversion in hardware.
        config->at(0).pixelFormat = libcamera::formats::BGR888;
        config->at(0).size        = { CAMERA_WIDTH, CAMERA_HEIGHT };
        config->at(0).bufferCount = 6;

        if (config->validate() == libcamera::CameraConfiguration::Invalid) {
            camera_->release();
            return false;
        }
        camera_->configure(config.get());
        stream_ = config->at(0).stream();

        allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
        allocator_->allocate(stream_);

        const int64_t frame_dur_us = 1'000'000 / fps_;

        for (const auto& buffer : allocator_->buffers(stream_)) {
            auto request = camera_->createRequest();
            request->addBuffer(stream_, buffer.get());
            request->controls().set(
                libcamera::controls::FrameDurationLimits,
                libcamera::Span<const int64_t, 2>({frame_dur_us, frame_dur_us}));
            requests_.push_back(std::move(request));
        }

        camera_->requestCompleted.connect(this, &Camera::on_request_completed);
        camera_->start();

        for (auto& req : requests_)
            camera_->queueRequest(req.get());

        return true;
    }

    void stop()
    {
        camera_->stop();
        camera_->requestCompleted.disconnect(this, &Camera::on_request_completed);
        allocator_->free(stream_);
        camera_->release();
        requests_.clear();
    }

private:
    libcamera::CameraManager*                        cm_;
    uint8_t                                          id_;
    queue<FramePacket>&                              queue_;
    int                                              fps_;
    CamDownsampler                                   downsampler_;

    std::shared_ptr<libcamera::Camera>               camera_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    libcamera::Stream*                               stream_ = nullptr;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;

    void on_request_completed(libcamera::Request* request)
    {
        if (request->status() == libcamera::Request::RequestCancelled)
            return;

        const libcamera::FrameBuffer* buf =
            request->buffers().begin()->second;
        const libcamera::FrameBuffer::Plane& plane = buf->planes()[0];

        void* mapped = mmap(nullptr, plane.length,
                            PROT_READ, MAP_SHARED,
                            plane.fd.get(), plane.offset);
        if (mapped == MAP_FAILED) {
            request->reuse(libcamera::Request::ReuseBuffers);
            camera_->queueRequest(request);
            return;
        }

        // Downsample 1920×1080 → 640×640.  Channel order (RGB) is preserved.
        std::vector<uint8_t> dst(INPUT_WIDTH * INPUT_HEIGHT * 3);
        downsampler_.process(static_cast<const uint8_t*>(mapped), dst.data());
        munmap(mapped, plane.length);

        // libcamera delivers BGR888 bytes regardless of format name.
        // Swap BGR->RGB in-place so FramePacket carries RGB for YOLOv8n.
        {
            uint8_t* p = dst.data();
            for (std::size_t i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; ++i, p += 3) {
                uint8_t tmp = p[0]; p[0] = p[2]; p[2] = tmp;
            }
        }

        // libcamera delivers BGR888 bytes regardless of format name.
        // Swap BGR->RGB in-place so FramePacket carries RGB for YOLOv8n.
        {
            uint8_t* p = dst.data();
            for (std::size_t i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; ++i, p += 3) {
                uint8_t tmp = p[0]; p[0] = p[2]; p[2] = tmp;
            }
        }

        FramePacket pkt;
        pkt.camera_id = id_;
        pkt.timestamp = buf->metadata().timestamp; // nanoseconds
        pkt.data      = std::move(dst);            // RGB bytes

        queue_.push(pkt);

        request->reuse(libcamera::Request::ReuseBuffers);
        camera_->queueRequest(request);
    }
};
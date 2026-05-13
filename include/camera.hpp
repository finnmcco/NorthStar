#pragma once
#include <cstdint>
#include <camera_queue.hpp>
#include <libcamera/libcamera.h>
#include <sys/mman.h>
#include "cam_downsampler.hpp"
#include <atomic>

class Camera {
public:
    Camera(libcamera::CameraManager* camera_manager, uint8_t id, CameraQueue& queue)
        : cm_(camera_manager), id_(id), queue_(queue)
    {
    }

    void start(){
        requests_.clear();

        camera_ = cm_->cameras()[id_]; //get the correct camera (based on ID) from libcamera's list of cameras
        camera_->acquire(); //claim exclusive access to the camera - nothing else can use it from now on

        //first, create a default configuration based on its role as a video capture camera
        auto config = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
        //now format to RGB888
        config->at(0).pixelFormat = libcamera::formats::BGR888;
        //now set the resolution here so we don't need to downscale later
        //config->at(0).size = { 640, 640 };
        //now apply the configuration to the camera hardware
        camera_->configure(config.get());
        stream_ = config->at(0).stream(); //save stream pointer as member variable to stop it going out of scope
        //pass the camera to libcamera's memory buffer allocator class
        allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
        //call allocate() on the stream configured above
        allocator_->allocate(stream_);
        const auto& buffers = allocator_->buffers(stream_);
        
        /* libcamera is asynchronous and event-driven - you don't ask it for frames,
        it tells you when they are ready. 

        For each buffer, you create a request object and attach the buffer to it
        "here's an empty buffer, please fill it with a frame and tell me when you're done"
        */
        for (const auto& buffer : buffers){ //for each buffer
            std::unique_ptr<libcamera::Request> request = camera_->createRequest(); //create a request object
            request->addBuffer(stream_, buffer.get()); //and attach the buffer to it
            requests_.push_back(std::move(request));  //add to class member
        }

        /* connect a handler to libcamera's requestCompleted signal -
        this signal fires every time the camera finishes filling a buffer
        on_request_completed receives the completed request and is called automatically
        by libcamera on a separate thread whenever a frame is ready */
        camera_->requestCompleted.connect(this, &Camera::on_request_completed);

        camera_->start();
        for (auto& request : requests_) {
            camera_->queueRequest(request.get());
        }
        
    }

    /*
    void stop()
    {
        // Disconnect the callback so no further frames are delivered
        // after we start tearing down resources
        camera_->requestCompleted.disconnect(this, &Camera::on_request_completed);

        // Stop the camera hardware from filling buffers
        // This will cancel any queued requests, triggering on_request_completed
        // one final time with RequestCancelled status for each pending request
        camera_->stop();

        // Free the allocated frame buffers
        allocator_->free(stream_);

        // Release exclusive access to the camera
        camera_->release();

        //debugging from claude:
        camera_.reset();
        allocator_.reset();
    }
        */
    void stop() {
        draining_.store(true);
        {
            // Wait for any in-flight callback to finish before tearing down
            std::lock_guard<std::mutex> lock(callback_mutex_);
        }

        camera_->stop();
        camera_->requestCompleted.disconnect(this, &Camera::on_request_completed);
        allocator_->free(stream_);
        camera_->release();
        camera_.reset();
        allocator_.reset();
        draining_.store(false);  // reset for next start()
    }

private:
    CameraQueue& queue_;
    uint8_t id_;
    libcamera::CameraManager* cm_; //instance of camera manager (shared - passed in from outside)
    std::shared_ptr<libcamera::Camera> camera_; //instance of camera
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    libcamera::Stream* stream_;
    CamDownsampler downsampler_;
    std::mutex callback_mutex_;
    std::atomic<bool> draining_{false};

    void on_request_completed(libcamera::Request* request)
    {
        if (request->status() == libcamera::Request::RequestCancelled) return;

        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (draining_.load()) return;  // bail before touching buffers or re-queueing

        // Get the buffer from the completed request
        // buffers() returns a map of stream -> FrameBuffer*
        // We only have one stream so we take the first entry
        const libcamera::FrameBuffer* buffer = request->buffers().begin()->second;

        // Get the first (and only) plane - for RGB888 all pixel data is in plane 0
        const libcamera::FrameBuffer::Plane& plane = buffer->planes()[0];

        // Memory-map the plane's file descriptor into our process's address space
        // This gives us a raw pointer we can read the pixel data from
        // PROT_READ  - we only need to read, not write
        // MAP_SHARED - shared with the camera hardware
        void* mapped = mmap(
            nullptr,          // let the OS choose the address
            plane.length,     // how many bytes to map
            PROT_READ,        // read only
            MAP_SHARED,       // shared mapping
            plane.fd.get(),   // file descriptor for the buffer
            plane.offset      // offset into the buffer
        );

        if (mapped == MAP_FAILED) {
            // mmap failed - log and re-queue without pushing a packet
            request->reuse(libcamera::Request::ReuseBuffers);
            camera_->queueRequest(request);
            return;
        }

        // Cast the void* to a byte pointer so we can copy the data
        const uint8_t* data_ptr = static_cast<const uint8_t*>(mapped);

        // In on_request_completed, replace the downsample line with:
        std::vector<uint8_t> downsampled_frame(640 * 640 * 3);
        bool ok = downsampler_.process(
            data_ptr,
            1920, 1080,
            1920 * 3,
            downsampled_frame.data(),
            640 * 3
        );
        if (!ok) {
            munmap(mapped, plane.length);
            request->reuse(libcamera::Request::ReuseBuffers);
            camera_->queueRequest(request);
            return;
        }   

        // Unmap the memory-mapped region now that we've copied the data
        munmap(mapped, plane.length);

        // Get the timestamp from the buffer metadata (in nanoseconds)
        uint64_t timestamp = buffer->metadata().timestamp;

        // Build the FramePacket and push into the shared queue
        FramePacket packet;
        packet.camera_id   = id_;
        packet.timestamp_us = timestamp / 1000; //us
        packet.data        = std::move(downsampled_frame);

        queue_.push(std::move(packet));

        // Reset the request so it can be submitted again
        // ReuseBuffers keeps the same buffers attached
        request->reuse(libcamera::Request::ReuseBuffers);

        // Re-queue the request so the camera has a buffer to write the next frame into
        camera_->queueRequest(request);
    }
};

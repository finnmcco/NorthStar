#pragma once
#include <chrono>
#include <memory>
#include <thread>

#include <libcamera/libcamera.h>

#include "camera.hpp"
#include "camera_config.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"

class CameraCapture {
public:
    CameraCapture(queue<FramePacket>& q, int fps = CAMERA_FPS)
        : camera_manager_(std::make_unique<libcamera::CameraManager>()),
          cam0_(camera_manager_.get(), 0, q, fps)
    {}

    bool start()
    {
        if (camera_manager_->start() != 0)
            return false;

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        if (camera_manager_->cameras().empty())
            return false;

        return cam0_.start();
    }

    void stop()
    {
        cam0_.stop();
        camera_manager_->stop();
    }

private:
    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    Camera                                    cam0_;
};
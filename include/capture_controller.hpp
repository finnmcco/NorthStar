#pragma once
#include <libcamera/libcamera.h>
#include "camera.hpp"
#include "camera_queue.hpp"
#include <thread>
#include <chrono>

class CaptureController {
public:
    CaptureController()
        :   camera_manager_(std::make_unique<libcamera::CameraManager>()),
            queue_(300),
            cam0_(camera_manager_.get(), 0, queue_),
            cam1_(camera_manager_.get(), 1, queue_)
    {
        camera_manager_->start();
    }

    void start_capture(){
        cam0_.start();
        cam1_.start();
        
        //wait for the cameras to wake up fully
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        timer_thread_ = std::thread([this] {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            stop_capture();
        });
    }

    void stop_capture() {
        cam0_.stop();
        cam1_.stop();
        queue_.stop();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        camera_manager_->stop();

        if (timer_thread_.joinable()) {
         timer_thread_.detach();
        }
    }
    
    CameraQueue& get_queue() {
        return queue_;
    }
    
    std::size_t queue_size() {
        return queue_.size();
    }

private:
    std::thread timer_thread_;
    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    CameraQueue queue_;
    Camera cam0_;
    Camera cam1_;
};

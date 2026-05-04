#pragma once
#include <libcamera/libcamera.h>
#include "camera.hpp"
#include "camera_queue.hpp"
#include "ir_queue.hpp"
#include <thread>
#include <chrono>

class CaptureController {
public:
    CaptureController()
        :   camera_manager_(std::make_unique<libcamera::CameraManager>()),
            cam_queue_(300),
            ir_queue_(50),
            ir_capture_(ir_queue_, "/dev/i2c-1"),
            cam0_(camera_manager_.get(), 0, queue_),
            cam1_(camera_manager_.get(), 1, queue_)
    {
        camera_manager_->start();
    }

    void start_capture(){
        cam0_.start();
        cam1_.start();
        ir_capture_.start();
        
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
        ir_capture_.stop();
        cam_queue_.stop();
        ir_queue_.stop();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        camera_manager_->stop();

        if (timer_thread_.joinable()) {
         timer_thread_.detach();
        }
    }
    
    CameraQueue& get_cam_queue() {
        return cam_queue_;
    }

    IRQueue& get_ir_queue() {
        return ir_queue_;
    }
    
    std::size_t cam_queue_size() {
        return cam_queue_.size();
    }

    std::size_t ir_queue_size() {
        return ir_queue_.size();
    }

private:
    std::thread timer_thread_;
    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    CameraQueue cam_queue_;
    IRQueue ir_queue_;
    IRCapture ir_capture_;

    Camera cam0_;
    Camera cam1_;
};

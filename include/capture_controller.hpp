#pragma once
#include <libcamera/libcamera.h>
#include "camera.hpp"
#include "camera_queue.hpp"
#include "ir_queue.hpp"
#include "ir_capture.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

class CaptureController {
public:
    CaptureController()
        :   camera_manager_(std::make_unique<libcamera::CameraManager>()),
            cam_queue_(300),
            ir_queue_(50),
            ir_capture_(ir_queue_, "/dev/i2c-1"),
            cam0_(camera_manager_.get(), 0, cam_queue_),
            cam1_(camera_manager_.get(), 1, cam_queue_)
    {
        camera_manager_->start();
    }

    ~CaptureController()
    {
        // Ensure everything is stopped and wake the timer thread if still
        // sleeping so it exits promptly.
        stop_capture();

        // Join the timer thread from outside — never join from inside the
        // timer thread itself (that would be a thread joining itself).
        if (timer_thread_.joinable())
            timer_thread_.join();
    }

    void start_capture()
    {
        cam0_.start();
        cam1_.start();
        ir_capture_.start();

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Timer thread: sleep 5 s (or until woken by stop_capture()),
        // then call stop_capture(). It does NOT join itself — the destructor
        // handles that from the main thread.
        timer_thread_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(timer_mutex_);
            timer_cv_.wait_for(lk, std::chrono::seconds(5));
            lk.unlock();
            stop_capture();
            // Thread exits here. Destructor will join() it.
        });
    }

    void stop_capture()
    {
        bool expected = false;
        if (!stopped_.compare_exchange_strong(expected, true)) return;

        cam0_.stop();
        cam1_.stop();
        ir_capture_.stop();
        cam_queue_.stop();
        ir_queue_.stop();

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        camera_manager_->stop();

        // Wake the timer thread if it is still sleeping so it exits promptly.
        // Do NOT join here — stop_capture() can be called from the timer
        // thread itself, and a thread cannot join itself.
        timer_cv_.notify_all();
    }

    CameraQueue& get_cam_queue() { return cam_queue_; }
    IRQueue&     get_ir_queue()  { return ir_queue_;  }

    std::size_t cam_queue_size() { return cam_queue_.size(); }
    std::size_t ir_queue_size()  { return ir_queue_.size();  }

private:
    std::atomic<bool>       stopped_{false};
    std::thread             timer_thread_;
    std::mutex              timer_mutex_;
    std::condition_variable timer_cv_;

    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    CameraQueue cam_queue_;
    IRQueue     ir_queue_;
    IRCapture   ir_capture_;
    Camera      cam0_;
    Camera      cam1_;
};
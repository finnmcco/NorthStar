#pragma once
#include <queue>
#include "frame_packet.hpp"
#include <mutex>
#include <condition_variable>
#include <optional>

class CameraQueue {
public:
    CameraQueue(std::size_t max_capacity)
        : max_capacity_(max_capacity), stopped_(false)
    {
    }

    bool push(FramePacket frame) { //push a frame into queue
        /* construct an object called lock of type std::unique_lock<std::mutex>,
        passing our mutex_ member into its constructor
        The constructor is what acquires the lock

        Mutex is required when multiple threads can access a shared resource
        */
        std::unique_lock<std::mutex> lock(mutex_); //now only one thread can push at a time

        if (stopped_ == true){
            return false;
        }

        if (queue_.size() >= max_capacity_){ //queue full: drop oldest frame
            queue_.pop();
        }

        queue_.push(std::move(frame)); //add new frame to the back. std::move avoids copying the frame buffer
        cv_.notify_one(); // if a consumer thread is waiting in pop(), WAKE the thread
        return true; //mutex goes out of scope here and unlocks
    }

    std::optional<FramePacket> pop(){ //pop a frame from queue
        /* we use unique_lock because the mutex needs to temporarily unlock 
        when waiting on the condition variable to become true
        */
        std::unique_lock<std::mutex> lock(mutex_); 

        //wait until either there is data in the queue, or the queue is shut down
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; }); 

        if(stopped_ && queue_.empty()){
            return std::nullopt; //signal to caller that queue is shut down
        }

        //now we've guaranteed the queue isn't empty and can read a frame from it
        //move the oldest frame into an output variable
        FramePacket frame = std::move(queue_.front());
        //now delete the frame from queue
        queue_.pop();
        return frame; //mutex now goes out of scope and unlocks
    }

    // Stop the queue: wakes all waiting threads.
    // After stop(), push() returns false, and pop() returns false once drained.
    void stop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all(); // any thread waiting in pop() wakes up
    }

    std::size_t size(){ //returns current number of frames in queue
        std::unique_lock<std::mutex> lock(mutex_);

        return queue_.size();
    }

private:
    std::size_t max_capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<FramePacket> queue_;
    bool stopped_;
};

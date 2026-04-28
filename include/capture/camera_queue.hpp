#include <queue>
#include "frame_packet.hpp"
#include <mutex>
#include <condition_variable>

class CameraQueue {
public:
    CameraQueue(std::size_t max_capacity)
        : max_capacity_(max_capacity)
    {
    }

    bool push(FramePacket frame) { //push a frame into queue
        /* construct an object called lock of type std::unique_lock<std::mutex>,
        passing our mutex_ member into its constructor
        The constructor is what acquires the lock

        Mutex is required when multiple threads can access a shared resource
        */
        std::unique_lock<std::mutex> lock(mutex_); //now only one thread can push at a time
    }

    FramePacket pop(){ //pop a frame from queue
        std::unique_lock<std::mutex> lock(mutex_);

    }

    std::size_t size(){ //returns current number of frames in queue
        std::unique_lock<std::mutex> lock(mutex_);
    }
private:
    std::size_t max_capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
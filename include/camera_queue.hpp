#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <cstddef>

/*
    queue<T>  (formerly CameraQueue)
    ════════════════════════════════
    Thread-safe, capacity-bounded FIFO.

    push(item)  — if at capacity, drops the oldest item (drop-oldest policy).
                  Returns false if stop() has been called.

    pop(out)    — blocks until an item is available or the queue is stopped.
                  Returns false when stopped AND empty — the consumer's exit signal.

    stop()      — wakes all waiters; after this push() returns false and
                  pop() drains then returns false.
*/
template<typename T>
class queue
{
public:
    explicit queue(std::size_t max_capacity)
        : max_capacity_(max_capacity), stopped_(false)
    {}

    bool push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopped_) return false;
        if (q_.size() >= max_capacity_) q_.pop();
        q_.push(std::move(item));
        cv_.notify_one();
        return true;
    }

    bool pop(T& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !q_.empty() || stopped_; });
        if (stopped_ && q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void stop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

    std::size_t size()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return q_.size();
    }

private:
    std::size_t             max_capacity_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::queue<T>           q_;
    bool                    stopped_;
};
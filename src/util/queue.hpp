#pragma once
#include <cstdint>
#include <condition_variable>
#include <cstddef>   // size_t
#include <deque>
#include <mutex>
#include <utility>   // std::move

template <typename T> // template means that the queue can hold any data type
class queue {
    public: // everything below this line can be called outside the class
        explicit queue(std::size_t capacity) // allow only direct construction, no implicit conversions
            : capacity_(capacity) // initialise member variables
        {
        }

        //queue contains things like std::mutex
        //if someone accidentally copied the queue, two queues would share the same mutex or internal state
        // => disable copying entirely
        queue(const queue&) = delete; // delete the copy constructor. This function is called in cases such as [ queue q2 = q1 ]
        queue& operator=(const queue&) = delete; // delete copy assignment operator, called in cases like [ q2 = q1 ]
        // delete tells the compiler that the function exists, but is illegal to use

        // Push an item into the queue.
        // Returns false if the queue has been stopped.
        bool push(T item)
        {
            std::unique_lock<std::mutex> lock(mutex_); // lock the queue's mutex (prevent other threads from modifying queue at the same time)

            if (stopped_) { // if the queue has been stopped somewhere, don't accept new items
                return false;
            }

            // Drop-oldest policy if full
            if (capacity_ > 0 && queue_.size() >= capacity_) { // if queue is full
                queue_.pop_front(); // drop the oldest frame
            }

            queue_.push_back(std::move(item)); // add the new item

            lock.unlock(); // now unlock the mutex
            cv_.notify_one(); // if a consumer thread is waiting in pop(), WAKE the thread
            return true;
        }

        // Pop an item from the queue (oldest). Works like a blocking read.
        // Returns false if stopped and the queue is empty (no more items will arrive).
        bool pop(T& out)
        {
            std::unique_lock<std::mutex> lock(mutex_); // again, lock the mutex

            // Wait until there is data OR we are stopped: BLOCK!
            cv_.wait(lock, [&] { return stopped_ || !queue_.empty(); });

            if (queue_.empty()) { // we have no data
                // stopped_ MUST be true here
                return false;
            }

            out = std::move(queue_.front()); // move the oldest item into an output variable
            queue_.pop_front(); // delete it from the queue
            return true;
        }

        // Stop the queue: wakes all waiting threads.
        // After stop(), push() returns false, and pop() returns false once drained.
        void stop()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stopped_ = true;
            lock.unlock();
            cv_.notify_all(); // any thread waiting in pop() wakes up
        }

        // Optional helpers (non-blocking)
        std::size_t size() const // returns current number of items
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        bool isStopped() const // lets other code check if shutdown has been signalled
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return stopped_;
        }
    
    private:
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    bool stopped_ = false;
};
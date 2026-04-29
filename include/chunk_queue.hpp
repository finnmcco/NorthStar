#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>

#include "config.hpp"

// Defined in main.cpp — set false by signal handler to trigger shutdown.
extern std::atomic<bool> g_running;

// Thread-safe bounded queue connecting the capture and inference threads.
//
// Drops the OLDEST chunk on overflow so the recogniser stays close to
// real-time rather than accumulating a growing backlog.
// Drop count is available at shutdown as a load diagnostic.

class ChunkQueue {
public:
    void push(std::vector<int16_t>&& chunk) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (static_cast<int>(q_.size()) >= Config::QUEUE_MAX) {
            q_.pop();
            ++drops_;
        }
        q_.push(std::move(chunk));
        cv_.notify_one();
    }

    // Blocks until a chunk is ready or shutdown() is called.
    // Returns false when woken with an empty queue (shutdown path).
    bool pop(std::vector<int16_t>& out) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this] { return !q_.empty() || !g_running; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void   shutdown() { cv_.notify_all(); }
    size_t drops()  const { return drops_.load(); }

private:
    std::queue<std::vector<int16_t>> q_;
    std::mutex                       mtx_;
    std::condition_variable          cv_;
    std::atomic<size_t>              drops_{0};
};

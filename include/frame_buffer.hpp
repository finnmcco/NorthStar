#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <opencv2/core.hpp>

class FrameBuffer {
public:
    explicit FrameBuffer(std::size_t max_size = 60) : max_size_(max_size) {}

    void push(uint64_t ts_ns, cv::Mat frame) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (buf_.size() >= max_size_) buf_.pop_front();
        buf_.push_back({ ts_ns, std::move(frame) });
    }

    // Returns the frame whose timestamp is closest to ts_ns.
    // Returns an empty Mat if the buffer is empty.
    cv::Mat find_closest(uint64_t ts_ns) const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (buf_.empty()) return {};
        const FrameEntry* best = nullptr;
        uint64_t best_delta = UINT64_MAX;
        for (const auto& e : buf_) {
            uint64_t delta = (e.timestamp_ns > ts_ns)
                           ? e.timestamp_ns - ts_ns
                           : ts_ns - e.timestamp_ns;
            if (delta < best_delta) { best_delta = delta; best = &e; }
        }
        return best ? best->frame.clone() : cv::Mat{};
    }

private:
    struct FrameEntry {
        uint64_t timestamp_ns;
        cv::Mat  frame;
    };

    mutable std::mutex      mtx_;
    std::deque<FrameEntry>  buf_;
    std::size_t             max_size_;
};
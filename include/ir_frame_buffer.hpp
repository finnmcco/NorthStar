#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

#include "MLX90640.hpp"  // adjust include name/path if needed

class IRFrameBuffer {
public:
    struct IRFrame {
        uint64_t timestamp_ns = 0;
        std::array<float, MLX90640::PIXEL_COUNT> temps{};
        float ambientTemp = 0.0f;
    };

    explicit IRFrameBuffer(std::size_t max_size = 60)
        : max_size_(max_size)
    {}

    void push(uint64_t ts_ns,
              const std::array<float, MLX90640::PIXEL_COUNT>& temps,
              float ambientTemp)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (buf_.size() >= max_size_) {
            buf_.pop_front();
        }

        buf_.push_back(IRFrame{
            ts_ns,
            temps,
            ambientTemp
        });
    }

    void push(const IRFrame& frame)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (buf_.size() >= max_size_) {
            buf_.pop_front();
        }

        buf_.push_back(frame);
    }

    // Returns the IR frame whose timestamp is closest to ts_ns.
    // Returns std::nullopt if the buffer is empty.
    std::optional<IRFrame> find_closest(uint64_t ts_ns) const
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (buf_.empty()) {
            return std::nullopt;
        }

        const IRFrame* best = nullptr;
        uint64_t best_delta = UINT64_MAX;

        for (const auto& e : buf_) {
            const uint64_t delta = (e.timestamp_ns > ts_ns)
                                 ? e.timestamp_ns - ts_ns
                                 : ts_ns - e.timestamp_ns;

            if (delta < best_delta) {
                best_delta = delta;
                best = &e;
            }
        }

        return best ? std::optional<IRFrame>(*best) : std::nullopt;
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return buf_.size();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        buf_.clear();
    }

private:
    mutable std::mutex     mtx_;
    std::deque<IRFrame>    buf_;
    std::size_t            max_size_;
};
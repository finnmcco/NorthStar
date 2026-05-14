#pragma once

#include "ir_aligner.hpp"
#include "ir_frame_buffer.hpp"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <vector>

class TempEstimator {
public:
    static constexpr int IR_WIDTH  = 32;
    static constexpr int IR_HEIGHT = 24;

    std::optional<float> estimate_temp(const IRAligner::PixelRect& box,
                                       const IRFrameBuffer::IRFrame& frame) const
    {
        if (!box.valid) {
            std::printf("[temp] invalid IR box\n");
            std::fflush(stdout);
            return std::nullopt;
        }

        if (box.x1 < box.x0 || box.y1 < box.y0) {
            std::printf("[temp] IR box has negative size: [%d %d %d %d]\n",
                        box.x0, box.y0, box.x1, box.y1);
            std::fflush(stdout);
            return std::nullopt;
        }

        if (box.x0 < 0 || box.y0 < 0 ||
            box.x1 >= IR_WIDTH || box.y1 >= IR_HEIGHT) {
            std::printf("[temp] IR box out of bounds: [%d %d %d %d]\n",
                        box.x0, box.y0, box.x1, box.y1);
            std::fflush(stdout);
            return std::nullopt;
        }

        std::vector<float> temps_in_box;
        temps_in_box.reserve(
            static_cast<std::size_t>((box.x1 - box.x0 + 1) *
                                     (box.y1 - box.y0 + 1))
        );

        for (int y = box.y0; y <= box.y1; ++y) {
            for (int x = box.x0; x <= box.x1; ++x) {
                const int pixel = y * IR_WIDTH + x;
                temps_in_box.push_back(frame.temps[pixel]);
            }
        }

        if (temps_in_box.empty()) {
            return std::nullopt;
        }

        return median(temps_in_box);
    }

private:
    static float median(std::vector<float> v)
    {
        std::sort(v.begin(), v.end());

        const std::size_t n = v.size();

        if (n % 2 != 0) {
            return v[n / 2];
        }

        return 0.5f * (v[(n - 1) / 2] + v[n / 2]);
    }
};
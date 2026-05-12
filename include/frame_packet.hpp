#pragma once
#include <cstdint>
#include <array>
#include "MLX90640.hpp" // for PIXEL_COUNT

struct FramePacket {
    uint8_t camera_id;
    uint64_t timestamp_us;
    std::vector<uint8_t> data;
};

struct IRPacket {
    uint64_t                                    timestamp_us;  // nanoseconds, monotonic
    std::array<float, MLX90640::PIXEL_COUNT>    temps;      // smoothed pixel temps in °C
    float                                       ambientTemp;
};


#pragma once
#include <cstdint>
#include <vector>

struct FramePacket {
    uint8_t              camera_id  = 0;
    uint64_t             timestamp  = 0;  // nanoseconds
    std::vector<uint8_t> data;
};
#pragma once
#include <cstdint>

struct FramePacket {
    uint8_t camera_id;
    uint64_t timestamp_us;
    std::vector<uint8_t> data;
};


#pragma once
#include <cstdint>

struct FramePacket {
    uint64_t timestamp;
    uint8_t* data;
};


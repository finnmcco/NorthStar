#include <cstdint>

struct FramePacket {
    uint8_t cam_id;
    uint64_t timestamp;
    std::vector<uint8_t> data;
};


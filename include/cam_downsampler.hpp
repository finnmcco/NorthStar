#pragma once
#include <cstdint>
#include <vector>

class CamDownsampler {
public:
    CamDownsampler(uint16_t inWidth, uint16_t inHeight, uint16_t outWidth, uint16_t outHeight);
    std::vector<uint8_t> process(const uint8_t* dataIn);
private:
    std::vector<uint16_t> in_start_y_;
    std::vector<uint16_t> in_start_x_;
    std::vector<uint16_t> in_end_y_;
    std::vector<uint16_t> in_end_x_;

    uint16_t inWidth_;
    uint16_t inHeight_;
    uint16_t outWidth_;
    uint16_t outHeight_;
};
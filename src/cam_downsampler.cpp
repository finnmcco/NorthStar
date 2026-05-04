#include "cam_downsampler.hpp"
#include <cmath>

CamDownsampler::CamDownsampler(uint16_t inWidth, uint16_t inHeight,
                               uint16_t outWidth, uint16_t outHeight)
    : inWidth_(inWidth), inHeight_(inHeight),
      outWidth_(outWidth), outHeight_(outHeight)
{
    in_start_x_.resize(outWidth_);
    in_end_x_.resize(outWidth_);
    in_start_y_.resize(outHeight_);
    in_end_y_.resize(outHeight_);

    float scaler  = static_cast<float>(inHeight_) / static_cast<float>(outHeight_);
    uint16_t crop_x = (inWidth_ - inHeight_) / 2;

    for (uint16_t i = 0; i < outWidth_; i++) {
        in_start_x_[i] = static_cast<uint16_t>(std::round(crop_x + i * scaler));
        in_end_x_[i]   = static_cast<uint16_t>(std::round(crop_x + (i + 1) * scaler));
    }
    for (uint16_t i = 0; i < outHeight_; i++) {
        in_start_y_[i] = static_cast<uint16_t>(std::round(i * scaler));
        in_end_y_[i]   = static_cast<uint16_t>(std::round((i + 1) * scaler));
    }
}

void CamDownsampler::process(const uint8_t* src, uint8_t* dst)
{
    for (uint16_t out_y = 0; out_y < outHeight_; out_y++) {
        for (uint16_t out_x = 0; out_x < outWidth_; out_x++) {

            uint16_t sx0 = in_start_x_[out_x], sx1 = in_end_x_[out_x];
            uint16_t sy0 = in_start_y_[out_y], sy1 = in_end_y_[out_y];

            uint32_t sum_b = 0, sum_g = 0, sum_r = 0, count = 0;

            for (uint32_t y = sy0; y < sy1; y++) {
                for (uint32_t x = sx0; x < sx1; x++) {
                    size_t off = (y * inWidth_ + x) * 3;
                    sum_b += src[off];
                    sum_g += src[off + 1];
                    sum_r += src[off + 2];
                    count++;
                }
            }

            if (count == 0) continue;

            size_t out_off = (out_y * outWidth_ + out_x) * 3;
            dst[out_off]     = static_cast<uint8_t>(sum_b / count);
            dst[out_off + 1] = static_cast<uint8_t>(sum_g / count);
            dst[out_off + 2] = static_cast<uint8_t>(sum_r / count);
        }
    }
}
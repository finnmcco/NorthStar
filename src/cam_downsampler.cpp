#include "cam_downsampler.hpp"
#include <cmath>

CamDownsampler::CamDownsampler(uint16_t inWidth, uint16_t inHeight, uint16_t outWidth, uint16_t outHeight)
    :   inWidth_(inWidth), inHeight_(inHeight), outWidth_(outWidth), outHeight_(outHeight)
{
    in_start_x_.resize(outWidth_);
    in_end_x_.resize(outWidth_);
    in_start_y_.resize(outHeight_);
    in_end_y_.resize(outHeight_);

    float scaler = static_cast<float>(inHeight_) / static_cast<float>(outHeight_);
    uint16_t crop_x = (inWidth_ - inHeight_) / 2;  //crop to a square based on the height (width gets chopped on each side)
    for (uint16_t i = 0; i < outWidth_; i++){
        in_start_x_[i] = round(crop_x + (i * scaler));
        in_end_x_[i] = round(crop_x + ((i + 1) * scaler));
    }
    for (uint16_t i = 0; i < outHeight; i++){
        in_start_y_[i] = round(i * scaler);
        in_end_y_[i] = round((i + 1) * scaler);
    }
}

std::vector<uint8_t> CamDownsampler::process(const uint8_t* dataIn){ //pass the address of the first byte in the frame
    std::vector<uint8_t> output_data;
    output_data.resize(outWidth_ * outHeight_ * 3); //R,G and B bytes for each pixel

    for (uint16_t out_y = 0; out_y < outHeight_; out_y++) {
        for (uint16_t out_x = 0; out_x < outWidth_; out_x++) { //scope: ENTIRE FRAME
            //look up the source pixel range for this output pixel
            uint16_t src_x_start = in_start_x_[out_x];
            uint16_t src_x_end   = in_end_x_[out_x];
            uint16_t src_y_start = in_start_y_[out_y];
            uint16_t src_y_end   = in_end_y_[out_y];
            //now average all source pixels in the box
            //defined by (src_x_start, src_y_start) to (src_x_end, src_y_end)
            uint32_t sum_blue = 0;
            uint32_t sum_green = 0;
            uint32_t sum_red = 0;
            uint32_t count = 0;
            for (uint32_t y = src_y_start; y < src_y_end; y++){ //scope: small "box" of input pixels corresponding to an output pixel
                //loop through every pixel in the "box" that we are averaging
                for (uint32_t x = src_x_start; x < src_x_end; x++){ 
                    //offset: y * inWidth_ gets the "y" part of the index within dataIn
                    //adding x then offsets the index to the exact correct pixel entry
                    //multiply by three to make sure we are at the start of a pixel (blue byte)
                    size_t offset = (y * inWidth_ + x) * 3;
                    sum_blue += dataIn[offset];
                    sum_green += dataIn[offset + 1];
                    sum_red += dataIn[offset + 2];
                    count++;
                }
            }
            size_t output_offset = (out_y * outWidth_ + out_x) * 3;
            if (count == 0) continue;
            output_data[output_offset] = sum_blue / count;
            output_data[output_offset + 1] = sum_green / count;
            output_data[output_offset + 2] = sum_red / count;
        }
    }

    return output_data;
}
#pragma once

#include <string>
#include <cstdint>
#include <vector>

class Hailo8Inference;

namespace HailoHelper {

    // Assumes PNG is already 640x640 RGB-equivalent image
    std::vector<uint8_t> run_pipeline_png(
        Hailo8Inference& hailo,
        const std::string& image_path
    );

}
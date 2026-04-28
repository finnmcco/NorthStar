#include "inference/halio8_helper.hpp"
#include "inference/hailo8_inference.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace {

constexpr int MODEL_WIDTH  = 640;
constexpr int MODEL_HEIGHT = 640;
constexpr int CHANNELS     = 3;

}

namespace HailoHelper {

std::vector<uint8_t> run_pipeline_png(
    Hailo8Inference& hailo,
    const std::string& image_path)
{
    // ------------------------------------
    // 1. Load PNG (OpenCV loads BGR)
    // ------------------------------------
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to load image: "
                  << image_path << "\n";
        return {};
    }

    // ------------------------------------
    // 2. Validate resolution
    // ------------------------------------
    if (bgr.cols != MODEL_WIDTH ||
        bgr.rows != MODEL_HEIGHT) {
        std::cerr << "Image is not 640x640\n";
        return {};
    }

    // ------------------------------------
    // 3. Convert BGR -> RGB
    // ------------------------------------
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    const size_t expected_size =
        MODEL_WIDTH * MODEL_HEIGHT * CHANNELS;

    if (rgb.total() * rgb.elemSize() != expected_size) {
        std::cerr << "Unexpected buffer size\n";
        return {};
    }

    // ------------------------------------
    // 4. Run inference
    // ------------------------------------
    if (!hailo.run(rgb.data)) {
        std::cerr << "Hailo inference failed\n";
        return {};
    }

    return hailo.get_output();
}

}
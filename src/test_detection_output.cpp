/*
    test_detection_output.cpp
    ══════════════════════════
    Runs one real image through inference, prints a detection table, and saves
    an annotated PNG.

    Usage:  ./test_detection_output <image_path> [hef_path]

    PASS  callback fires within INFERENCE_TIMEOUT_S (zero detections is valid).
    FAIL  image load error, Hailo init error, or timeout.
*/

#include "config.hpp"
#include "inference_config.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"

static void print_result(bool pass, const std::string& detail = "")
{
    std::cout << (pass ? "\n  [PASS]\n"
                       : "\n  [FAIL]" + (detail.empty() ? "" : "  " + detail) + "\n");
}

static void print_detections(const std::vector<Detection>& dets, int w, int h)
{
    if (dets.empty()) { std::cout << "\n  No detections above threshold.\n"; return; }

    auto sorted = dets;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b){
                  return a.confidence > b.confidence;
              });

    constexpr int CW = 20, SW = 8, BW = 8;
    std::cout << "\n  "
              << std::left  << std::setw(CW) << "Class"
              << std::right << std::setw(SW)  << "Conf %"
              << std::setw(BW) << "x1" << std::setw(BW) << "y1"
              << std::setw(BW) << "x2" << std::setw(BW) << "y2" << "\n";
    std::cout << "  " << std::string(CW + SW + BW * 4, '-') << "\n";

    for (const auto& d : sorted) {
        std::cout << "  "
                  << std::left  << std::setw(CW) << COCO_CLASSES[d.object_id]
                  << std::right << std::setw(SW)
                      << std::fixed << std::setprecision(1) << (d.confidence * 100.f)
                  << std::setw(BW) << static_cast<int>(d.box.x_min * w)
                  << std::setw(BW) << static_cast<int>(d.box.y_min * h)
                  << std::setw(BW) << static_cast<int>(d.box.x_max * w)
                  << std::setw(BW) << static_cast<int>(d.box.y_max * h)
                  << "\n";
    }
    std::cout << "\n  Total: " << dets.size() << " detection(s)\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: test_detection_output <image_path> [hef_path]\n";
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string hef_path   = (argc >= 3) ? argv[2] : DEFAULT_HEF_PATH;

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: detection output\n"
              << "══════════════════════════════════════════\n"
              << "  Image : " << image_path << "\n";

    cv::Mat bgr_orig = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr_orig.empty()) { print_result(false, "could not load image"); return 1; }
    std::cout << "  Original size : " << bgr_orig.cols << "×" << bgr_orig.rows << "\n";

    cv::Mat bgr;
    if (bgr_orig.cols != INPUT_WIDTH || bgr_orig.rows != INPUT_HEIGHT) {
        std::cout << "  Resizing to " << INPUT_WIDTH << "×" << INPUT_HEIGHT << "\n";
        cv::resize(bgr_orig, bgr, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    } else {
        bgr = bgr_orig;
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    std::cout << "  Model : " << hef_path << "\n"
              << "  Initialising Hailo ... "; std::cout.flush();

    Hailo8Inference hailo(hef_path);

    std::promise<std::vector<std::vector<uint8_t>>> result_promise;
    auto result_future = result_promise.get_future();
    std::atomic<bool> callback_fired{false};

    hailo.register_callback(
        [&](uint8_t, uint64_t, std::vector<std::vector<uint8_t>> output)
        {
            bool expected = false;
            if (callback_fired.compare_exchange_strong(expected, true))
                result_promise.set_value(std::move(output));
        });

    if (!hailo.initialize()) {
        std::cout << "FAILED\n"; print_result(false, "Hailo init error"); return 1;
    }
    std::cout << "OK\n  Running inference ... "; std::cout.flush();

    if (!hailo.write_frame(rgb.data, 0, 0)) {
        std::cout << "FAILED\n"; print_result(false, "write_frame returned false");
        hailo.stop(); return 1;
    }

    if (result_future.wait_for(std::chrono::seconds(INFERENCE_TIMEOUT_S))
            == std::future_status::timeout) {
        std::cout << "TIMEOUT\n"; print_result(false, "callback timed out");
        hailo.stop(); return 1;
    }
    std::cout << "OK\n";
    hailo.stop();

    const auto detections = parse_detections(result_future.get());
    print_detections(detections, bgr.cols, bgr.rows);

    cv::Mat annotated = bgr.clone();
    draw_detections(annotated, detections);

    namespace fs = std::filesystem;
    const fs::path p(image_path);
    const std::string out_path =
        (p.parent_path() / (p.stem().string() + "_annotated.png")).string();

    if (cv::imwrite(out_path, annotated))
        std::cout << "  Annotated image saved: " << out_path << "\n";
    else
        std::cerr << "  Warning: could not save annotated image\n";

    print_result(true);
    return 0;
}
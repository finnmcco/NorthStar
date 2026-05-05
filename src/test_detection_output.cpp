/*
    test_detection_output.cpp
    ══════════════════════════
    Runs a single real image through the full inference + NMS parsing stack and
    prints every detection with its class label, confidence score, and pixel-
    space bounding box.  Also saves an annotated copy of the image to disk.

    This test is for visual verification: do the detections look right?

    Usage
    ─────
        ./test_detection_output <image_path> [hef_path]

        image_path   640×640 PNG or JPEG (resized automatically if needed)
        hef_path     default: DEFAULT_HEF_PATH (set in CMakeLists.txt)

    Output
    ──────
    Terminal:
        A table of all detections above CONF_THRESHOLD, sorted by confidence.

    File:
        <image_path>_annotated.png   — bounding boxes + labels drawn on the image.

    Pass / fail criteria
    ─────────────────────
    PASS  callback fires within 5 s (zero detections is still a PASS — it means
          the model ran cleanly; whether detections are correct is up to the user)
    FAIL  image load error, Hailo init error, or timeout
*/

#include "config.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
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

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_result(bool pass, const std::string& detail = "")
{
    if (pass)
        std::cout << "\n  [PASS]\n";
    else
        std::cout << "\n  [FAIL]" << (detail.empty() ? "" : "  " + detail) << "\n";
}

// Print a formatted detection table to stdout.
static void print_detections(const std::vector<Detection>& dets, int img_w, int img_h)
{
    if (dets.empty()) {
        std::cout << "\n  No detections above threshold.\n";
        return;
    }

    // Sort by descending confidence.
    auto sorted = dets;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b){ return a.score > b.score; });

    // Column widths
    constexpr int CW = 20; // class name
    constexpr int SW = 8;  // score
    constexpr int BW = 8;  // box coord

    std::cout << "\n  "
              << std::left  << std::setw(CW) << "Class"
              << std::right << std::setw(SW) << "Conf %"
              << std::setw(BW) << "x1"
              << std::setw(BW) << "y1"
              << std::setw(BW) << "x2"
              << std::setw(BW) << "y2"
              << "\n";
    std::cout << "  " << std::string(CW + SW + BW * 4, '-') << "\n";

    for (const auto& d : sorted) {
        const int x1 = static_cast<int>(d.x_min * img_w);
        const int y1 = static_cast<int>(d.y_min * img_h);
        const int x2 = static_cast<int>(d.x_max * img_w);
        const int y2 = static_cast<int>(d.y_max * img_h);

        std::cout << "  "
                  << std::left  << std::setw(CW) << COCO_CLASSES[d.class_id]
                  << std::right << std::setw(SW)
                      << std::fixed << std::setprecision(1) << (d.score * 100.f)
                  << std::setw(BW) << x1
                  << std::setw(BW) << y1
                  << std::setw(BW) << x2
                  << std::setw(BW) << y2
                  << "\n";
    }
    std::cout << "\n  Total: " << dets.size() << " detection(s)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: test_detection_output <image_path> [hef_path]\n";
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string hef_path   = (argc >= 3) ? argv[2]
                                               : DEFAULT_HEF_PATH;

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: detection output\n"
              << "══════════════════════════════════════════\n";

    // ── Load image ────────────────────────────────────────────────────────────
    std::cout << "  Image : " << image_path << "\n";

    cv::Mat bgr_orig = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr_orig.empty()) {
        print_result(false, "could not load image");
        return 1;
    }
    std::cout << "  Original size : " << bgr_orig.cols << "×" << bgr_orig.rows << "\n";

    cv::Mat bgr;
    if (bgr_orig.cols != 640 || bgr_orig.rows != 640) {
        std::cout << "  Resizing to 640×640\n";
        cv::resize(bgr_orig, bgr, cv::Size(640, 640));
    } else {
        bgr = bgr_orig;
    }

    // Convert to RGB for Hailo.
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    // ── Initialise Hailo ──────────────────────────────────────────────────────
    std::cout << "  Model : " << hef_path << "\n";
    std::cout << "  Initialising Hailo ... ";
    std::cout.flush();

    Hailo8Inference hailo(hef_path);

    std::promise<std::vector<std::vector<uint8_t>>> result_promise;
    auto result_future = result_promise.get_future();
    std::atomic<bool> callback_fired{false};

    hailo.register_callback(
        [&](uint8_t /*camera_id*/,
            uint64_t /*timestamp_ns*/,
            std::vector<std::vector<uint8_t>> output)
        {
            bool expected = false;
            if (callback_fired.compare_exchange_strong(expected, true))
                result_promise.set_value(std::move(output));
        });

    if (!hailo.initialize()) {
        std::cout << "FAILED\n";
        print_result(false, "Hailo init error");
        return 1;
    }
    std::cout << "OK\n";

    // ── Run inference ─────────────────────────────────────────────────────────
    std::cout << "  Running inference ... ";
    std::cout.flush();

    if (!hailo.write_frame(rgb.data, /*camera_id=*/0, /*timestamp_ns=*/0)) {
        std::cout << "FAILED\n";
        print_result(false, "write_frame returned false");
        hailo.stop();
        return 1;
    }

    const auto status = result_future.wait_for(std::chrono::seconds(5));
    if (status == std::future_status::timeout) {
        std::cout << "TIMEOUT\n";
        print_result(false, "callback did not fire within 5 s");
        hailo.stop();
        return 1;
    }
    std::cout << "OK\n";

    hailo.stop();

    // ── Parse and print detections ────────────────────────────────────────────
    const auto raw_output  = result_future.get();
    const auto detections  = parse_detections(raw_output);

    print_detections(detections, bgr.cols, bgr.rows);

    // ── Save annotated image ──────────────────────────────────────────────────
    // draw_detections expects BGR.
    cv::Mat annotated = bgr.clone();
    draw_detections(annotated, detections);

    namespace fs = std::filesystem;
    const fs::path input_path(image_path);
    const std::string out_path =
        (input_path.parent_path() / (input_path.stem().string() + "_annotated.png")).string();

    if (cv::imwrite(out_path, annotated))
        std::cout << "  Annotated image saved: " << out_path << "\n";
    else
        std::cerr << "  Warning: could not save annotated image to " << out_path << "\n";

    print_result(true);
    return 0;
}
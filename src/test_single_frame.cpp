/*
    test_single_frame.cpp
    ══════════════════════
    Verifies that a single frame makes it through the Hailo inference pipeline
    and that the callback fires with a correctly-sized output buffer.

    No real camera or image file is required — a synthetic RGB test pattern is
    generated if no image path is given.

    Usage
    ─────
        ./test_single_frame [hef_path] [image_path]

    Pass / fail criteria
    ─────────────────────
    PASS  callback fires within 5 s AND output buffer == expected frame size
    FAIL  timeout OR buffer size mismatch OR Hailo init error
*/

#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inference/hailo8_inference.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Fill a 640×640 RGB buffer with a diagonal gradient test pattern.
static std::vector<uint8_t> make_test_pattern()
{
    constexpr int W = 640, H = 640, C = 3;
    std::vector<uint8_t> buf(W * H * C);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int i = (y * W + x) * C;
            buf[i + 0] = static_cast<uint8_t>((x * 255) / (W - 1)); // R
            buf[i + 1] = static_cast<uint8_t>((y * 255) / (H - 1)); // G
            buf[i + 2] = 128;                                         // B
        }
    return buf;
}

// Load a 640×640 RGB buffer from an image file.
// Returns empty vector on failure.
static std::vector<uint8_t> load_image_rgb(const std::string& path)
{
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "  Could not load image: " << path << "\n";
        return {};
    }
    if (bgr.cols != 640 || bgr.rows != 640) {
        std::cout << "  Resizing from " << bgr.cols << "x" << bgr.rows << " to 640x640\n";
        cv::resize(bgr, bgr, cv::Size(640, 640));
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    return std::vector<uint8_t>(rgb.data, rgb.data + 640 * 640 * 3);
}

static void print_result(bool pass, const std::string& detail = "")
{
    if (pass)
        std::cout << "\n  [PASS]\n";
    else
        std::cout << "\n  [FAIL]" << (detail.empty() ? "" : "  " + detail) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const std::string hef_path   = (argc >= 2) ? argv[1]
                                               : "/usr/share/hailo-models/yolov8s_h8.hef";
    const std::string image_path = (argc >= 3) ? argv[2] : "";

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: single frame inference\n"
              << "══════════════════════════════════════════\n";

    // ── Load / generate frame ─────────────────────────────────────────────────
    std::vector<uint8_t> frame_rgb;
    if (image_path.empty()) {
        std::cout << "  Input : synthetic 640×640 RGB test pattern\n";
        frame_rgb = make_test_pattern();
    } else {
        std::cout << "  Input : " << image_path << "\n";
        frame_rgb = load_image_rgb(image_path);
        if (frame_rgb.empty()) {
            print_result(false, "image load failed");
            return 1;
        }
    }

    // ── Initialise Hailo ──────────────────────────────────────────────────────
    std::cout << "  Model : " << hef_path << "\n";
    std::cout << "  Initialising Hailo ... ";
    std::cout.flush();

    Hailo8Inference hailo(hef_path);

    // Promise carries the first output stream buffer — sufficient to check size.
    std::promise<std::vector<uint8_t>> result_promise;
    auto result_future = result_promise.get_future();

    std::atomic<bool> callback_fired{false};
    hailo.register_callback(
        [&](uint8_t  /*camera_id*/,
            uint64_t /*timestamp_ns*/,
            std::vector<std::vector<uint8_t>> outputs)
        {
            bool expected = false;
            if (callback_fired.compare_exchange_strong(expected, true))
                result_promise.set_value(outputs.empty() ? std::vector<uint8_t>{} : outputs[0]);
        });

    if (!hailo.initialize()) {
        std::cout << "FAILED\n";
        print_result(false, "Hailo init error");
        return 1;
    }
    std::cout << "OK\n";
    std::cout << "  Expected output size : " << hailo.output_frame_size() << " bytes\n";

    // ── Write frame ───────────────────────────────────────────────────────────
    std::cout << "  Writing frame to Hailo ... ";
    std::cout.flush();

    if (!hailo.write_frame(frame_rgb.data(), /*camera_id=*/0, /*timestamp_ns=*/0)) {
        std::cout << "FAILED\n";
        print_result(false, "write_frame returned false");
        hailo.stop();
        return 1;
    }
    std::cout << "OK\n";

    // ── Wait for callback ─────────────────────────────────────────────────────
    std::cout << "  Waiting for result (timeout 5 s) ... ";
    std::cout.flush();

    const auto status = result_future.wait_for(std::chrono::seconds(5));

    if (status == std::future_status::timeout) {
        std::cout << "TIMEOUT\n";
        print_result(false, "callback did not fire within 5 s");
        hailo.stop();
        return 1;
    }
    std::cout << "received\n";

    // ── Validate output ───────────────────────────────────────────────────────
    const auto output        = result_future.get();
    const std::size_t expected_size = hailo.output_frame_size();
    const std::size_t actual_size   = output.size();

    std::cout << "  Output buffer size   : " << actual_size << " bytes";
    if (actual_size == expected_size) {
        std::cout << " ✓\n";
    } else {
        std::cout << "  (expected " << expected_size << ") ✗\n";
        print_result(false, "output size mismatch");
        hailo.stop();
        return 1;
    }

    hailo.stop();
    print_result(true);
    return 0;
}
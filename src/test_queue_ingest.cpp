/*
    test_queue_ingest.cpp
    ══════════════════════
    Verifies that N frames pushed through the inference pipeline all produce N
    callbacks — none dropped, none duplicated.

    Frames are generated as synthetic test patterns (no camera or image file
    needed) so this test can run entirely offline against real Hailo hardware.

    Usage
    ─────
        ./test_queue_ingest [hef_path] [n_frames]

        hef_path  default: /usr/share/hailo-models/yolov8s_h8.hef
        n_frames  default: 8

    Pass / fail criteria
    ─────────────────────
    PASS  all n_frames callbacks arrive within the deadline AND
          each callback carries the correct camera_id and a non-empty buffer
    FAIL  count mismatch, timeout, wrong camera_id, or empty buffer
*/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Fill a 640×640 RGB buffer with a pattern that encodes frame_index in blue.
static void make_test_frame(std::vector<uint8_t>& dst, int frame_index)
{
    constexpr int W = 640, H = 640, C = 3;
    dst.resize(W * H * C);
    const uint8_t blue = static_cast<uint8_t>(frame_index & 0xFF);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int i = (y * W + x) * C;
            dst[i + 0] = static_cast<uint8_t>((x * 255) / (W - 1));
            dst[i + 1] = static_cast<uint8_t>((y * 255) / (H - 1));
            dst[i + 2] = blue;
        }
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
    const std::string hef_path = (argc >= 2) ? argv[1]
                                             : "/usr/share/hailo-models/yolov8s_h8.hef";
    const int n_frames = (argc >= 3) ? std::stoi(argv[2]) : 8;

    if (n_frames <= 0) {
        std::cerr << "n_frames must be > 0\n";
        return 1;
    }

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: queue ingest (" << n_frames << " frames)\n"
              << "══════════════════════════════════════════\n";

    // ── Callback tracking ─────────────────────────────────────────────────────
    std::atomic<int>        received_count{0};
    std::atomic<int>        error_count{0};
    std::mutex              done_mutex;
    std::condition_variable done_cv;

    // ── Initialise Hailo ──────────────────────────────────────────────────────
    std::cout << "  Model : " << hef_path << "\n";
    std::cout << "  Initialising Hailo ... ";
    std::cout.flush();

    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t  camera_id,
            uint64_t /*timestamp_ns*/,
            std::vector<std::vector<uint8_t>> outputs)
        {
            bool ok = true;

            if (camera_id != 0) {
                std::cerr << "  callback: unexpected camera_id " << static_cast<int>(camera_id) << "\n";
                ok = false;
            }
            if (outputs.empty() || outputs[0].empty()) {
                std::cerr << "  callback: empty output buffer\n";
                ok = false;
            }
            if (!ok) ++error_count;

            const int n = ++received_count;
            std::cout << "  callback " << n << "/" << n_frames
                      << "  buffer=" << (outputs.empty() ? 0 : outputs[0].size()) << " bytes"
                      << (ok ? "" : "  [ERROR]") << "\n";

            if (n >= n_frames) {
                std::lock_guard<std::mutex> lk(done_mutex);
                done_cv.notify_one();
            }
        });

    if (!hailo.initialize()) {
        std::cout << "FAILED\n";
        print_result(false, "Hailo init error");
        return 1;
    }
    std::cout << "OK\n";

    // ── Generate and write synthetic frames directly ───────────────────────────
    // No queue or buffer pool needed — frames are written to Hailo sequentially.
    std::cout << "  Writing " << n_frames << " synthetic frames to Hailo ... \n";

    int written = 0;
    for (int i = 0; i < n_frames; ++i) {
        std::vector<uint8_t> frame;
        make_test_frame(frame, i);

        const uint64_t ts_ns = static_cast<uint64_t>(i) * 33'000'000ULL; // fake 30fps
        if (!hailo.write_frame(frame.data(), /*camera_id=*/0, ts_ns)) {
            std::cerr << "  write_frame failed at frame " << i << "\n";
            print_result(false, "write_frame failed");
            hailo.stop();
            return 1;
        }
        ++written;
    }
    std::cout << "  Wrote " << written << "/" << n_frames << " frames to Hailo\n";

    // ── Wait for all callbacks ────────────────────────────────────────────────
    const auto deadline = std::chrono::seconds(n_frames * 2);
    std::cout << "  Waiting for " << n_frames << " callbacks"
              << " (timeout " << n_frames * 2 << " s) ...\n";

    {
        std::unique_lock<std::mutex> lk(done_mutex);
        const bool all_arrived = done_cv.wait_for(lk, deadline, [&] {
            return received_count.load() >= n_frames;
        });

        if (!all_arrived) {
            std::cout << "  Timeout: received " << received_count.load()
                      << "/" << n_frames << " callbacks\n";
            print_result(false, "timeout waiting for callbacks");
            hailo.stop();
            return 1;
        }
    }

    hailo.stop();

    // ── Final verdict ─────────────────────────────────────────────────────────
    std::cout << "\n  Summary\n"
              << "    Frames written    : " << written                << "\n"
              << "    Callbacks received: " << received_count.load() << "\n"
              << "    Validation errors : " << error_count.load()    << "\n";

    const bool pass = (received_count.load() == n_frames) && (error_count.load() == 0);
    print_result(pass, pass ? "" : "count or validation mismatch");
    return pass ? 0 : 1;
}
/*
    test_queue_ingest.cpp
    ══════════════════════
    Verifies that N frames pushed into a queue are all consumed and produce N
    inference callbacks — none dropped, none duplicated.

    Frames are generated as synthetic test patterns (no camera or image file
    needed) so this test can run entirely offline against real Hailo hardware.

    Usage
    ─────
        ./test_queue_ingest [hef_path] [n_frames]

        hef_path  default: DEFAULT_HEF_PATH (set in CMakeLists.txt)
        n_frames  default: 8

    Pass / fail criteria
    ─────────────────────
    PASS  all n_frames callbacks arrive within the deadline AND
          each callback carries the correct camera_id and a non-empty buffer
    FAIL  count mismatch, timeout, wrong camera_id, or empty buffer
*/

#include "config.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "capture/frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "util/buffer_pool.hpp"
#include "util/queue.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Fill a 640×640 RGB buffer with a pattern that encodes frame_index in blue.
static void make_test_frame(uint8_t* dst, int frame_index)
{
    constexpr int W = 640, H = 640, C = 3;
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
                                             : DEFAULT_HEF_PATH;
    const int n_frames = (argc >= 3) ? std::stoi(argv[2]) : 8;

    if (n_frames <= 0) {
        std::cerr << "n_frames must be > 0\n";
        return 1;
    }

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: queue ingest (" << n_frames << " frames)\n"
              << "══════════════════════════════════════════\n";

    // ── Callback tracking ─────────────────────────────────────────────────────
    // Written on the Hailo read thread, read on the main thread — use atomic
    // counter + condition variable so main can block until all results arrive.

    std::atomic<int>        received_count{0};
    std::atomic<int>        error_count{0};   // wrong camera_id or empty buffer
    std::mutex              done_mutex;
    std::condition_variable done_cv;

    // ── Initialise Hailo ──────────────────────────────────────────────────────
    std::cout << "  Model : " << hef_path << "\n";
    std::cout << "  Initialising Hailo ... ";
    std::cout.flush();

    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t camera_id,
            uint64_t /*timestamp_ns*/,
            std::vector<uint8_t> output)
        {
            // Validate each result as it arrives.
            bool ok = true;

            if (camera_id != 0) {
                std::cerr << "  callback: unexpected camera_id " << (int)camera_id << "\n";
                ok = false;
            }
            if (output.empty()) {
                std::cerr << "  callback: empty output buffer\n";
                ok = false;
            }
            if (!ok)
                ++error_count;

            const int n = ++received_count;
            std::cout << "  callback " << n << "/" << n_frames
                      << "  buffer=" << output.size() << " bytes"
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

    // ── Build pool and queue, push synthetic frames ───────────────────────────
    constexpr std::size_t kBytesPerFrame = 640 * 640 * 3;
    BufferPool pool(static_cast<std::size_t>(n_frames) + 2, kBytesPerFrame);
    queue<FramePacket> frameQueue(static_cast<std::size_t>(n_frames) + 2);

    std::cout << "  Pushing " << n_frames << " synthetic frames into queue ... ";
    std::cout.flush();

    for (int i = 0; i < n_frames; ++i) {
        uint8_t* buf = pool.try_acquire();
        if (!buf) {
            std::cout << "FAILED (pool exhausted at frame " << i << ")\n";
            print_result(false, "BufferPool exhausted");
            hailo.stop();
            return 1;
        }
        make_test_frame(buf, i);

        FramePacket pkt;
        pkt.camera_id = 0;
        pkt.timestamp = static_cast<uint64_t>(i) * 33'000'000ULL; // fake 30fps timestamps
        pkt.data      = buf;

        frameQueue.push(pkt);
    }
    frameQueue.stop(); // no more frames; pop() will drain then return false
    std::cout << "OK\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    // Runs on the main thread, mirroring the consumer in main.cpp.
    std::cout << "  Running consumer loop ... \n";

    FramePacket pkt{};
    int written = 0;
    while (frameQueue.pop(pkt)) {
        hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);
        pool.release(pkt.data); // buffer no longer needed after write_frame returns
        ++written;
    }
    std::cout << "  Wrote " << written << "/" << n_frames << " frames to Hailo\n";

    if (written != n_frames) {
        print_result(false, "not all frames were written");
        hailo.stop();
        return 1;
    }

    // ── Wait for all callbacks ────────────────────────────────────────────────
    // Timeout = n_frames * 2 s (very generous; Hailo ~30 ms per frame)
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
              << "    Frames written   : " << written                << "\n"
              << "    Callbacks received: " << received_count.load() << "\n"
              << "    Validation errors: " << error_count.load()     << "\n";

    const bool pass = (received_count.load() == n_frames) && (error_count.load() == 0);
    print_result(pass, pass ? "" : "count or validation mismatch");
    return pass ? 0 : 1;
}
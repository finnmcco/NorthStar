/*
    test_queue_ingest.cpp
    ══════════════════════
    Pushes N synthetic frames through the queue and Hailo, asserts that
    exactly N callbacks arrive with valid data.

    Usage:  ./test_queue_ingest [hef_path] [n_frames]
*/

#include "config.hpp"
#include "inference_config.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"

static void make_test_frame(std::vector<uint8_t>& dst, int frame_index)
{
    dst.resize(INPUT_WIDTH * INPUT_HEIGHT * 3);
    const uint8_t blue = static_cast<uint8_t>(frame_index & 0xFF);
    for (int y = 0; y < INPUT_HEIGHT; ++y)
        for (int x = 0; x < INPUT_WIDTH; ++x) {
            const int i = (y * INPUT_WIDTH + x) * 3;
            dst[i + 0] = static_cast<uint8_t>((x * 255) / (INPUT_WIDTH  - 1));
            dst[i + 1] = static_cast<uint8_t>((y * 255) / (INPUT_HEIGHT - 1));
            dst[i + 2] = blue;
        }
}

static void print_result(bool pass, const std::string& detail = "")
{
    std::cout << (pass ? "\n  [PASS]\n"
                       : "\n  [FAIL]" + (detail.empty() ? "" : "  " + detail) + "\n");
}

int main(int argc, char** argv)
{
    const std::string hef_path = (argc >= 2) ? argv[1] : DEFAULT_HEF_PATH;
    const int n_frames = (argc >= 3) ? std::stoi(argv[2]) : 8;

    if (n_frames <= 0) { std::cerr << "n_frames must be > 0\n"; return 1; }

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: queue ingest (" << n_frames << " frames)\n"
              << "══════════════════════════════════════════\n";

    std::atomic<int>        received_count{0};
    std::atomic<int>        error_count{0};
    std::mutex              done_mutex;
    std::condition_variable done_cv;

    std::cout << "  Model : " << hef_path << "\n";
    std::cout << "  Initialising Hailo ... "; std::cout.flush();

    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t camera_id, uint64_t, std::vector<std::vector<uint8_t>> output)
        {
            bool ok = true;
            if (camera_id != 0) { std::cerr << "  unexpected camera_id\n"; ok = false; }
            if (output.empty()) { std::cerr << "  empty output\n";         ok = false; }
            if (!ok) ++error_count;

            const int n = ++received_count;
            std::cout << "  callback " << n << "/" << n_frames
                      << "  streams=" << output.size()
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

    queue<FramePacket> frameQueue(static_cast<std::size_t>(n_frames) + 2);

    std::cout << "  Pushing " << n_frames << " synthetic frames ... ";
    std::cout.flush();

    for (int i = 0; i < n_frames; ++i) {
        FramePacket pkt;
        pkt.camera_id = 0;
        pkt.timestamp = static_cast<uint64_t>(i) * 33'000'000ULL;
        make_test_frame(pkt.data, i);
        frameQueue.push(pkt);
    }
    frameQueue.stop();
    std::cout << "OK\n  Running consumer loop ...\n";

    FramePacket pkt{};
    int written = 0;
    while (frameQueue.pop(pkt)) {
        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);
        ++written;
    }
    std::cout << "  Wrote " << written << "/" << n_frames << " frames\n";

    if (written != n_frames) {
        print_result(false, "not all frames written");
        hailo.stop();
        return 1;
    }

    const auto deadline = std::chrono::seconds(n_frames * 2);
    std::cout << "  Waiting for " << n_frames << " callbacks ...\n";
    {
        std::unique_lock<std::mutex> lk(done_mutex);
        if (!done_cv.wait_for(lk, deadline,
                              [&]{ return received_count.load() >= n_frames; })) {
            print_result(false, "timeout");
            hailo.stop();
            return 1;
        }
    }

    hailo.stop();

    std::cout << "\n  Summary\n"
              << "    Written  : " << written                << "\n"
              << "    Received : " << received_count.load()  << "\n"
              << "    Errors   : " << error_count.load()     << "\n";

    const bool pass = (received_count.load() == n_frames) && (error_count.load() == 0);
    print_result(pass, pass ? "" : "count or validation mismatch");
    return pass ? 0 : 1;
}
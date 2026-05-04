#include "config.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "frame_packet.hpp"
#include "camera_queue.hpp"
#include "inference/hailo8_inference.hpp"

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

int main(int argc, char** argv)
{
    const std::string hef_path = (argc >= 2) ? argv[1] : DEFAULT_HEF_PATH;
    const int n_frames = (argc >= 3) ? std::stoi(argv[2]) : 8;

    if (n_frames <= 0) {
        std::cerr << "n_frames must be > 0\n";
        return 1;
    }

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: queue ingest (" << n_frames << " frames)\n"
              << "══════════════════════════════════════════\n";

    std::atomic<int>        received_count{0};
    std::atomic<int>        error_count{0};
    std::mutex              done_mutex;
    std::condition_variable done_cv;

    std::cout << "  Model : " << hef_path << "\n";
    std::cout << "  Initialising Hailo ... ";
    std::cout.flush();

    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t camera_id, uint64_t, std::vector<uint8_t> output)
        {
            bool ok = true;
            if (camera_id != 0) { std::cerr << "  unexpected camera_id\n"; ok = false; }
            if (output.empty()) { std::cerr << "  empty output\n"; ok = false; }
            if (!ok) ++error_count;

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

    queue<FramePacket> frameQueue(static_cast<std::size_t>(n_frames) + 2);

    std::cout << "  Pushing " << n_frames << " synthetic frames into queue ... ";
    std::cout.flush();

    for (int i = 0; i < n_frames; ++i) {
        FramePacket pkt;
        pkt.camera_id = 0;
        pkt.timestamp = static_cast<uint64_t>(i) * 33'000'000ULL;
        make_test_frame(pkt.data, i);
        frameQueue.push(pkt);
    }
    frameQueue.stop();
    std::cout << "OK\n";

    std::cout << "  Running consumer loop ...\n";

    FramePacket pkt{};
    int written = 0;
    while (frameQueue.pop(pkt)) {
        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);
        ++written;
    }
    std::cout << "  Wrote " << written << "/" << n_frames << " frames to Hailo\n";

    if (written != n_frames) {
        print_result(false, "not all frames written");
        hailo.stop();
        return 1;
    }

    const auto deadline = std::chrono::seconds(n_frames * 2);
    std::cout << "  Waiting for " << n_frames << " callbacks"
              << " (timeout " << n_frames * 2 << " s) ...\n";

    {
        std::unique_lock<std::mutex> lk(done_mutex);
        const bool all_arrived = done_cv.wait_for(lk, deadline, [&] {
            return received_count.load() >= n_frames;
        });

        if (!all_arrived) {
            print_result(false, "timeout waiting for callbacks");
            hailo.stop();
            return 1;
        }
    }

    hailo.stop();

    std::cout << "\n  Summary\n"
              << "    Frames written    : " << written                << "\n"
              << "    Callbacks received: " << received_count.load()  << "\n"
              << "    Validation errors : " << error_count.load()     << "\n";

    const bool pass = (received_count.load() == n_frames) && (error_count.load() == 0);
    print_result(pass, pass ? "" : "count or validation mismatch");
    return pass ? 0 : 1;
}
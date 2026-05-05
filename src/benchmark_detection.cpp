/*
    benchmark_detection.cpp  —  NorthStar throughput benchmark (live camera).

    Usage:  ./benchmark_detection
    Output: per-frame inf_ms / e2e_ms, periodic summaries with percentiles.
*/

#include "config.hpp"
#include "camera_config.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"

static constexpr int  WARMUP_FRAMES   = 30;
static constexpr int  REPORT_INTERVAL = 100;

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

static double to_ms(uint64_t ns) { return ns / 1.0e6; }

struct WriteRecord {
    uint64_t write_accepted_ns;
    uint64_t push_ns;
};

struct Stats {
    std::mutex              mutex;
    std::vector<double>     inf_ms;
    std::vector<double>     e2e_ms;
    std::atomic<int>        frame_count{0};
    std::atomic<bool>       warmed_up{false};
    Clock::time_point       wall_start;
    std::queue<WriteRecord> write_records;

    void reserve(std::size_t n) {
        std::lock_guard<std::mutex> lk(mutex);
        inf_ms.reserve(n);
        e2e_ms.reserve(n);
    }
};

static void print_stats(const Stats& s, int frames, double elapsed_s)
{
    auto report = [](const std::vector<double>& v, const char* name) {
        if (v.empty()) return;
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double mn   = *std::min_element(v.begin(), v.end());
        double mx   = *std::max_element(v.begin(), v.end());
        auto   sv   = v; std::sort(sv.begin(), sv.end());
        std::cout << "  " << name
                  << "  mean=" << mean << "ms"
                  << "  min="  << mn   << "ms"
                  << "  max="  << mx   << "ms"
                  << "  p50="  << sv[sv.size()*50/100] << "ms"
                  << "  p95="  << sv[sv.size()*95/100] << "ms"
                  << "  p99="  << sv[sv.size()*99/100] << "ms\n";
    };

    std::cout << "\n══════════════════════════════════════════\n"
              << "  BENCHMARK — " << frames << " frames\n"
              << "  Wall time : " << elapsed_s << "s\n"
              << "  Throughput: " << (frames / elapsed_s) << " fps\n"
              << "──────────────────────────────────────────\n";
    report(s.inf_ms, "Hailo inference");
    report(s.e2e_ms, "End-to-end     ");
    std::cout << "══════════════════════════════════════════\n\n";
}

int main()
{
    std::signal(SIGINT, on_sigint);

    Hailo8Inference hailo(DEFAULT_HEF_PATH);
    Stats stats;
    stats.reserve(4096);

    hailo.register_callback(
        [&stats](uint8_t /*camera_id*/,
                 uint64_t /*timestamp_ns*/,
                 std::vector<std::vector<uint8_t>> raw_output)
        {
            const uint64_t now = now_ns();
            auto detections = parse_detections(raw_output);

            std::lock_guard<std::mutex> lk(stats.mutex);
            if (stats.write_records.empty()) return;

            WriteRecord wr = stats.write_records.front();
            stats.write_records.pop();
            if (!stats.warmed_up.load()) return;

            const double inf = to_ms(now - wr.write_accepted_ns);
            const double e2e = to_ms(now - wr.push_ns);
            stats.inf_ms.push_back(inf);
            stats.e2e_ms.push_back(e2e);

            int fc = ++stats.frame_count;
            std::cout << "frame " << fc
                      << "  inf=" << inf << "ms"
                      << "  e2e=" << e2e << "ms"
                      << "  dets=" << detections.size() << "\n";

            if (fc % REPORT_INTERVAL == 0)
                print_stats(stats, fc,
                            Ms(Clock::now() - stats.wall_start).count() / 1000.0);
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    queue<FramePacket> frameQueue(FRAME_QUEUE_DEPTH);
    CameraCapture capture(frameQueue);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "Warming up (" << WARMUP_FRAMES << " frames)...\n";

    FramePacket pkt{};
    int warmup_count = 0;

    while (g_running && frameQueue.pop(pkt))
    {
        const uint64_t push_ns = pkt.timestamp;

        if (warmup_count < WARMUP_FRAMES) {
            hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);
            if (++warmup_count == WARMUP_FRAMES) {
                std::cout << "Measuring...\n\n";
                stats.wall_start = Clock::now();
                stats.warmed_up  = true;
            }
            continue;
        }

        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);
        const uint64_t write_done = now_ns();

        {
            std::lock_guard<std::mutex> lk(stats.mutex);
            stats.write_records.push({ write_done, push_ns });
        }
    }

    frameQueue.stop();
    hailo.stop();
    capture.stop();

    {
        std::lock_guard<std::mutex> lk(stats.mutex);
        if (!stats.inf_ms.empty()) {
            std::cout << "\n── Final Report ──\n";
            print_stats(stats, stats.frame_count.load(),
                        Ms(Clock::now() - stats.wall_start).count() / 1000.0);
        } else {
            std::cout << "No frames measured.\n";
        }
    }
    return 0;
}
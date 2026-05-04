/*
    benchmark_detection.cpp  —  NorthStar throughput benchmark (live camera)
    ══════════════════════════════════════════════════════════════════════════

    Measures end-to-end inference performance using the real camera pipeline.
    For a no-libcamera equivalent, use image_bench instead.

    Usage
    ─────
        ./benchmark_detection

    Output
    ──────
    Per-frame timing is printed every REPORT_INTERVAL frames and as a final
    summary.  Columns:
        inf_ms   — time from write_frame() return to callback (pure Hailo time)
        e2e_ms   — time from frame leaving the camera to callback

    Thread layout
    ─────────────
        CameraCapture worker  →  frameQueue  →  main (write_frame)
                                                        │
                                          Hailo read thread (on_result)
*/

#include "config.hpp"
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

#include "capture/camera_capture.hpp"
#include "capture/frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "util/buffer_pool.hpp"
#include "util/detection_utils.hpp"
#include "util/queue.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int  WARMUP_FRAMES   = 30;
static constexpr int  REPORT_INTERVAL = 100;
static constexpr char HEF_PATH[] = DEFAULT_HEF_PATH;

// ─────────────────────────────────────────────────────────────────────────────
//  Shutdown
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// ─────────────────────────────────────────────────────────────────────────────
//  Timing helpers
// ─────────────────────────────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

static double to_ms(uint64_t ns) { return ns / 1.0e6; }

// ─────────────────────────────────────────────────────────────────────────────
//  Per-frame write record
//
//  write_frame() returns before the Hailo result arrives, so we FIFO-queue the
//  write timestamp here.  The callback pops the matching record to compute
//  inference duration.  Safe because Hailo processes frames in-order and only
//  the main thread calls write_frame().
// ─────────────────────────────────────────────────────────────────────────────
struct WriteRecord {
    uint64_t write_accepted_ns; // timestamp when write_frame() returned
    uint64_t push_ns;           // frame timestamp set by CameraCapture
};

// ─────────────────────────────────────────────────────────────────────────────
//  Stats (callback writes, main thread prints)
// ─────────────────────────────────────────────────────────────────────────────
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
        double p50  = sv[sv.size() * 50 / 100];
        double p95  = sv[sv.size() * 95 / 100];
        double p99  = sv[sv.size() * 99 / 100];
        std::cout << "  " << name
                  << "  mean=" << mean << "ms"
                  << "  min="  << mn   << "ms"
                  << "  max="  << mx   << "ms"
                  << "  p50="  << p50  << "ms"
                  << "  p95="  << p95  << "ms"
                  << "  p99="  << p99  << "ms\n";
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

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    // ── Hailo ────────────────────────────────────────────────────────────────
    Hailo8Inference hailo(HEF_PATH);

    Stats stats;
    stats.reserve(4096);

    // Register callback BEFORE initialize() — the read thread starts inside
    // initialize() and must have the callback pointer ready on first result.
    hailo.register_callback(
        [&stats](uint8_t /*camera_id*/,
                 uint64_t /*timestamp_ns*/,
                 std::vector<uint8_t> raw_output)
        {
            const uint64_t now = now_ns();

            auto detections = parse_nms_output(raw_output);

            std::lock_guard<std::mutex> lk(stats.mutex);

            if (stats.write_records.empty())
                return; // shouldn't happen

            WriteRecord wr = stats.write_records.front();
            stats.write_records.pop();

            if (!stats.warmed_up.load())
                return; // still warming up — discard timing

            const double inf = to_ms(now - wr.write_accepted_ns);
            const double e2e = to_ms(now - wr.push_ns);

            stats.inf_ms.push_back(inf);
            stats.e2e_ms.push_back(e2e);

            int fc = ++stats.frame_count;

            std::cout << "frame " << fc
                      << "  inf=" << inf << "ms"
                      << "  e2e=" << e2e << "ms"
                      << "  dets=" << detections.size() << "\n";

            if (fc % REPORT_INTERVAL == 0) {
                double elapsed =
                    Ms(Clock::now() - stats.wall_start).count() / 1000.0;
                print_stats(stats, fc, elapsed);
            }
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    // ── Camera pipeline ───────────────────────────────────────────────────────
    constexpr std::size_t kBuffers       = 8;
    constexpr std::size_t kBytesPerFrame = 640 * 640 * 3;

    queue<FramePacket> frameQueue(4);
    BufferPool         pool(kBuffers, kBytesPerFrame);

    CameraCapture capture(frameQueue, pool, /*fps=*/30);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "Warming up (" << WARMUP_FRAMES << " frames)...\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    // Single thread: pop → write_frame → release pool buffer.
    // Hailo results arrive asynchronously via the registered callback.
    FramePacket pkt{};
    int warmup_count = 0;

    while (g_running && frameQueue.pop(pkt))
    {
        const uint64_t push_ns = pkt.timestamp; // set by CameraCapture::run()

        // Warmup: push frames through Hailo but don't record timing.
        if (warmup_count < WARMUP_FRAMES) {
            hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);
            pool.release(pkt.data);
            ++warmup_count;

            if (warmup_count == WARMUP_FRAMES) {
                std::cout << "Measuring...\n\n";
                stats.wall_start = Clock::now();
                stats.warmed_up  = true;
            }
            continue;
        }

        // Measured frames: record write timestamp for the callback.
        hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);
        const uint64_t write_done = now_ns();

        pool.release(pkt.data); // Hailo owns its copy; pool buffer is free

        {
            std::lock_guard<std::mutex> lk(stats.mutex);
            stats.write_records.push({ write_done, push_ns });
        }
    }

    // ── Shutdown and final report ─────────────────────────────────────────────
    frameQueue.stop();
    hailo.stop();   // aborts streams and joins the read thread
    capture.stop();

    {
        std::lock_guard<std::mutex> lk(stats.mutex);
        if (!stats.inf_ms.empty()) {
            double elapsed =
                Ms(Clock::now() - stats.wall_start).count() / 1000.0;
            std::cout << "\n── Final Report ──\n";
            print_stats(stats, stats.frame_count.load(), elapsed);
        } else {
            std::cout << "No frames measured (all warmup or camera stopped early).\n";
        }
    }

    return 0;
}
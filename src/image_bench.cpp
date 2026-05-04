/*
    image_bench.cpp  —  NorthStar image-folder benchmark
    ══════════════════════════════════════════════════════

    Measures end-to-end inference performance using pre-captured images
    instead of live cameras.  No libcamera dependency required.

    Usage
    ─────
        ./image_bench  <image_root>  [fps]

        image_root   Directory containing cam0/ and (optionally) cam1/ subfolders.
                     Each subfolder should hold 640×640 PNG/JPEG images named so
                     that alphabetical order matches capture order.
        fps          Simulated camera rate (default 30).  Controls how fast
                     ImageCapture pushes frames; doesn't affect Hailo throughput.

    Example
    ───────
        mkdir -p test_images/cam0 test_images/cam1
        # copy your test PNGs in ...
        ./image_bench test_images 30

    Output
    ──────
    Per-frame timing is printed every REPORT_INTERVAL frames and as a final
    summary.  Columns:
        inf_ms   — time inside Hailo (write accepted → result received)
        e2e_ms   — time from ImageCapture push to callback (queue + Hailo)
        n_dets   — number of detections above threshold

    Thread layout
    ─────────────
        ImageCapture worker  →  frameQueue  →  main (write_frame)
                                                        │
                                          Hailo read thread (on_result)
*/

#include "config.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "capture/frame_packet.hpp"
#include "capture/image_capture.hpp"
#include "inference/hailo8_inference.hpp"
#include "util/buffer_pool.hpp"
#include "util/detection_utils.hpp"
#include "util/queue.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int  WARMUP_FRAMES    = 10;
static constexpr int  REPORT_INTERVAL  = 50;
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
//  Per-frame metadata stored between write_frame() and on_result()
//  to measure inference duration.  Protected by write_mutex_; only needed
//  for the benchmark because on_result fires asynchronously.
// ─────────────────────────────────────────────────────────────────────────────
struct WriteRecord {
    uint64_t write_accepted_ns; // when write_frame() returned
    uint64_t push_ns;           // when ImageCapture pushed the frame
};

// ─────────────────────────────────────────────────────────────────────────────
//  Stats
// ─────────────────────────────────────────────────────────────────────────────
struct Stats {
    std::mutex           mutex;
    std::vector<double>  inf_ms;   // write_frame return → callback
    std::vector<double>  e2e_ms;   // queue push → callback
    std::atomic<int>     frame_count{0};
    std::atomic<bool>    warmed_up{false};
    Clock::time_point    wall_start;

    // FIFO write records so the callback can compute inf_ms.
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
int main(int argc, char** argv)
{
    std::signal(SIGINT, on_sigint);

    const std::string image_root = (argc >= 2) ? argv[1] : ".";
    const int         fps        = (argc >= 3) ? std::stoi(argv[2]) : 30;

    // ── Hailo ────────────────────────────────────────────────────────────────
    Hailo8Inference hailo(HEF_PATH);

    Stats stats;
    stats.reserve(4096);

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
                return;

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

    // ── Image capture ─────────────────────────────────────────────────────────
    constexpr std::size_t kBuffers       = 12;
    constexpr std::size_t kBytesPerFrame = 640 * 640 * 3;

    queue<FramePacket> frameQueue(8);
    BufferPool         pool(kBuffers, kBytesPerFrame);

    // loop=false: stop after one pass through the image set.
    ImageCapture capture(image_root, frameQueue, pool, fps, /*loop=*/false);
    if (!capture.start()) {
        std::cerr << "Failed to start ImageCapture\n";
        return 1;
    }

    std::cout << "Warming up (" << WARMUP_FRAMES << " frames)...\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    FramePacket pkt{};
    int warmup_count = 0;

    while (g_running && frameQueue.pop(pkt))
    {
        const uint64_t push_ns = pkt.timestamp; // set by ImageCapture::run()

        // Warmup: push frames through Hailo but don't record timing.
        if (warmup_count < WARMUP_FRAMES) {
            hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);
            pool.release(pkt.data);
            ++warmup_count;

            if (warmup_count == WARMUP_FRAMES) {
                std::cout << "Measuring...\n\n";
                stats.wall_start  = Clock::now();
                stats.warmed_up   = true;
            }
            continue;
        }

        const uint64_t write_start = now_ns();
        hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);
        const uint64_t write_done  = now_ns();

        pool.release(pkt.data);

        // Record write timing for the callback to use.
        {
            std::lock_guard<std::mutex> lk(stats.mutex);
            stats.write_records.push({ write_done, push_ns });
        }

        (void)write_start; // available if you want write-accept latency too
    }

    // ── Shutdown and final report ─────────────────────────────────────────────
    frameQueue.stop();
    hailo.stop();
    capture.stop();

    {
        std::lock_guard<std::mutex> lk(stats.mutex);
        if (!stats.inf_ms.empty()) {
            double elapsed =
                Ms(Clock::now() - stats.wall_start).count() / 1000.0;
            std::cout << "\n── Final Report ──\n";
            print_stats(stats, stats.frame_count.load(), elapsed);
        } else {
            std::cout << "No frames measured (all warmup or no images found).\n";
        }
    }

    return 0;
}
/*
    benchmark_detection.cpp  —  NorthStar throughput benchmark
    ════════════════════════════════════════════════════════════
    Captures live frames and feeds them directly to Hailo (model expects BGR).
    per-stage timing stats.  Prints an interim report every REPORT_INTERVAL
    frames and a final report on exit.

    Usage
    ─────
        ./benchmark_detection

    Thread layout
    ─────────────
        libcamera  →  CameraQueue  →  main (write_frame)
                                              │
                                 Hailo read thread (callback) → stats

    Colour
    ──────
    camera.hpp configures libcamera as BGR888.  The compiled model expects BGR
    so frames are passed directly to write_frame() with no conversion.
*/

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <csignal>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <mutex>
#include <thread>
#include <queue>

#include <opencv2/opencv.hpp>
#include <pthread.h>

#include "config.hpp"
#include "colour.hpp"
#include "inference_config.hpp"
#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int WARMUP_FRAMES    = 30;
static constexpr int REPORT_INTERVAL  = 100;

// ─────────────────────────────────────────────────────────────────────────────
//  Shutdown
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr)
        g_controller_ptr->stop_capture();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-frame timing record
// ─────────────────────────────────────────────────────────────────────────────
struct FrameTiming {
    double preprocess_ms = 0.0;  // no preprocessing (BGR sent directly)
    double inference_ms  = 0.0;  // write_frame() → callback
    double e2e_ms        = 0.0;  // frame dequeued → callback
};

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

static double to_ms(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

// ─────────────────────────────────────────────────────────────────────────────
//  Stats printer
// ─────────────────────────────────────────────────────────────────────────────
static void print_stats(int frame_count,
                         const std::vector<FrameTiming>& timings,
                         double elapsed_wall_s)
{
    auto report = [&](const char* name, auto getter) {
        std::vector<double> v;
        v.reserve(timings.size());
        for (const auto& t : timings) v.push_back(getter(t));
        std::sort(v.begin(), v.end());
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double p50  = v[v.size() * 50 / 100];
        double p95  = v[v.size() * 95 / 100];
        double p99  = v[v.size() * 99 / 100];
        std::cout << "  " << name
                  << "  mean=" << mean << "ms"
                  << "  min="  << v.front() << "ms"
                  << "  max="  << v.back()  << "ms"
                  << "  p50="  << p50  << "ms"
                  << "  p95="  << p95  << "ms"
                  << "  p99="  << p99  << "ms\n";
    };

    std::cout << "\n══════════════════════════════════════════\n"
              << "  BENCHMARK — " << frame_count << " frames\n"
              << "  Wall time  : " << elapsed_wall_s << " s\n"
              << "  Throughput : " << (frame_count / elapsed_wall_s) << " fps\n"
              << "──────────────────────────────────────────\n";
    report("Preprocess (none)", [](const FrameTiming& t){ return t.preprocess_ms; });
    report("Hailo inference  ", [](const FrameTiming& t){ return t.inference_ms;  });
    report("End-to-end       ", [](const FrameTiming& t){ return t.e2e_ms;        });
    std::cout << "══════════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    // Pin main (consumer) thread to core 2.
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(2, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    // ── Hailo ─────────────────────────────────────────────────────────────────
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    // Timing state shared between consumer thread and Hailo read thread.
    // pending_timings is a FIFO mirroring the frame order through Hailo.
    std::mutex                 timing_mutex;
    std::queue<FrameTiming*>   pending_timings;
    std::vector<FrameTiming>   completed_timings;
    std::atomic<int>           frame_count{0};
    std::atomic<bool>          wall_started{false};
    Clock::time_point          wall_start;

    completed_timings.reserve(4096);

    hailo.register_callback(
        [&](uint8_t                           /*camera_id*/,
            uint64_t                          /*timestamp_ns*/,
            std::vector<std::vector<uint8_t>> raw_outputs)
        {
            const uint64_t cb_ns = now_ns();

            // Parse detections (keeps Hailo pipeline moving).
            auto detections = parse_detections(raw_outputs);

            std::lock_guard<std::mutex> lk(timing_mutex);
            if (pending_timings.empty()) return;

            FrameTiming* t = pending_timings.front();
            pending_timings.pop();

            // Inference time = write_frame() return → callback.
            // e2e already has dequeue_ns baked in from the consumer side.
            t->inference_ms = to_ms(cb_ns) - t->inference_ms;  // see consumer loop
            t->e2e_ms       = to_ms(cb_ns - static_cast<uint64_t>(t->e2e_ms));

            const int fc = ++frame_count;

            std::cout << "Frame " << fc
                      << "  pre=" << t->preprocess_ms << "ms"
                      << "  inf=" << t->inference_ms  << "ms"
                      << "  e2e=" << t->e2e_ms        << "ms"
                      << "  dets=" << detections.size()
                      << "\n";

            completed_timings.push_back(*t);
            delete t;

            if (fc % REPORT_INTERVAL == 0 && wall_started.load()) {
                double elapsed = Ms(Clock::now() - wall_start).count() / 1000.0;
                print_stats(fc, completed_timings, elapsed);
            }
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    // ── Camera ────────────────────────────────────────────────────────────────
    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    auto& cam_q = controller.get_cam_queue();

    // ── Consumer loop ─────────────────────────────────────────────────────────
    std::cout << "Warming up for " << WARMUP_FRAMES << " frames...\n";
    int warmup_count = 0;

    while (g_running) {
        auto pkt = cam_q.pop();
        if (!pkt) break;

        // Warmup: drain frames without timing or inference.
        if (warmup_count < WARMUP_FRAMES) {
            ++warmup_count;
            if (warmup_count == WARMUP_FRAMES) {
                std::cout << "Warmup done. Measuring...\n\n";
                wall_start   = Clock::now();
                wall_started = true;
            }
            continue;
        }

        const uint64_t dq_ns = now_ns();

        // ── Prepare colour order per config.hpp ──────────────────────────────
        hailo_prepare(pkt->data);
        const double preprocess_ms = 0.0;

        // ── Build timing record and push to FIFO before write_frame() ────────
        // inference_ms is initially set to now_ns() as a start timestamp;
        // the callback converts it to a duration by subtracting from cb_ns.
        // e2e_ms stores dq_ns as a raw value; same conversion in callback.
        auto* timing          = new FrameTiming();
        timing->preprocess_ms = preprocess_ms;
        timing->inference_ms  = to_ms(now_ns());   // start timestamp (ns→ms)
        timing->e2e_ms        = static_cast<double>(dq_ns);  // raw ns

        {
            std::lock_guard<std::mutex> lk(timing_mutex);
            pending_timings.push(timing);
        }

        hailo.write_frame(pkt->data.data(), pkt->camera_id, pkt->timestamp_us * 1000ULL);
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    controller.stop_capture();
    hailo.stop();

    // ── Final report ──────────────────────────────────────────────────────────
    if (!completed_timings.empty() && wall_started.load()) {
        double elapsed = Ms(Clock::now() - wall_start).count() / 1000.0;
        std::cout << "\n── Final Report ──\n";
        print_stats(frame_count.load(), completed_timings, elapsed);
    }

    std::cout << "Done. Measured " << frame_count.load() << " frames.\n";
    return 0;
}
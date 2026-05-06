/*
    sim_session.cpp  —  NorthStar full pipeline simulation
    ═══════════════════════════════════════════════════════
    Simulates a triggered capture session (in production this would fire on a
    button press interrupt).  Captures for a fixed interval, runs every frame
    through Hailo, then prints a full report with per-frame and aggregate timing
    for every stage of the pipeline.

    Usage
    ─────
        ./sim_session [duration_seconds]   default: 5

    Note: CaptureController has an internal 5-second capture timer.  If
    duration_seconds exceeds 5 the camera will stop at 5 s regardless.

    Output — two sections
    ─────────────────────
    1. Live progress: one line per callback as it arrives.

    2. Post-session report:

        ══════════════════════════════════════════════════════════════
          SESSION  —  5.01 s   147 frames in   147 frames out
        ══════════════════════════════════════════════════════════════

          Camera capture
          ──────────────
          Frames captured : 147
          Frame rate      : 29.3 fps  (mean 34.1 ms between frames)
          Inter-frame     : min=32.1  max=38.2  p50=33.9  p95=36.8 ms

          Hailo inference
          ───────────────
          Frames inferred : 147
          Inference time  : mean=12.3  min=10.1  max=18.2  p50=12.0  p95=15.8 ms
          End-to-end      : mean=46.4  min=42.1  max=55.2  p50=45.9  p95=52.1 ms
          (end-to-end = time from frame leaving camera to callback firing)

          Per-frame log
          ─────────────
          Frame   1  inf=12.1ms  e2e=45.2ms  2 det(s)
            [0] person  91%  (0.35,0.10)-(0.65,0.95)
            [1] bottle  47%  (0.48,0.60)-(0.52,0.72)
          Frame   2  inf=11.8ms  e2e=44.9ms  0 det(s)
          ...

          Object summary
          ──────────────
          person      seen in 143 / 147 frames
          bottle      seen in  12 / 147 frames

    Timing definitions
    ──────────────────
        inter_frame_ms   time between successive frames arriving at the consumer
        inference_ms     write_frame() accepted → callback fires  (pure Hailo)
        e2e_ms           frame dequeued → callback fires  (queue wait + Hailo)

    Colour
    ──────
    camera.hpp configures libcamera as BGR888.  The compiled model expects BGR,
    so frames are passed directly to write_frame() with no conversion.
*/

#include "config.hpp"
#include "colour.hpp"
#include "inference_config.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr)
        g_controller_ptr->stop_capture();
}

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
//  Per-frame record  —  filled in two stages
//
//  Stage 1 (consumer thread):   dequeue_ns, write_ns populated in the loop.
//  Stage 2 (Hailo read thread): callback_ns and packet filled via FIFO.
// ─────────────────────────────────────────────────────────────────────────────
struct FrameRecord {
    uint32_t        frame_idx     = 0;
    uint64_t        camera_ts_ns  = 0;   // libcamera monotonic timestamp
    uint64_t        dequeue_ns    = 0;   // wall clock: frame popped from queue
    uint64_t        write_ns      = 0;   // wall clock: write_frame() returned
    uint64_t        callback_ns   = 0;   // wall clock: callback fired

    InferencePacket packet;              // detections

    double inter_frame_ms() const;       // computed in report from adjacent records
    double inference_ms()   const { return to_ms(callback_ns - write_ns);   }
    double e2e_ms()         const { return to_ms(callback_ns - dequeue_ns); }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Stats helper — percentiles from a sorted copy
// ─────────────────────────────────────────────────────────────────────────────
struct Percentiles {
    double mean, min, max, p50, p95, p99;

    static Percentiles from(std::vector<double> v)
    {
        if (v.empty()) return {};
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        Percentiles p;
        p.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
        p.min  = v.front();
        p.max  = v.back();
        p.p50  = v[n * 50 / 100];
        p.p95  = v[n * 95 / 100];
        p.p99  = v[n * 99 / 100];
        return p;
    }

    void print(const char* name) const
    {
        std::cout << "  " << name
                  << "  mean=" << std::fixed << std::setprecision(1) << mean << "ms"
                  << "  min="  << min  << "ms"
                  << "  max="  << max  << "ms"
                  << "  p50="  << p50  << "ms"
                  << "  p95="  << p95  << "ms"
                  << "  p99="  << p99  << "ms\n";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::signal(SIGINT, on_sigint);

    const int         duration_s = (argc >= 2) ? std::stoi(argv[1]) : 5;
    const std::string hef_path   = DEFAULT_HEF_PATH;

    std::cout << "══════════════════════════════════════════\n"
              << "  NorthStar  —  session simulation\n"
              << "  Duration : " << duration_s << " s\n"
              << "  Model    : " << hef_path   << "\n"
              << "══════════════════════════════════════════\n\n";

    // ── Session storage ───────────────────────────────────────────────────────
    std::mutex                   records_mutex;
    std::vector<FrameRecord>     completed_records;
    std::queue<FrameRecord*>     pending_records;  // FIFO: consumer → callback
    completed_records.reserve(duration_s * 40);

    // ── Hailo ────────────────────────────────────────────────────────────────
    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t                           camera_id,
            uint64_t                          timestamp_ns,
            std::vector<std::vector<uint8_t>> raw_outputs)
        {
            const uint64_t cb_ns = now_ns();

            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_outputs);

            std::lock_guard<std::mutex> lk(records_mutex);

            if (pending_records.empty()) return;

            FrameRecord* rec  = pending_records.front();
            pending_records.pop();

            rec->callback_ns = cb_ns;
            rec->packet      = std::move(pkt);

            const int n = static_cast<int>(completed_records.size()) + 1;

            std::cout << "  [" << std::setw(4) << n << "]"
                      << "  inf=" << std::fixed << std::setprecision(1)
                      << rec->inference_ms() << "ms"
                      << "  e2e=" << rec->e2e_ms() << "ms"
                      << "  dets=" << rec->packet.detections.size()
                      << "\n";

            completed_records.push_back(std::move(*rec));
            delete rec;
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    // ── Camera ────────────────────────────────────────────────────────────────
    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    std::cout << "Session started — capturing for " << duration_s
              << " s (Ctrl-C to stop early)\n\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    const auto deadline  = Clock::now() + std::chrono::seconds(duration_s);
    uint32_t   frame_idx = 0;

    auto& cam_q = controller.get_cam_queue();

    while (g_running && Clock::now() < deadline) {
        auto pkt = cam_q.pop();
        if (!pkt) break;

        const uint64_t dq_ns = now_ns();

        auto* rec         = new FrameRecord();
        rec->frame_idx    = ++frame_idx;
        rec->camera_ts_ns = pkt->timestamp_us * 1000ULL;
        rec->dequeue_ns   = dq_ns;

        {
            std::lock_guard<std::mutex> lk(records_mutex);
            pending_records.push(rec);
        }

        // Prepare colour order per config.hpp (NORTHSTAR_HAILO_BGR).
        hailo_prepare(pkt->data);
        hailo.write_frame(pkt->data.data(), pkt->camera_id, pkt->timestamp_us * 1000ULL);
        rec->write_ns = now_ns();
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    g_running = false;
    controller.stop_capture();
    hailo.stop(); // joins read thread — all callbacks have fired by here

    // ── Report ────────────────────────────────────────────────────────────────
    std::lock_guard<std::mutex> lk(records_mutex);

    const int n = static_cast<int>(completed_records.size());
    if (n == 0) {
        std::cout << "\nNo frames completed.\n";
        return 0;
    }

    const double wall_s =
        to_ms(completed_records.back().callback_ns -
              completed_records.front().dequeue_ns) / 1000.0;

    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "  SESSION  —  "
              << std::fixed << std::setprecision(2) << wall_s << " s  |  "
              << frame_idx << " frames captured  |  "
              << n         << " frames inferred\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    // ── Camera section ────────────────────────────────────────────────────────
    std::vector<double> inter_ms;
    inter_ms.reserve(n - 1);
    for (int i = 1; i < n; ++i)
        inter_ms.push_back(
            to_ms(completed_records[i].dequeue_ns -
                  completed_records[i-1].dequeue_ns));

    std::cout << "  Camera capture\n"
              << "  ──────────────\n"
              << "  Frames captured : " << frame_idx << "\n"
              << "  Frame rate      : "
              << std::fixed << std::setprecision(1)
              << (frame_idx / wall_s) << " fps\n";
    if (!inter_ms.empty())
        Percentiles::from(inter_ms).print("  Inter-frame    ");

    // ── Inference section ─────────────────────────────────────────────────────
    std::vector<double> inf_ms_vec, e2e_ms_vec;
    inf_ms_vec.reserve(n);
    e2e_ms_vec.reserve(n);
    for (const auto& r : completed_records) {
        inf_ms_vec.push_back(r.inference_ms());
        e2e_ms_vec.push_back(r.e2e_ms());
    }

    std::cout << "\n  Hailo inference\n"
              << "  ───────────────\n"
              << "  Frames inferred : " << n << "\n";
    Percentiles::from(inf_ms_vec).print("  Inference time ");
    Percentiles::from(e2e_ms_vec).print("  End-to-end     ");
    std::cout << "  (end-to-end = frame dequeued → callback fired)\n";

    // ── Per-frame log ─────────────────────────────────────────────────────────
    std::cout << "\n  Per-frame log\n"
              << "  ─────────────\n";

    for (int i = 0; i < n; ++i) {
        const auto& r = completed_records[i];
        std::cout << "  Frame " << std::setw(4) << (i + 1)
                  << "  inf=" << std::fixed << std::setprecision(1)
                  << r.inference_ms() << "ms"
                  << "  e2e=" << r.e2e_ms() << "ms"
                  << "  " << r.packet.detections.size() << " det(s)\n";

        for (std::size_t d = 0; d < r.packet.detections.size(); ++d) {
            const Detection& det = r.packet.detections[d];
            std::cout << "    [" << d << "] "
                      << std::left << std::setw(14)
                      << COCO_CLASSES[det.object_id]
                      << std::right
                      << std::setw(3)
                      << static_cast<int>(det.confidence * 100) << "%"
                      << "  ("
                      << std::fixed << std::setprecision(2)
                      << det.box.x_min << "," << det.box.y_min << ")-("
                      << det.box.x_max << "," << det.box.y_max << ")\n";
        }
    }

    // ── Object summary ────────────────────────────────────────────────────────
    std::vector<int> frames_with_class(N_CLASSES, 0);
    for (const auto& r : completed_records) {
        std::vector<bool> seen(N_CLASSES, false);
        for (const auto& d : r.packet.detections)
            seen[d.object_id] = true;
        for (int c = 0; c < N_CLASSES; ++c)
            if (seen[c]) ++frames_with_class[c];
    }

    std::cout << "\n  Object summary\n"
              << "  ──────────────\n";

    bool any = false;
    for (int c = 0; c < N_CLASSES; ++c) {
        if (frames_with_class[c] > 0) {
            std::cout << "  " << std::left  << std::setw(20) << COCO_CLASSES[c]
                      << std::right << std::setw(4) << frames_with_class[c]
                      << " / " << n << " frames\n";
            any = true;
        }
    }
    if (!any) std::cout << "  No objects detected.\n";

    std::cout << "══════════════════════════════════════════════════════════════\n";
    return 0;
}
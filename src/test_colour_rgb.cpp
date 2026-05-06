/*
    test_colour_rgb.cpp  —  colour test: sends RGB to Hailo
    ═════════════════════════════════════════════════════════
    Identical to sim_session but camera frames are converted BGR→RGB
    before write_frame().  Compare object summary against test_colour_bgr
    to determine which colour order the model expects.

    Usage:  ./test_colour_rgb [duration_seconds]   default: 5
*/

#include "config.hpp"
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

static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
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

struct FrameRecord {
    uint32_t        frame_idx    = 0;
    uint64_t        camera_ts_ns = 0;
    uint64_t        dequeue_ns   = 0;
    uint64_t        write_ns     = 0;
    uint64_t        callback_ns  = 0;
    InferencePacket packet;
    double inference_ms() const { return to_ms(callback_ns - write_ns);   }
    double e2e_ms()       const { return to_ms(callback_ns - dequeue_ns); }
};

struct Percentiles {
    double mean, min, max, p50, p95, p99;
    static Percentiles from(std::vector<double> v) {
        if (v.empty()) return {};
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        return { std::accumulate(v.begin(),v.end(),0.0)/n,
                 v.front(), v.back(),
                 v[n*50/100], v[n*95/100], v[n*99/100] };
    }
    void print(const char* name) const {
        std::cout << "  " << name
                  << "  mean=" << std::fixed << std::setprecision(1) << mean << "ms"
                  << "  min=" << min << "ms  max=" << max << "ms"
                  << "  p50=" << p50 << "ms  p95=" << p95 << "ms\n";
    }
};

int main(int argc, char** argv)
{
    std::signal(SIGINT, on_sigint);

    const int         duration_s = (argc >= 2) ? std::stoi(argv[1]) : 5;
    const std::string hef_path   = DEFAULT_HEF_PATH;

    std::cout << "══════════════════════════════════════════\n"
              << "  COLOUR TEST — sending RGB to Hailo\n"
              << "  Duration : " << duration_s << " s\n"
              << "  Model    : " << hef_path   << "\n"
              << "══════════════════════════════════════════\n\n";

    std::mutex               records_mutex;
    std::vector<FrameRecord> completed_records;
    std::queue<FrameRecord*> pending_records;
    completed_records.reserve(duration_s * 40);

    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t camera_id, uint64_t timestamp_ns,
            std::vector<std::vector<uint8_t>> raw_outputs)
        {
            const uint64_t cb_ns = now_ns();
            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_outputs);

            std::lock_guard<std::mutex> lk(records_mutex);
            if (pending_records.empty()) return;
            FrameRecord* rec = pending_records.front();
            pending_records.pop();
            rec->callback_ns = cb_ns;
            rec->packet      = std::move(pkt);

            const int n = static_cast<int>(completed_records.size()) + 1;
            std::cout << "  [" << std::setw(4) << n << "]"
                      << "  cam=" << static_cast<int>(rec->packet.camera_id)
                      << "  inf=" << std::fixed << std::setprecision(1)
                      << rec->inference_ms() << "ms"
                      << "  e2e=" << rec->e2e_ms() << "ms"
                      << "  dets=" << rec->packet.detections.size() << "\n";
            for (const auto& d : rec->packet.detections)
                std::cout << "        " << COCO_CLASSES[d.object_id]
                          << "  " << static_cast<int>(d.confidence * 100) << "%\n";

            completed_records.push_back(std::move(*rec));
            delete rec;
        });

    if (!hailo.initialize()) { std::cerr << "Failed to initialise Hailo\n"; return 1; }

    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    std::cout << "Capturing for " << duration_s << " s (Ctrl-C to stop early)\n\n";

    const auto deadline = Clock::now() + std::chrono::seconds(duration_s);
    uint32_t frame_idx  = 0;
    auto& cam_q = controller.get_cam_queue();

    while (g_running && Clock::now() < deadline) {
        auto pkt = cam_q.pop();
        if (!pkt) break;

        const uint64_t dq_ns = now_ns();
        auto* rec         = new FrameRecord();
        rec->frame_idx    = ++frame_idx;
        rec->camera_ts_ns = pkt->timestamp_us * 1000ULL;
        rec->dequeue_ns   = dq_ns;
        { std::lock_guard<std::mutex> lk(records_mutex); pending_records.push(rec); }

        // BGR→RGB in-place.
        uint8_t* p = pkt->data.data();
        for (std::size_t i = 0; i < pkt->data.size(); i += 3)
            std::swap(p[i], p[i + 2]);

        hailo.write_frame(p, pkt->camera_id, pkt->timestamp_us * 1000ULL);
        rec->write_ns = now_ns();
    }

    g_running = false;
    controller.stop_capture();
    hailo.stop();

    std::lock_guard<std::mutex> lk(records_mutex);
    const int n = static_cast<int>(completed_records.size());
    if (n == 0) { std::cout << "\nNo frames completed.\n"; return 0; }

    // Object summary
    std::vector<int> frames_with_class(N_CLASSES, 0);
    for (const auto& r : completed_records) {
        std::vector<bool> seen(N_CLASSES, false);
        for (const auto& d : r.packet.detections) seen[d.object_id] = true;
        for (int c = 0; c < N_CLASSES; ++c) if (seen[c]) ++frames_with_class[c];
    }

    std::cout << "\n══════════════════════════════════════════\n"
              << "  RESULT (RGB)  —  " << n << " frames\n"
              << "══════════════════════════════════════════\n";
    bool any = false;
    for (int c = 0; c < N_CLASSES; ++c) {
        if (frames_with_class[c] > 0) {
            std::cout << "  " << std::left << std::setw(20) << COCO_CLASSES[c]
                      << std::right << std::setw(4) << frames_with_class[c]
                      << " / " << n << " frames\n";
            any = true;
        }
    }
    if (!any) std::cout << "  No objects detected.\n";
    std::cout << "══════════════════════════════════════════\n";
    return 0;
}
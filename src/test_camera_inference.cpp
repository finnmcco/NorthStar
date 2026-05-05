/*
    test_camera_inference.cpp
    ══════════════════════════
    Captures from the live camera for a fixed interval, runs every frame
    through Hailo, collects InferencePackets, then prints a full report.

    Usage:  ./test_camera_inference [duration_seconds] [hef_path]
*/

#include "config.hpp"
#include "camera_config.hpp"
#include "inference_config.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

static void print_inference_packet(int frame_index, const InferencePacket& pkt)
{
    std::cout << "  Frame " << std::setw(4) << frame_index
              << "  cam="   << (int)pkt.camera_id
              << "  ts="    << pkt.timestamp << " ns"
              << "  "       << pkt.detections.size() << " detection(s)\n";

    for (std::size_t i = 0; i < pkt.detections.size(); ++i) {
        const Detection& d = pkt.detections[i];
        std::cout << "    [" << i << "]  "
                  << std::left  << std::setw(16) << COCO_CLASSES[d.object_id]
                  << std::right
                  << " conf=" << std::setw(3)
                  << static_cast<int>(d.confidence * 100) << "%"
                  << "  box=("
                  << std::fixed << std::setprecision(2)
                  << d.box.x_min << ", " << d.box.y_min << ")-("
                  << d.box.x_max << ", " << d.box.y_max << ")\n";
    }
}

int main(int argc, char** argv)
{
    std::signal(SIGINT, on_sigint);

    const int         duration_s = (argc >= 2) ? std::stoi(argv[1]) : 5;
    const std::string hef_path   = (argc >= 3) ? argv[2] : DEFAULT_HEF_PATH;

    std::cout << "══════════════════════════════════════════\n"
              << "  TEST: camera inference  (" << duration_s << " s)\n"
              << "  Model: " << hef_path << "\n"
              << "══════════════════════════════════════════\n\n";

    std::mutex                   results_mutex;
    std::vector<InferencePacket> results;
    results.reserve(duration_s * 35);

    Hailo8Inference hailo(hef_path);

    hailo.register_callback(
        [&](uint8_t                           camera_id,
            uint64_t                          timestamp_ns,
            std::vector<std::vector<uint8_t>> raw_outputs)
        {
            // parse_detections returns vector<Detection> (same type as
            // InferencePacket::detections) so we can move directly.
            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_outputs);

            std::lock_guard<std::mutex> lk(results_mutex);
            const int n = static_cast<int>(results.size()) + 1;
            results.push_back(std::move(pkt));
            std::cout << "  frame " << std::setw(4) << n
                      << "  dets=" << results.back().detections.size() << "\n";
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n"; return 1;
    }

    queue<FramePacket> frameQueue(FRAME_QUEUE_DEPTH);
    CameraCapture capture(frameQueue);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n"; return 1;
    }

    std::cout << "Capturing for " << duration_s << " s — Ctrl-C to stop early\n\n";

    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::seconds(duration_s);

    FramePacket pkt{};
    while (g_running && Clock::now() < deadline) {
        if (!frameQueue.pop(pkt)) break;
        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);
    }

    g_running = false;
    capture.stop();
    frameQueue.stop();
    hailo.stop();

    std::cout << "\n══════════════════════════════════════════════════════════════\n";

    {
        std::lock_guard<std::mutex> lk(results_mutex);

        const int    n_frames = static_cast<int>(results.size());
        const double actual_s = (n_frames > 1)
            ? static_cast<double>(
                results.back().timestamp - results.front().timestamp) / 1e9
            : static_cast<double>(duration_s);
        const double fps = (actual_s > 0) ? n_frames / actual_s : 0.0;

        std::cout << "  RESULTS — " << n_frames << " frames  ("
                  << std::fixed << std::setprecision(1) << fps << " fps)\n"
                  << "══════════════════════════════════════════════════════════════\n\n";

        for (int i = 0; i < n_frames; ++i)
            print_inference_packet(i + 1, results[i]);

        std::vector<int> class_counts(N_CLASSES, 0);
        int total_dets = 0;
        for (const auto& r : results)
            for (const auto& d : r.detections) {
                ++class_counts[d.object_id];
                ++total_dets;
            }

        std::cout << "\n──────────────────────────────────────────────────────────────\n"
                  << "  Summary: " << total_dets << " total detections across "
                  << n_frames << " frames\n\n";

        for (int c = 0; c < N_CLASSES; ++c)
            if (class_counts[c] > 0)
                std::cout << "    " << std::left << std::setw(20)
                          << COCO_CLASSES[c] << class_counts[c] << " time(s)\n";

        std::cout << "══════════════════════════════════════════════════════════════\n";
    }

    return 0;
}
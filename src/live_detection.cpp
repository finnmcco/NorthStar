/*
    live_detection.cpp
    ══════════════════
    Single-camera live detection demo.

    Thread layout
    ─────────────
        libcamera callback  →  frameQueue  →  main thread (write_frame)
                                                    │
                                       Hailo read thread (on_detection)
                                                    │
                                       on_detection() → imshow
*/

#include "config.hpp"
#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <optional>

#include <opencv2/opencv.hpp>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

static std::mutex             g_display_mutex;
static std::optional<cv::Mat> g_latest_bgr;
static std::optional<cv::Mat> g_display_frame;

static void on_detection(uint8_t              camera_id,
                          uint64_t             /*timestamp_ns*/,
                          std::vector<std::vector<uint8_t>> output)
{
    auto detections = parse_detections(output);

    for (const auto& d : detections) {
        std::cout << "[cam" << (int)camera_id << "] "
                  << COCO_CLASSES[d.class_id]
                  << " conf=" << d.score
                  << " box=("
                  << (int)(d.x_min * 640) << ","
                  << (int)(d.y_min * 640) << ")-("
                  << (int)(d.x_max * 640) << ","
                  << (int)(d.y_max * 640) << ")\n";
    }

    {
        std::lock_guard<std::mutex> lk(g_display_mutex);
        if (g_latest_bgr.has_value()) {
            cv::Mat annotated = g_latest_bgr->clone();
            draw_detections(annotated, detections);
            g_display_frame = std::move(annotated);
        }
    }
}

int main()
{
    std::signal(SIGINT, on_sigint);

    // ── Hailo ────────────────────────────────────────────────────────────────
    Hailo8Inference hailo(DEFAULT_HEF_PATH);
    hailo.register_callback(on_detection);

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    // ── Camera pipeline ───────────────────────────────────────────────────────
    queue<FramePacket> frameQueue(4);
    CameraCapture capture(frameQueue, /*fps=*/30);

    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "Live detection running — press Ctrl-C or q to quit\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    FramePacket pkt{};

    while (g_running && frameQueue.pop(pkt))
    {
        // Snapshot frame as BGR for display (frames from pipeline are BGR)
        {
            std::lock_guard<std::mutex> lk(g_display_mutex);
            cv::Mat bgr_view(640, 640, CV_8UC3, pkt.data.data(), 640 * 3);
            bgr_view.copyTo(g_latest_bgr.emplace());
        }

        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);

        {
            std::lock_guard<std::mutex> lk(g_display_mutex);
            if (g_display_frame.has_value()) {
                cv::imshow("Live Detection", *g_display_frame);
                g_display_frame.reset();
            }
        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            g_running = false;
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    capture.stop();
    frameQueue.stop();
    hailo.stop();
    cv::destroyAllWindows();

    return 0;
}
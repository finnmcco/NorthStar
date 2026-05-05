/*
    live_detection.cpp  —  Single-camera live detection demo.

    Thread layout
    ─────────────
        libcamera  →  frameQueue  →  main (write_frame)
                                           │
                              Hailo read thread (on_detection) → imshow
*/

#include "config.hpp"
#include "camera_config.hpp"
#include "inference_config.hpp"
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
                  << COCO_CLASSES[d.object_id]
                  << " conf=" << d.confidence
                  << " box=("
                  << (int)(d.box.x_min * INPUT_WIDTH)  << ","
                  << (int)(d.box.y_min * INPUT_HEIGHT) << ")-("
                  << (int)(d.box.x_max * INPUT_WIDTH)  << ","
                  << (int)(d.box.y_max * INPUT_HEIGHT) << ")\n";
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

    Hailo8Inference hailo(DEFAULT_HEF_PATH);
    hailo.register_callback(on_detection);

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

    std::cout << "Live detection running — press Ctrl-C or q to quit\n";

    FramePacket pkt{};

    while (g_running && frameQueue.pop(pkt))
    {
        {
            std::lock_guard<std::mutex> lk(g_display_mutex);
            cv::Mat bgr_view(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3,
                             pkt.data.data(), INPUT_WIDTH * 3);
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

        if (cv::waitKey(1) == 'q')
            g_running = false;
    }

    capture.stop();
    frameQueue.stop();
    hailo.stop();
    cv::destroyAllWindows();
    return 0;
}
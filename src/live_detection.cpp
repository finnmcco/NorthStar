/*
    live_detection.cpp  —  Single-camera live detection demo.

    Colour note
    ───────────
    camera.hpp configures libcamera as BGR888, so FramePacket::data is BGR.
    The compiled model expects BGR, so frames are passed directly to write_frame().
    OpenCV display also uses BGR — so no conversion is needed anywhere.

    Thread layout
    ─────────────
        libcamera  →  CameraQueue  →  main loop (write_frame)
                                            │
                               Hailo read thread (on_detection) → imshow
*/

#include "config.hpp"
#include "colour.hpp"
#include "inference_config.hpp"
#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>

#include <opencv2/opencv.hpp>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"

static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr)
        g_controller_ptr->stop_capture();
}

static std::mutex             g_display_mutex;
static std::optional<cv::Mat> g_latest_bgr;    // BGR — for OpenCV annotation
static std::optional<cv::Mat> g_display_frame;

static void on_detection(uint8_t              camera_id,
                          uint64_t             /*timestamp_ns*/,
                          std::vector<std::vector<uint8_t>> output)
{
    auto detections = parse_detections(output);

    for (const auto& d : detections) {
        std::cout << "[cam" << static_cast<int>(camera_id) << "] "
                  << COCO_CLASSES[d.object_id]
                  << " conf=" << d.confidence
                  << " box=("
                  << static_cast<int>(d.box.x_min * INPUT_WIDTH)  << ","
                  << static_cast<int>(d.box.y_min * INPUT_HEIGHT) << ")-("
                  << static_cast<int>(d.box.x_max * INPUT_WIDTH)  << ","
                  << static_cast<int>(d.box.y_max * INPUT_HEIGHT) << ")\n";
    }

    std::lock_guard<std::mutex> lk(g_display_mutex);
    if (g_latest_bgr.has_value()) {
        cv::Mat annotated = g_latest_bgr->clone();
        draw_detections(annotated, detections); // expects BGR — correct
        g_display_frame = std::move(annotated);
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

    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    std::cout << "Live detection running — press Ctrl-C or q to quit\n";

    auto& cam_q = controller.get_cam_queue();

    while (g_running) {
        auto pkt = cam_q.pop();
        if (!pkt) break;  // queue stopped

        // pkt->data is BGR — send directly to Hailo (model expects BGR).
        // Also store for display/annotation.
        {
            std::lock_guard<std::mutex> lk(g_display_mutex);
            cv::Mat bgr_view(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3,
                             pkt->data.data(), INPUT_WIDTH * 3);
            bgr_view.copyTo(g_latest_bgr.emplace());
        }

        // pkt->data is BGR — prepare colour order per config.hpp, then send.
        hailo_prepare(pkt->data);
        hailo.write_frame(pkt->data.data(), pkt->camera_id, pkt->timestamp_us * 1000ULL);

        // Display latest annotated frame if one is ready.
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

    hailo.stop();
    cv::destroyAllWindows();
    return 0;
}
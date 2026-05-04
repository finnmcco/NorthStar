/*
    live_detection.cpp
    ══════════════════
    Single-camera live detection demo.

    Thread layout
    ─────────────
        libcamera callback thread  →  frameQueue  →  main thread (write_frame)
                                                           │
                                              Hailo read thread (callback)
                                                           │
                                              on_detection() → imshow

    The main thread pops frames from the queue and pushes them into Hailo.
    The Hailo read thread fires on_detection() with parsed results, which
    draws boxes and queues the annotated frame for display.

    Display is intentionally decoupled from inference: imshow is called at
    whatever rate frames arrive, not locked to the Hailo result rate.
*/

#include "config.hpp"
#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <optional>

#include <opencv2/opencv.hpp>

#include "capture/camera_capture.hpp"
#include "capture/frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "util/buffer_pool.hpp"
#include "util/detection_utils.hpp"
#include "util/queue.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Graceful shutdown
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// ─────────────────────────────────────────────────────────────────────────────
//  Shared display state
//  g_latest_bgr   — most recent raw frame as BGR, written by consumer loop
//                   before write_frame() so the data is still valid.
//  g_display_frame — most recent annotated frame, written by callback,
//                    read by consumer loop for imshow.
//  Both protected by g_display_mutex.
// ─────────────────────────────────────────────────────────────────────────────
static std::mutex             g_display_mutex;
static std::optional<cv::Mat> g_latest_bgr;
static std::optional<cv::Mat> g_display_frame;

static void on_detection(uint8_t              camera_id,
                          uint64_t             /*timestamp_ns*/,
                          std::vector<uint8_t> output)
{
    auto detections = parse_nms_output(output);

    // Print detections to terminal
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

    // Annotate the cached BGR frame that was snapshotted in the consumer loop
    // just before write_frame() was called.  Always update g_display_frame so
    // imshow tracks the live feed even when there are no detections.
    {
        std::lock_guard<std::mutex> lk(g_display_mutex);
        if (g_latest_bgr.has_value()) {
            cv::Mat annotated = g_latest_bgr->clone();
            draw_detections(annotated, detections);
            g_display_frame = std::move(annotated);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    // ── Hailo ────────────────────────────────────────────────────────────────
    const std::string hef_path = DEFAULT_HEF_PATH;
    Hailo8Inference hailo(hef_path);
    hailo.register_callback(on_detection);

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

    std::cout << "Live detection running — press Ctrl-C or q to quit\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    // Pop frames from the queue and push into Hailo.  This is the only thread
    // calling write_frame() so no additional locking is needed.
    FramePacket pkt{};

    while (g_running && frameQueue.pop(pkt))
    {
        // Snapshot the current frame as BGR before handing it to Hailo.
        // This must happen while pkt.data is still valid (before pool.release).
        // The callback will clone this to produce its annotated display frame.
        {
            std::lock_guard<std::mutex> lk(g_display_mutex);
            cv::Mat rgb_view(640, 640, CV_8UC3, pkt.data, 640 * 3);
            cv::cvtColor(rgb_view, g_latest_bgr.emplace(), cv::COLOR_RGB2BGR);
        }

        // write_frame() blocks only until Hailo accepts the data (< 1 ms),
        // then returns.  The result arrives asynchronously via on_detection().
        hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);

        // Release the pool buffer immediately — Hailo has its own internal
        // copy of the data after write_frame() returns.
        pool.release(pkt.data);

        // Display the most recently annotated frame if one is ready.
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
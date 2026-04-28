#include <iostream>
#include <atomic>
#include <csignal>
#include <string>

#include <opencv2/opencv.hpp>

#include "capture/camera_capture.hpp"
#include "capture/frame_packet.hpp"
#include "util/queue.hpp"
#include "util/buffer_pool.hpp"
#include "util/detection_utils.hpp"
#include "inference/hailo8_inference.hpp"

// ─────────────────────────────────────────────
//  Graceful Ctrl-C shutdown
// ─────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    // ── Hailo init ──────────────────────────
    const std::string hef_path = "/usr/share/hailo-models/yolov6n_h8.hef";
    Hailo8Inference hailo(hef_path);
    if (!hailo.initialize()) {
        std::cerr << "Failed to initialize Hailo\n";
        return 1;
    }

    // ── Camera pipeline ─────────────────────
    constexpr std::size_t kBuffers       = 6;
    constexpr std::size_t kBytesPerFrame = 640 * 640 * 3;

    queue<FramePacket> frameQueue(4);
    BufferPool pool(kBuffers, kBytesPerFrame);

    CameraCapture capture(frameQueue, pool, 30);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "Live detection running — press Ctrl-C or q to quit\n";

    // ── Consumer loop ───────────────────────
    FramePacket pkt{};
    int frame_count = 0;

    while (g_running && frameQueue.pop(pkt))
    {
        cv::Mat bgr(640, 640, CV_8UC3, pkt.data, 640 * 3);  // camera = BGR

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);  // RGB for Hailo
        // ── Inference ───────────────────────
        if (!hailo.run(bgr.data)) {
            std::cerr << "Inference failed on frame " << frame_count << "\n";
            pool.release(pkt.data);
            continue;
        }

        // ── Parse ───────────────────────────
        auto detections = parse_nms_output(hailo.get_output());
        {
            const uint8_t* ptr = hailo.get_output().data();
            for (int cls = 0; cls < 80; ++cls) {
                float count_f;
                std::memcpy(&count_f, ptr, sizeof(float));
                ptr += sizeof(float);
                std::cout << "cls=" << cls << " count=" << count_f << "\n";
                ptr += 100 * sizeof(hailo_bbox_float32_t);
            }
        }
        draw_detections(bgr, detections);  // draw on BGR
        cv::imshow("Live Detection", bgr);  // display BGR
        // ── Print to terminal ───────────────
        for (const auto& d : detections) {
            int x1 = (int)(d.x_min * 640);
            int y1 = (int)(d.y_min * 640);
            int x2 = (int)(d.x_max * 640);
            int y2 = (int)(d.y_max * 640);
            std::cout << "Frame " << frame_count
                      << " [" << COCO_CLASSES[d.class_id] << "] "
                      << "conf=" << d.score
                      << " box=(" << x1 << "," << y1
                      << ")-(" << x2 << "," << y2 << ")\n";
        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            g_running = false;

        ++frame_count;
        pool.release(pkt.data);
    }

    // ── Shutdown ────────────────────────────
    capture.stop();
    frameQueue.stop();
    cv::destroyAllWindows();

    std::cout << "Stopped after " << frame_count << " frames\n";
    return 0;
}
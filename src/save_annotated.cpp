/*
    save_annotated.cpp
    ═══════════════════
    Captures from both cameras, runs inference, draws bounding boxes on every
    frame, and saves them as JPEGs for visual comparison.

    Usage
    ─────
        ./save_annotated bgr [duration_seconds]
        ./save_annotated rgb [duration_seconds]

    Output
    ──────
        ./annotated_bgr/frame_0001_cam0.jpg
        ./annotated_bgr/frame_0001_cam1.jpg
        ./annotated_rgb/frame_0001_cam0.jpg
        ...

    Each image has bounding boxes and class labels drawn on it.
    Compare the two folders to see which colour order gives correct detections.
*/

#include "config.hpp"
#include "inference_config.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"

namespace fs = std::filesystem;

static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}

using Clock = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
//  Frame record — carries the BGR image through to the callback for annotation
// ─────────────────────────────────────────────────────────────────────────────
struct FrameRecord {
    uint32_t             frame_idx  = 0;
    uint8_t              camera_id  = 0;
    std::vector<uint8_t> bgr_data;   // original BGR frame for annotation
    InferencePacket      packet;
};

int main(int argc, char** argv)
{
    if (argc < 2 || (std::string(argv[1]) != "bgr" && std::string(argv[1]) != "rgb")) {
        std::cerr << "Usage: save_annotated <bgr|rgb> [duration_seconds]\n";
        return 1;
    }

    const bool use_rgb   = (std::string(argv[1]) == "rgb");
    const int  duration_s = (argc >= 3) ? std::stoi(argv[2]) : 5;
    const std::string out_dir = use_rgb ? "annotated_rgb" : "annotated_bgr";

    fs::create_directories(out_dir);

    std::cout << "══════════════════════════════════════════\n"
              << "  SAVE ANNOTATED — sending " << (use_rgb ? "RGB" : "BGR") << " to Hailo\n"
              << "  Duration : " << duration_s << " s\n"
              << "  Output   : ./" << out_dir << "/\n"
              << "  Model    : " << DEFAULT_HEF_PATH << "\n"
              << "══════════════════════════════════════════\n\n";

    // ── Session storage ───────────────────────────────────────────────────────
    std::mutex               records_mutex;
    std::queue<FrameRecord*> pending_records;
    std::atomic<int>         saved_count{0};

    // ── Hailo ────────────────────────────────────────────────────────────────
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    hailo.register_callback(
        [&](uint8_t  camera_id,
            uint64_t timestamp_ns,
            std::vector<std::vector<uint8_t>> raw_outputs)
        {
            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_outputs);

            FrameRecord* rec = nullptr;
            {
                std::lock_guard<std::mutex> lk(records_mutex);
                if (pending_records.empty()) return;
                rec = pending_records.front();
                pending_records.pop();
            }

            rec->packet = std::move(pkt);

            // Build annotated BGR image.
            cv::Mat bgr(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3,
                        rec->bgr_data.data(), INPUT_WIDTH * 3);
            cv::Mat annotated = bgr.clone();
            draw_detections(annotated, rec->packet.detections);

            // Label each detection on the image.
            for (const auto& d : rec->packet.detections) {
                const int x1 = static_cast<int>(d.box.x_min * INPUT_WIDTH);
                const int y1 = static_cast<int>(d.box.y_min * INPUT_HEIGHT);
                const std::string label = std::string(COCO_CLASSES[d.object_id])
                    + " " + std::to_string(static_cast<int>(d.confidence * 100)) + "%";
                cv::putText(annotated, label, cv::Point(x1, std::max(y1 - 5, 10)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }

            // Build filename: frame_0001_cam0.jpg
            std::ostringstream fname;
            fname << out_dir << "/frame_"
                  << std::setw(4) << std::setfill('0') << rec->frame_idx
                  << "_cam" << static_cast<int>(rec->camera_id) << ".jpg";

            cv::imwrite(fname.str(), annotated);

            const int n = ++saved_count;
            std::cout << "  saved " << fname.str()
                      << "  dets=" << rec->packet.detections.size() << "\n";

            delete rec;
        });

    if (!hailo.initialize()) { std::cerr << "Failed to initialise Hailo\n"; return 1; }

    // ── Camera ────────────────────────────────────────────────────────────────
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

        auto* rec        = new FrameRecord();
        rec->frame_idx   = ++frame_idx;
        rec->camera_id   = pkt->camera_id;
        rec->bgr_data    = pkt->data;   // keep original BGR for annotation

        {
            std::lock_guard<std::mutex> lk(records_mutex);
            pending_records.push(rec);
        }

        if (use_rgb) {
            // BGR→RGB in-place on a copy so annotation stays BGR.
            std::vector<uint8_t> rgb_data = pkt->data;
            for (std::size_t i = 0; i < rgb_data.size(); i += 3)
                std::swap(rgb_data[i], rgb_data[i + 2]);
            hailo.write_frame(rgb_data.data(), pkt->camera_id, pkt->timestamp_us * 1000ULL);
        } else {
            hailo.write_frame(pkt->data.data(), pkt->camera_id, pkt->timestamp_us * 1000ULL);
        }
    }

    g_running = false;
    controller.stop_capture();
    hailo.stop();

    std::cout << "\nDone. Saved " << saved_count.load()
              << " annotated frames to ./" << out_dir << "/\n";
    return 0;
}
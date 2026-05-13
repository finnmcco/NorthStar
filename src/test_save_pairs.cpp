/*
    test_save_pairs.cpp  —  NorthStar pair capture test
    ═════════════════════════════════════════════════════

    Runs the full pipeline, collects the first 10 FilteredInferencePairs,
    and for each pair saves:

        pair_saves/pair_N_cam0.png   — cam0 frame with bounding boxes drawn
        pair_saves/pair_N_cam1.png   — cam1 frame with bounding boxes drawn
        pair_saves/pair_N.txt        — struct data (timestamp, object, boxes)

    Stops automatically after 10 pairs.

    Usage:
        ./test_save_pairs <vosk_model_dir> <word1> [word2 ...]
        e.g. ./test_save_pairs ../model_inf/vosk-model-small-en-us-0.15 person cup

    To remove this test:
        - Delete src/test_save_pairs.cpp
        - Remove the test_save_pairs block from CMakeLists.txt
*/

#include "config.hpp"
#include "colour.hpp"
#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "coco_lookup.hpp"
#include "pipeline.hpp"
#include "inference/hailo8_inference.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int         TARGET_PAIRS  = 10;
static constexpr std::size_t FRAME_BUF_MAX = 60;   // ~2s at 30fps per camera
static constexpr const char* OUTPUT_DIR    = "pair_saves";

// ─────────────────────────────────────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────────────────────────────────────

std::atomic<bool> g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_signal(int) {
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}

// ─────────────────────────────────────────────────────────────────────────────
//  FrameBuffer — stores recent frames per camera for retrieval by timestamp
// ─────────────────────────────────────────────────────────────────────────────

struct FrameEntry {
    uint64_t  timestamp_ns;
    cv::Mat   frame;          // BGR, 1920×1080
};

class FrameBuffer {
public:
    void push(uint64_t ts_ns, cv::Mat frame) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (buf_.size() >= FRAME_BUF_MAX) buf_.pop_front();
        buf_.push_back({ ts_ns, std::move(frame) });
    }

    // Returns the frame whose timestamp is closest to ts_ns.
    // Returns an empty Mat if the buffer is empty.
    cv::Mat find_closest(uint64_t ts_ns) const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (buf_.empty()) return {};
        const FrameEntry* best = nullptr;
        uint64_t best_delta = UINT64_MAX;
        for (const auto& e : buf_) {
            uint64_t delta = (e.timestamp_ns > ts_ns)
                           ? e.timestamp_ns - ts_ns
                           : ts_ns - e.timestamp_ns;
            if (delta < best_delta) { best_delta = delta; best = &e; }
        }
        return best ? best->frame.clone() : cv::Mat{};
    }

private:
    mutable std::mutex       mtx_;
    std::deque<FrameEntry>   buf_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Save helpers
// ─────────────────────────────────────────────────────────────────────────────

static void save_struct(int n, const FilteredInferencePair& pair) {
    std::string path = std::string(OUTPUT_DIR) + "/pair_" + std::to_string(n) + ".txt";
    std::ofstream f(path);

    f << "pair:          " << n << "\n";
    f << "timestamp_avg: " << pair.timestamp_avg << " ns\n";
    f << "object_id:     " << static_cast<int>(pair.object_id)
      << " (" << COCO_CLASSES[pair.object_id] << ")\n\n";

    f << "cam0_detections: " << pair.cam0_detections.size() << "\n";
    for (std::size_t i = 0; i < pair.cam0_detections.size(); ++i) {
        const auto& d = pair.cam0_detections[i];
        f << "  [" << i << "] conf=" << d.confidence
          << "  box=[" << d.box.x_min << " " << d.box.y_min
          << " "       << d.box.x_max << " " << d.box.y_max << "]\n";
    }

    f << "\ncam1_detections: " << pair.cam1_detections.size() << "\n";
    for (std::size_t i = 0; i < pair.cam1_detections.size(); ++i) {
        const auto& d = pair.cam1_detections[i];
        f << "  [" << i << "] conf=" << d.confidence
          << "  box=[" << d.box.x_min << " " << d.box.y_min
          << " "       << d.box.x_max << " " << d.box.y_max << "]\n";
    }

    std::printf("  saved %s\n", path.c_str());
}

static void save_annotated(int n, int cam, cv::Mat frame,
                           const std::vector<Detection>& dets)
{
    if (frame.empty()) {
        std::printf("  [warn] no frame found for pair %d cam%d\n", n, cam);
        return;
    }
    draw_detections(frame, dets);
    std::string path = std::string(OUTPUT_DIR) + "/pair_" + std::to_string(n)
                     + "_cam" + std::to_string(cam) + ".png";
    cv::imwrite(path, frame);
    std::printf("  saved %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <vosk_model_dir> <word1> [word2 ...]\n"
            "  e.g. %s ../model_inf/vosk-model-small-en-us-0.15 person cup\n",
            argv[0], argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::filesystem::create_directories(OUTPUT_DIR);
    std::printf("Output directory: %s/\n\n", OUTPUT_DIR);

    // ── Frame buffers ─────────────────────────────────────────────────────────
    FrameBuffer frame_buf_cam0, frame_buf_cam1;

    // ── Pair counter + done signal ────────────────────────────────────────────
    std::atomic<int>    pair_count{0};
    std::mutex          done_mtx;
    std::condition_variable done_cv;

    // ── DetectionFilter ───────────────────────────────────────────────────────
    DetectionFilter filter([&](FilteredInferencePair pair) {
        int n = ++pair_count;
        std::printf("\n[pair %d/%d]  object=%s  ts=%llu\n",
                    n, TARGET_PAIRS,
                    COCO_CLASSES[pair.object_id],
                    static_cast<unsigned long long>(pair.timestamp_avg));

        // Retrieve closest frames from buffer.
        cv::Mat f0 = frame_buf_cam0.find_closest(pair.timestamp_avg);
        cv::Mat f1 = frame_buf_cam1.find_closest(pair.timestamp_avg);

        save_annotated(n, 0, f0, pair.cam0_detections);
        save_annotated(n, 1, f1, pair.cam1_detections);
        save_struct(n, pair);

        if (n >= TARGET_PAIRS) {
            std::printf("\n%d pairs collected — stopping.\n", TARGET_PAIRS);
            g_running = false;
            if (g_controller_ptr) g_controller_ptr->stop_capture();
            done_cv.notify_all();
        }
    });

    // ── Mic pipeline ──────────────────────────────────────────────────────────
    Pipeline::Config mic_cfg;
    mic_cfg.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string word = argv[i];
        const uint8_t     id   = coco_id_for_word(word);
        if (id == 255) {
            std::fprintf(stderr, "Warning: \"%s\" is not a COCO class.\n", word.c_str());
        } else {
            std::printf("  Registered: \"%s\" -> COCO id %d\n", word.c_str(), id);
        }
        mic_cfg.object_list.push_back(word);
        mic_cfg.object_ids.push_back(id);
    }

    mic_cfg.on_detection = [](const DetectionResult& r) {
        std::printf("  [mic] %-12s  id=%-3d  (%s)\n",
                    r.word.c_str(), r.object_id, r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    mic_cfg.on_intent = [&filter](uint8_t id) {
        std::printf("  [mic] arming filter -> COCO id %d\n", id);
        filter.on_intent(id);
    };

    // ── Hailo inference ───────────────────────────────────────────────────────
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    hailo.register_callback(
        [&filter](uint8_t camera_id, uint64_t timestamp_ns,
                  std::vector<std::vector<uint8_t>> raw_output)
        {
            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_output);
            filter.on_inference_packet(std::move(pkt));
        });

    if (!hailo.initialize()) {
        std::fprintf(stderr, "Failed to initialise Hailo\n");
        return 1;
    }

    // ── Camera capture ────────────────────────────────────────────────────────
    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    // ── Camera consumer thread: pop → store frame → send to Hailo ────────────
    std::thread cam_thread([&]() {
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            // Store a copy of the raw BGR frame BEFORE any processing.
            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            uint64_t ts_ns = pkt->timestamp_us * 1000ULL;

            if (pkt->camera_id == 0)
                frame_buf_cam0.push(ts_ns, frame.clone());
            else
                frame_buf_cam1.push(ts_ns, frame.clone());

            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(), pkt->camera_id, ts_ns);
        }
    });

    // ── Mic pipeline ──────────────────────────────────────────────────────────
    Pipeline mic_pipeline(std::move(mic_cfg));
    mic_pipeline.start();

    std::printf("\nSpeak a registered object name to arm the filter.\n");
    std::printf("Will collect %d pairs then stop automatically.\n\n", TARGET_PAIRS);

    // Wait until 10 pairs collected or Ctrl-C.
    {
        std::unique_lock<std::mutex> lk(done_mtx);
        done_cv.wait(lk, [] { return !g_running.load(); });
    }

    mic_pipeline.stop();
    cam_thread.join();
    hailo.stop();

    std::printf("\nDone. %d pair(s) saved to %s/\n", pair_count.load(), OUTPUT_DIR);
    return 0;
}

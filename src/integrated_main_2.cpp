/*
    integrated_main.cpp  —  always-on stereo depth test
    
    No button, no mic. Captures continuously, hunts for person (COCO id 0),
    computes stereo depth on every emitted pair. Ctrl-C to exit.
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"
#include "frame_buffer.hpp"
#include "stereo_distance.hpp"
#include "inference/hailo8_inference.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

#include <opencv2/core.hpp>


// ─── Globals ─────────────────────────────────────────────────────────────────
std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static std::atomic<uint64_t> g_cam_frames_consumed{0};
static std::atomic<uint64_t> g_ir_frames_consumed{0};
static std::atomic<uint64_t> g_hailo_callbacks{0};
static std::atomic<uint64_t> g_pairs_emitted{0};

static FrameBuffer g_frame_buf_cam0;
static FrameBuffer g_frame_buf_cam1;
static StereoDepthEstimator* g_depth_ptr = nullptr;

static constexpr uint8_t TARGET_OBJECT_ID = 0;  // person

static void on_signal(int /*sig*/)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}


// ─── FilteredInferencePair callback: compute depth and print ────────────────
static void on_filtered_pair(FilteredInferencePair pair)
{
    const uint64_t n = ++g_pairs_emitted;

    /*
    std::printf("\n[filter] PAIR #%llu  object_id=%d  ts=%llu  "
                "cam0=%zu det  cam1=%zu det\n",
                static_cast<unsigned long long>(n),
                pair.object_id,
                static_cast<unsigned long long>(pair.timestamp_avg),
                pair.cam0_detections.size(),
                pair.cam1_detections.size());
    */

    if (pair.cam0_detections.empty() || pair.cam1_detections.empty()) {
        std::printf("[filter] pair missing detections on a side; skipping depth\n");
        std::fflush(stdout);
        return;
    }

    // Highest-confidence cam0 detection
    const Detection* best_cam0 = &pair.cam0_detections.front();
    for (const auto& d : pair.cam0_detections) {
        if (d.confidence > best_cam0->confidence) best_cam0 = &d;
    }

    /**
    std::printf("[filter]   cam0  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                best_cam0->confidence,
                best_cam0->box.x_min, best_cam0->box.y_min,
                best_cam0->box.x_max, best_cam0->box.y_max);

    for (const auto& d : pair.cam1_detections)
        std::printf("[filter]   cam1  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    d.confidence,
                    d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max);
    */
    
    // Only compute depth every 5th pair to reduce CPU load
    if (n % 5 != 0) {
        std::printf("[depth] skipped (pair #%llu, computing every 5th)\n",
                    static_cast<unsigned long long>(n));
        std::fflush(stdout);
        return;
    }


    // Look up frames closest to pair's timestamp
    cv::Mat left  = g_frame_buf_cam0.find_closest(pair.timestamp_avg);
    cv::Mat right = g_frame_buf_cam1.find_closest(pair.timestamp_avg);

    if (left.empty() || right.empty()) {
        std::printf("[depth] no frame in buffer (cam0_empty=%d cam1_empty=%d)\n",
                    left.empty(), right.empty());
        std::fflush(stdout);
        return;
    }

    if (!g_depth_ptr) {
        std::printf("[depth] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    auto depth_m = g_depth_ptr->compute(left, right, best_cam0->box);

    if (depth_m) {
        std::printf("[depth] Z = %.2f m  (object_id=%d)\n",
                    *depth_m, pair.object_id);
    } else {
        std::printf("[depth] could not compute (textureless / out of range / bbox outside rectified image)\n");
    }
    std::fflush(stdout);
}


int main(int /*argc*/, char* /*argv*/[])
{
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[main] NorthStar always-on stereo depth test — initialising\n");
    std::printf("[main] target: person (COCO id %d)\n", TARGET_OBJECT_ID);

    // ── Filter (armed immediately) ───────────────────────────────────────────
    DetectionFilter filter(on_filtered_pair);

    // ── Hailo inference ──────────────────────────────────────────────────────
    std::printf("[main] creating Hailo inference (hef: %s)\n", DEFAULT_HEF_PATH);
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    hailo.register_callback(
        [&filter](uint8_t camera_id,
                  uint64_t timestamp_ns,
                  std::vector<std::vector<uint8_t>> raw_output)
        {
            ++g_hailo_callbacks;

            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_output);

            filter.on_inference_packet(std::move(pkt));
        });

    std::printf("[main] initialising Hailo...\n");
    if (!hailo.initialize()) {
        std::fprintf(stderr, "[main] FATAL: Hailo initialise failed\n");
        return 1;
    }
    std::printf("[main] Hailo initialised: input=%zu bytes, output streams=%zu\n",
                hailo.input_frame_size(), hailo.num_output_streams());

    // ── Capture controller ──────────────────────────────────────────────────
    std::printf("[main] creating CaptureController\n");
    CaptureController controller;
    g_controller_ptr = &controller;

    // ── Stereo calibration ──────────────────────────────────────────────────
    std::printf("[main] loading stereo calibration\n");
    StereoDepthEstimator depth("../cam_calibration/stereo_calib_640.yaml");
    g_depth_ptr = &depth;
    std::printf("[main] stereo calibration loaded: baseline=%.3f m\n",
                depth.baseline_m());

    // ── Cam consumer thread ──────────────────────────────────────────────────
    std::thread cam_thread([&]() {
        std::printf("[cam] thread started\n");
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            const uint64_t n = ++g_cam_frames_consumed;
            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;

            // Buffer the BGR frame BEFORE hailo_prepare mutates it
            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            if (pkt->camera_id == 0) g_frame_buf_cam0.push(ts_ns, frame.clone());
            else                     g_frame_buf_cam1.push(ts_ns, frame.clone());

            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(), pkt->camera_id, ts_ns);

            if (n % 60 == 1) {
                std::printf("[cam] consumed frame #%llu  cam=%d  qsize=%zu\n",
                            static_cast<unsigned long long>(n),
                            pkt->camera_id,
                            controller.cam_queue_size());
                std::fflush(stdout);
            }
        }
        std::printf("[cam] thread exiting — %llu frames total\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()));
    });

    // ── IR consumer thread (drain only) ──────────────────────────────────────
    std::thread ir_thread([&]() {
        std::printf("[ir] thread started\n");
        auto& ir_q = controller.get_ir_queue();
        while (auto pkt = ir_q.pop()) {
            (void)pkt;
            ++g_ir_frames_consumed;
        }
        std::printf("[ir] thread exiting\n");
    });

    // ── Arm the filter and start capture ─────────────────────────────────────
    filter.on_intent(TARGET_OBJECT_ID);
    controller.start_capture();
    std::printf("[cap] capture started — filter armed for person\n");

    std::printf("\n────────────────────────────────────────────────────\n");
    std::printf(" NorthStar always-on stereo depth test.\n");
    std::printf("  - Continuously detecting person + computing depth.\n");
    std::printf("  - Ctrl-C to exit.\n");
    std::printf("────────────────────────────────────────────────────\n\n");
    std::fflush(stdout);

    // ── Main idle loop ───────────────────────────────────────────────────────
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    std::printf("\n[main] shutdown initiated\n");
    controller.stop_capture();
    controller.shutdown();
    hailo.stop();
    cam_thread.join();
    ir_thread.join();

    std::printf("\n[main] FINAL: cam_frames=%llu  ir_frames=%llu  "
                "hailo_cb=%llu  pairs=%llu\n",
                static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                static_cast<unsigned long long>(g_hailo_callbacks.load()),
                static_cast<unsigned long long>(g_pairs_emitted.load()));
    std::printf("[main] goodbye\n");
    return 0;
}
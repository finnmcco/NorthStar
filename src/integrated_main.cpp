/*
    integrated_main.cpp  —  NorthStar end-to-end perception pipeline

    Subsystems:
        Button        — gates camera + IR capture (held = on, released = off)
        Sensor Ingest — CaptureController (camera + IR)
        Inference     — Hailo8Inference (always-on, processes frames as they arrive)
        Mic Intent    — Pipeline (always-on, Vosk ASR → COCO id)
        Filter        — DetectionFilter (correlates cam0/cam1 detections)

    Output: FilteredInferencePair structs printed to stdout (placeholder for
            downstream stereo distance estimator).

    Log prefixes:
        [main]       lifecycle / shutdown
        [button]     press / release events
        [cap]        capture controller state
        [cam]        camera consumer thread
        [ir]         IR consumer thread
        [hailo]      Hailo inference callbacks
        [mic]        mic pipeline + intent
        [filter]     filtered pair output
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "coco_lookup.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"
#include "pipeline.hpp"
#include "button-driver.h"
#include "gpio.h"
#include "inference/hailo8_inference.hpp"
#include "frame_buffer.hpp"
#include "stereo_distance.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>
#include "ir_frame_buffer.hpp"
#include "ir_aligner.hpp"
#include "temp_estimator.hpp"
#include "direction_estimator.hpp"
#include "output_struct.hpp"

static IRFrameBuffer g_frame_buf_ir;
static IRAligner* g_ir_aligner_ptr = nullptr;
static TempEstimator* g_temp_estimator_ptr = nullptr;
static DirectionEstimator* g_dir_estimator_ptr = nullptr;


// ─── Globals for signal handling ─────────────────────────────────────────────
std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;
std::atomic<bool> mic_armed{false};

// ─── Counters for periodic reporting ─────────────────────────────────────────
static std::atomic<uint64_t> g_cam_frames_consumed{0};
static std::atomic<uint64_t> g_ir_frames_consumed{0};
static std::atomic<uint64_t> g_hailo_callbacks{0};
static std::atomic<uint64_t> g_pairs_emitted{0};

static FrameBuffer g_frame_buf_cam0;
static FrameBuffer g_frame_buf_cam1;
static StereoDepthEstimator* g_depth_ptr = nullptr;

static void on_signal(int /*sig*/)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}


// ─── FilteredInferencePair print callback ───────────────────────────────────
static void on_filtered_pair(FilteredInferencePair pair)
{
    const uint64_t n = ++g_pairs_emitted;
    OutputReport output_report; 
    output_report.object_id = pair.object_id;

    std::printf("\n[filter] PAIR #%llu  object_id=%-3d  ts=%-12llu  "
                "cam0=%zu det  cam1=%zu det\n",
                static_cast<unsigned long long>(n),
                pair.object_id,
                static_cast<unsigned long long>(pair.timestamp_avg),
                pair.cam0_detections.size(),
                pair.cam1_detections.size());

    if (pair.cam0_detections.empty() || pair.cam1_detections.empty()) {
        std::printf("[filter] pair missing detections on a side; skipping depth\n");
        std::fflush(stdout);
        return;
    }

    // Highest-confidence cam0 detection (filter already guarantees these are
    // all of the target class).
    const Detection* best_cam0 = &pair.cam0_detections.front();
    for (const auto& d : pair.cam0_detections) {
        if (d.confidence > best_cam0->confidence) best_cam0 = &d;
    }

    std::printf("[filter]   cam0  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                best_cam0->confidence,
                best_cam0->box.x_min, best_cam0->box.y_min,
                best_cam0->box.x_max, best_cam0->box.y_max);

    for (const auto& d : pair.cam1_detections)
        std::printf("[filter]   cam1  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    d.confidence,
                    d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max);

    // ── Retrieve frames closest to this pair's timestamp ────────────────────
    cv::Mat left  = g_frame_buf_cam0.find_closest(pair.timestamp_avg);
    cv::Mat right = g_frame_buf_cam1.find_closest(pair.timestamp_avg);

    if (left.empty() || right.empty()) {
        std::printf("[depth] no frame in buffer for pair ts=%llu  "
                    "(cam0_empty=%d cam1_empty=%d)\n",
                    static_cast<unsigned long long>(pair.timestamp_avg),
                    left.empty(), right.empty());
        std::fflush(stdout);
        return;
    }

    // ── Compute stereo depth at the cam0 bounding box ───────────────────────
    if (!g_depth_ptr) {
        std::printf("[depth] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    auto depth_m = g_depth_ptr->compute(left, right, best_cam0->box);

    if (depth_m) {
        std::printf("[depth] Z = %.2f m  (object_id=%d)\n",
                    *depth_m, pair.object_id);
        output_report.distance = *depth_m;
    } else {
        std::printf("[depth] could not compute (textureless / out of range / "
                    "bbox outside rectified image)\n");
    }
    std::fflush(stdout);


    // now find the IR frame and align it
    auto ir = g_frame_buf_ir.find_closest(pair.timestamp_avg);

    if (!ir) {
        std::printf("[ir] no IR frame in buffer for pair ts=%llu\n",
                    static_cast<unsigned long long>(pair.timestamp_avg));
        std::fflush(stdout);
        return;
    }
    if (!g_ir_aligner_ptr) {
        std::printf("[ir] aligner not initialised\n");
        std::fflush(stdout);
        return;
    }

    IRAligner::PixelRect ir_box =
        g_ir_aligner_ptr->project_bbox(best_cam0->box);

    if (!ir_box.valid) {
        std::printf("[ir] projected bbox invalid / outside IR image\n");
        std::fflush(stdout);
        return;
    }


    std::printf("[ir] projected bbox: [%d %d %d %d]\n",
                ir_box.x0, ir_box.y0, ir_box.x1, ir_box.y1);
    std::fflush(stdout);

    if (!g_temp_estimator_ptr) {
        std::printf("[temp] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    //now estimate the temperature inside the IR projected bounding box
    auto median_temp = g_temp_estimator_ptr->estimate_temp(ir_box, *ir);
    if (median_temp) {
        std::printf("[temp] median temperature in box: %.2f C\n", *median_temp);
        output_report.temp = *median_temp;
    } else {
        std::printf("[temp] could not estimate median temperature\n");
    }

    // Direction estimation:
    if (!g_dir_estimator_ptr) {
        std::printf("[direction] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    Direction direct = g_dir_estimator_ptr->get_direction(best_cam0->box);
    output_report.direction = g_dir_estimator_ptr->cvt_to_string(direct);

    const char* object_name = coco_word_for_id(output_report.object_id);

    if (!output_report.distance || !output_report.temp) {
        std::printf("[output] incomplete report: distance=%s temp=%s\n",
                    output_report.distance ? "yes" : "no",
                    output_report.temp ? "yes" : "no");
        std::fflush(stdout);
        return;
    }

    std::printf(
        "Your %s is in the %s of your vision, about %.2f metres away. It is %.2f degrees C.\n",
        object_name,
        output_report.direction.c_str(),
        *output_report.distance,
        *output_report.temp
    );
    std::fflush(stdout);
}



int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <vosk_model_dir> <word1> [word2 ...]\n"
            "  e.g. %s ../model_inf/vosk-model-small-en-us-0.15 cup person\n",
            argv[0], argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[main] NorthStar integrated pipeline — initialising\n");

    std::printf("[main] setting up GPIO\n");
    gpio::setupGpio();

    // ── Filter ───────────────────────────────────────────────────────────────
    std::printf("[main] creating DetectionFilter\n");
    DetectionFilter filter(on_filtered_pair);

    // ── Mic pipeline config ──────────────────────────────────────────────────
    std::printf("[main] configuring mic pipeline (vosk model: %s)\n", argv[1]);
    Pipeline::Config mic_cfg;
    mic_cfg.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string word = argv[i];
        const uint8_t     id   = coco_id_for_word(word);
        if (id == 255) {
            std::fprintf(stderr,
                "[main] WARNING: \"%s\" is not a COCO class — will never match.\n",
                word.c_str());
        } else {
            std::printf("[main]   registered: \"%s\" -> COCO id %d\n",
                        word.c_str(), id);
        }
        mic_cfg.object_list.push_back(word);
        mic_cfg.object_ids.push_back(id);
    }

    mic_cfg.on_detection = [](const DetectionResult& r) {
        // Only print when armed, to avoid console spam between presses.
        if (!mic_armed.load()) return;
        std::printf("[mic] heard \"%-12s\"  id=%-3d  (%s)\n",
                    r.word.c_str(), r.object_id,
                    r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    mic_cfg.on_intent = [&filter](uint8_t id) {
        if (!mic_armed.load()) {
            std::printf("[mic] intent ignored (button not held): id=%d\n", id);
            std::fflush(stdout);
            return;
        }
        std::printf("[mic] arming filter -> COCO id %d\n", id);
        std::fflush(stdout);
        filter.on_intent(id);
    };

    
    // ── Hailo inference ──────────────────────────────────────────────────────
    std::printf("[main] creating Hailo inference (hef: %s)\n", DEFAULT_HEF_PATH);
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    /*
    hailo.register_callback(
        [&filter](uint8_t camera_id,
                  uint64_t timestamp_ns,
                  std::vector<std::vector<uint8_t>> raw_output)
        {
            const uint64_t n = ++g_hailo_callbacks;

            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_output);

            // Log every callback at a low rate to avoid flooding stdout.
            if (n % 30 == 1) {
                std::printf("[hailo] callback #%llu  cam=%d  detections=%zu  "
                            "(armed=%s)\n",
                            static_cast<unsigned long long>(n),
                            camera_id,
                            pkt.detections.size(),
                            filter.is_armed() ? "yes" : "no");
                std::fflush(stdout);
            }

            filter.on_inference_packet(std::move(pkt));
        });
    */

    hailo.register_callback(
    [&filter](uint8_t camera_id,
              uint64_t timestamp_ns,
              std::vector<std::vector<uint8_t>> raw_output)
    {
        const uint64_t n = ++g_hailo_callbacks;

        InferencePacket pkt;
        pkt.camera_id  = camera_id;
        pkt.timestamp  = timestamp_ns;
        pkt.detections = parse_detections(raw_output);

        // If filter is armed, check whether any detection matches the
        // target class and log each match individually. This gives a
        // clear per-frame view of how often each camera sees the object.
        if (filter.is_armed()) {
            const uint8_t target = filter.target_object_id();
            int matches = 0;
            for (const auto& d : pkt.detections) {
                if (d.object_id == target) {
                    ++matches;
                    std::printf("[hailo] HIT cb#%llu  cam=%d  target=%d  "
                                "conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                                static_cast<unsigned long long>(n),
                                camera_id, target, d.confidence,
                                d.box.x_min, d.box.y_min,
                                d.box.x_max, d.box.y_max);
                }
            }
            if (matches > 0) std::fflush(stdout);
        }

        filter.on_inference_packet(std::move(pkt));
    });

    std::printf("[main] initialising Hailo...\n");
    if (!hailo.initialize()) {
        std::fprintf(stderr, "[main] FATAL: Hailo initialise failed\n");
        gpio::teardownGpio();
        return 1;
    }
    std::printf("[main] Hailo initialised: input=%zu bytes, output streams=%zu\n",
                hailo.input_frame_size(), hailo.num_output_streams());

    // ── Capture controller ──────────────────────────────────────────────────
    std::printf("[main] creating CaptureController\n");
    CaptureController controller;
    g_controller_ptr = &controller;

    std::printf("[main] loading stereo calibration\n");
    StereoDepthEstimator depth("../cam_calibration/stereo_calib_640.yaml");  // adjust path as needed
    g_depth_ptr = &depth;
    std::printf("[main] stereo calibration loaded: baseline=%.3f m\n", depth.baseline_m());

    // IR Aligner -------------------
    IRAligner ir_aligner("../cam_calibration/ir_alignment.yaml");
    g_ir_aligner_ptr = &ir_aligner;

    //Temp estimator ----------
    TempEstimator temp_estimator;
    g_temp_estimator_ptr = &temp_estimator;

    //Direction estimator -----
    DirectionEstimator dir_estimator;
    g_dir_estimator_ptr = &dir_estimator;

    // ── Camera consumer thread ───────────────────────────────────────────────
    std::printf("[main] spawning cam consumer thread\n");
    std::thread cam_thread([&]() {
        std::printf("[cam] thread started\n");
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            const uint64_t n = ++g_cam_frames_consumed;
            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;

            // Store a copy of the BGR frame in the per-camera buffer BEFORE
            // hailo_prepare() (which mutates the data in place).
            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            if (pkt->camera_id == 0) g_frame_buf_cam0.push(ts_ns, frame.clone());
            else                     g_frame_buf_cam1.push(ts_ns, frame.clone());

            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(),
                              pkt->camera_id,
                              ts_ns);

            if (n % 60 == 1) {
                std::printf("[cam] consumed frame #%llu  cam=%d  qsize=%zu\n",
                            static_cast<unsigned long long>(n),
                            pkt->camera_id,
                            controller.cam_queue_size());
                std::fflush(stdout);
            }
        }
        std::printf("[cam] thread exiting (queue stopped) — %llu frames total\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()));
    });

    // ── IR consumer thread ───────────────────────────────────────────────────
    std::printf("[main] spawning IR consumer thread\n");
    std::thread ir_thread([&]() {
        std::printf("[ir] thread started\n");
        auto& ir_q = controller.get_ir_queue();

        while (auto pkt = ir_q.pop()) {
            const uint64_t n = ++g_ir_frames_consumed;

            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;
            g_frame_buf_ir.push(ts_ns, pkt->temps, pkt->ambientTemp);

            if (n % 10 == 1) {
                std::printf("[ir] consumed frame #%llu  qsize=%zu  irbuf=%zu  ambient=%.2f C\n",
                            static_cast<unsigned long long>(n),
                            controller.ir_queue_size(),
                            g_frame_buf_ir.size(),
                            pkt->ambientTemp);
                std::fflush(stdout);
            }
        }

        std::printf("[ir] thread exiting (queue stopped) — %llu frames total\n",
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()));
    });

    // ── Mic pipeline (always-on) ─────────────────────────────────────────────
    std::printf("[main] starting mic pipeline\n");
    Pipeline mic_pipeline(std::move(mic_cfg));
    mic_pipeline.start();

    
    // ── Button ───────────────────────────────────────────────────────────────
    std::printf("[main] registering button callbacks\n");
    button_driver::ButtonDriver btn;

    btn.registerPressCallback([&]() {
        std::printf("\n[button] PRESS  — capture ON, mic active\n");
        std::fflush(stdout);
        mic_armed.store(true);
        controller.start_capture();
        std::printf("[cap] capture started\n");
        std::fflush(stdout);
    });

    btn.registerReleaseCallback([&]() {
        std::printf("\n[button] RELEASE  — capture OFF, mic ignored, filter reset\n");
        std::fflush(stdout);
        mic_armed.store(false);
        controller.stop_capture();
        filter.reset();
        std::printf("[cap] capture stopped\n");
        std::printf("[main] session totals: cam_frames=%llu  ir_frames=%llu  "
                    "hailo_cb=%llu  pairs=%llu\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                    static_cast<unsigned long long>(g_hailo_callbacks.load()),
                    static_cast<unsigned long long>(g_pairs_emitted.load()));
        std::fflush(stdout);
    });
    

    /*
    // ── Button ───────────────────────────────────────────────────────────────
    std::printf("[main] registering button callbacks\n");
    button_driver::ButtonDriver btn;

    static constexpr uint8_t HARDCODED_OBJECT_ID = 67;  // cell phone

    btn.registerPressCallback([&]() {
        std::printf("\n[button] PRESS  — capture ON, filter armed for cell phone (id=%d)\n",
                    HARDCODED_OBJECT_ID);
        std::fflush(stdout);
        mic_armed.store(true);
        controller.start_capture();
        filter.on_intent(HARDCODED_OBJECT_ID);
        std::printf("[cap] capture started\n");
        std::fflush(stdout);
    });

    btn.registerReleaseCallback([&]() {
        std::printf("\n[button] RELEASE  — capture OFF, filter reset\n");
        std::fflush(stdout);
        mic_armed.store(false);
        controller.stop_capture();
        filter.reset();
        std::printf("[cap] capture stopped\n");
        std::printf("[main] session totals: cam_frames=%llu  ir_frames=%llu  "
                    "hailo_cb=%llu  pairs=%llu\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                    static_cast<unsigned long long>(g_hailo_callbacks.load()),
                    static_cast<unsigned long long>(g_pairs_emitted.load()));
        std::fflush(stdout);
    });
    */

    std::printf("\n────────────────────────────────────────────────────\n");
    std::printf(" NorthStar ready.\n");
    std::printf("  - Hold button to capture frames.\n");
    std::printf("  - Speak a registered object name while held to arm filter.\n");
    std::printf("  - Ctrl-C to exit.\n");
    std::printf("────────────────────────────────────────────────────\n\n");
    std::fflush(stdout);

    // ── Main idle loop ───────────────────────────────────────────────────────
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    std::printf("\n[main] shutdown initiated\n");

    std::printf("[main] stopping capture\n");
    controller.stop_capture();

    std::printf("[main] stopping mic pipeline\n");
    mic_pipeline.stop();

    std::printf("[main] shutting down capture controller (will stop queues)\n");
    controller.shutdown();

    std::printf("[main] joining cam thread\n");
    cam_thread.join();

    std::printf("[main] joining ir thread\n");
    ir_thread.join();

    std::printf("[main] stopping Hailo\n");
    hailo.stop();

    std::printf("[main] tearing down GPIO\n");
    gpio::teardownGpio();

    std::printf("\n[main] FINAL: cam_frames=%llu  ir_frames=%llu  "
                "hailo_cb=%llu  pairs=%llu  mic_drops=%zu\n",
                static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                static_cast<unsigned long long>(g_hailo_callbacks.load()),
                static_cast<unsigned long long>(g_pairs_emitted.load()),
                mic_pipeline.queue_drops());

    std::printf("[main] goodbye\n");
    return 0;
}
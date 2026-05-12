/*
    main.cpp  —  NorthStar integrated perception pipeline
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
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

std::atomic<bool> g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_signal(int)
{
    g_running = false;
    if (g_controller_ptr)
        g_controller_ptr->stop_capture();
}

static void on_filtered_pair(FilteredInferencePair pair)
{
    std::printf("\n[pair]  object_id=%-3d  ts=%-12llu  cam0=%zu det  cam1=%zu det\n",
                pair.object_id,
                static_cast<unsigned long long>(pair.timestamp_avg),
                pair.cam0_detections.size(),
                pair.cam1_detections.size());
    
    
    for (const auto& d : pair.cam0_detections)
        std::printf("  cam0  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    d.confidence, d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max);

    for (const auto& d : pair.cam1_detections)
        std::printf("  cam1  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    d.confidence, d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max);

    std::fflush(stdout);
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <vosk_model_dir> <word1> [word2 ...]\n"
            "  e.g. %s ../model_inf/vosk-model-small-en-us-0.15 cat dog person cup\n",
            argv[0], argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // ── DetectionFilter ───────────────────────────────────────────────────────
    DetectionFilter filter(on_filtered_pair);

    // ── Mic pipeline ──────────────────────────────────────────────────────────
    Pipeline::Config mic_cfg;
    mic_cfg.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string word = argv[i];
        const uint8_t     id   = coco_id_for_word(word);

        if (id == 255) {
            std::fprintf(stderr,
                "Warning: \"%s\" is not a COCO class — will never match a detection.\n",
                word.c_str());
        } else {
            std::printf("  Registered: \"%s\" -> COCO id %d\n", word.c_str(), id);
        }

        mic_cfg.object_list.push_back(word);
        mic_cfg.object_ids.push_back(id);
    }

    mic_cfg.on_detection = [](const DetectionResult& r) {
        std::printf("  [mic] %-12s  id=%-3d  (%s)\n",
                    r.word.c_str(), r.object_id,
                    r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    mic_cfg.on_intent = [&filter](uint8_t id) {
        std::printf("  [mic] arming filter -> COCO id %d\n", id);
        filter.on_intent(id);
    };

    // ── Hailo inference ───────────────────────────────────────────────────────
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    hailo.register_callback(
        [&filter](uint8_t camera_id,
                  uint64_t timestamp_ns,
                  std::vector<std::vector<uint8_t>> raw_output)
        {
            auto detections = parse_detections(raw_output);

            std::printf("[hailo] cam=%d  ts=%llu  detections=%zu",
                        camera_id,
                        static_cast<unsigned long long>(timestamp_ns),
                        detections.size());
            for (const auto& d : detections)
                std::printf("  {id=%d conf=%.2f}", d.object_id, d.confidence);
            std::printf("\n");
            std::fflush(stdout);

            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = std::move(detections);
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

    // ── Camera consumer thread: pop frames → Hailo ────────────────────────────
    std::thread cam_thread([&]() {
        auto& cam_q = controller.get_cam_queue();
        int frame_count = 0;
        while (auto pkt = cam_q.pop()) {
            ++frame_count;
            if (frame_count % 30 == 1)
                std::printf("[cam] frame %d  camera_id=%d\n",
                            frame_count, pkt->camera_id);
            std::fflush(stdout);
            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(), pkt->camera_id,
                              pkt->timestamp_us * 1000ULL);
        }
        std::printf("[cam] thread exited — %d frames total\n", frame_count);
    });

    // ── Mic pipeline ──────────────────────────────────────────────────────────
    Pipeline mic_pipeline(std::move(mic_cfg));
    mic_pipeline.start();

    std::printf("\nNorthStar running — speak a registered object name.\n");
    std::printf("Ctrl-C to quit.\n\n");

    while (g_running)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // ── Shutdown ──────────────────────────────────────────────────────────────
    mic_pipeline.stop();
    cam_thread.join();
    hailo.stop();

    std::printf("\nMic queue drops: %zu\n", mic_pipeline.queue_drops());
    return 0;
}
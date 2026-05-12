#include "pipeline.hpp"
#include "detection_filter.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// g_running is referenced by ChunkQueue (chunk_queue.hpp) and pipeline.cpp.
std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

// ---------------------------------------------------------------------------
// COCO word -> index lookup
// Maps a plain English word to its COCO class index (0-79).
// Returns 255 if the word is not a recognised COCO class.
// ---------------------------------------------------------------------------

static uint8_t coco_id_for_word(const std::string& word) {
    static const char* const COCO_CLASSES[80] = {
        "person",        "bicycle",       "car",           "motorcycle",
        "airplane",      "bus",           "train",         "truck",
        "boat",          "traffic light", "fire hydrant",  "stop sign",
        "parking meter", "bench",         "bird",          "cat",
        "dog",           "horse",         "sheep",         "cow",
        "elephant",      "bear",          "zebra",         "giraffe",
        "backpack",      "umbrella",      "handbag",       "tie",
        "suitcase",      "frisbee",       "skis",          "snowboard",
        "sports ball",   "kite",          "baseball bat",  "baseball glove",
        "skateboard",    "surfboard",     "tennis racket", "bottle",
        "wine glass",    "cup",           "fork",          "knife",
        "spoon",         "bowl",          "banana",        "apple",
        "sandwich",      "orange",        "broccoli",      "carrot",
        "hot dog",       "pizza",         "donut",         "cake",
        "chair",         "couch",         "potted plant",  "bed",
        "dining table",  "toilet",        "tv",            "laptop",
        "mouse",         "remote",        "keyboard",      "cell phone",
        "microwave",     "oven",          "toaster",       "sink",
        "refrigerator",  "book",          "clock",         "vase",
        "scissors",      "teddy bear",    "hair drier",    "toothbrush"
    };
    for (int i = 0; i < 80; ++i)
        if (word == COCO_CLASSES[i]) return static_cast<uint8_t>(i);
    return 255;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <model_dir> <word1> [word2 ...]\n"
            "  Words must be COCO class names, e.g. cat dog person cup\n",
            argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Build object_list and parallel object_ids from command-line words.
    Pipeline::Config cfg;
    cfg.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string word = argv[i];
        const uint8_t     id   = coco_id_for_word(word);

        if (id == 255) {
            std::fprintf(stderr,
                "Warning: \"%s\" is not a COCO class name — it will never match "
                "a detection and on_intent will fire with id=0.\n",
                word.c_str());
        } else {
            std::printf("  Registered: \"%s\" -> COCO id %d\n", word.c_str(), id);
        }

        cfg.object_list.push_back(word);
        cfg.object_ids.push_back(id);
    }

    // DetectionFilter: arms on intent, pairs cam0/cam1 inference packets,
    // emits FilteredInferencePair when both cameras see the target.
    DetectionFilter filter([](FilteredInferencePair pair) {
        std::printf("\n[filter] PAIR  object_id=%-3d  ts_avg=%-12llu"
                    "  cam0_dets=%zu  cam1_dets=%zu\n",
                    pair.object_id,
                    static_cast<unsigned long long>(pair.timestamp_avg),
                    pair.cam0_detections.size(),
                    pair.cam1_detections.size());
        std::fflush(stdout);
    });

    // on_detection: log the word recognition event.
    cfg.on_detection = [](const DetectionResult& r) {
        std::printf("  [mic] %-12s  id=%-3d  (%s)\n",
                    r.word.c_str(),
                    r.object_id,
                    r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    // on_intent: arm the filter with the resolved COCO id.
    cfg.on_intent = [&filter](uint8_t id) {
        std::printf("  [mic] arming filter for COCO id %d\n", id);
        filter.on_intent(id);
    };

    try {
        Pipeline pipeline(std::move(cfg));
        pipeline.start();

        std::printf("\nListening — Ctrl+C to stop.\n\n");
        while (g_running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        pipeline.stop();
        std::printf("\nQueue drops during run: %zu\n", pipeline.queue_drops());

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }

    return 0;
}

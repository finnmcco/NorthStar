#pragma once
#include "chunk_queue.hpp"
#include "intent_detector.hpp"

#include <functional>
#include <string>
#include <thread>
#include <vector>

using DetectionCallback = std::function<void(const DetectionResult&)>;

// Fires with the resolved COCO object_id when an intent is confirmed.
// Used to arm DetectionFilter without depending on the string word.
using IntentCallback = std::function<void(uint8_t object_id)>;

// Owns and coordinates the two-thread audio pipeline.
//
// capture_thread:
//   AudioCapture (ALSA) → left ch + MIC_GAIN → Resampler (48k→16k) → ChunkQueue
//
// inference_thread:
//   ChunkQueue → IntentDetector (Vosk) → debounce → DetectionCallback
//                                                  → IntentCallback (object_id)
//
// Resampling lives in the capture thread so the queue carries compact
// 16kHz chunks and the inference thread is entirely dedicated to Vosk.

class Pipeline {
public:
    struct Config {
        std::string              alsa_device  = ::Config::ALSA_DEVICE;
        std::string              model_path;
        std::vector<std::string> object_list;

        // Parallel to object_list: COCO class indices (0–79) for each word.
        // If shorter than object_list the remainder defaults to 0.
        // Example: object_list = {"cat", "dog"}
        //          object_ids  = {15,    16}
        std::vector<uint8_t>     object_ids;

        // Fired for every confirmed detection (word + object_id + is_final).
        DetectionCallback        on_detection;

        // Optional: fired with just the COCO object_id on every confirmed
        // detection.  Wire this to DetectionFilter::on_intent().
        IntentCallback           on_intent;

        bool                     use_partials = true;
    };

    explicit Pipeline(Config cfg);
    ~Pipeline();

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void start();
    void stop();

    size_t queue_drops() const { return queue_.drops(); }

private:
    Config      cfg_;
    ChunkQueue  queue_;
    std::thread capture_thread_;
    std::thread inference_thread_;

    void capture_loop();
    void inference_loop();

    // Returns the COCO object_id for a matched word, or 0 if not found.
    uint8_t resolve_object_id(const std::string& word) const;
};

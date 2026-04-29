#pragma once
#include "chunk_queue.hpp"
#include "intent_detector.hpp"

#include <functional>
#include <string>
#include <thread>
#include <vector>

using DetectionCallback = std::function<void(const DetectionResult&)>;

// Owns and coordinates the two-thread audio pipeline.
//
// capture_thread:
//   AudioCapture (ALSA) → left ch + MIC_GAIN → Resampler (48k→16k) → ChunkQueue
//
// inference_thread:
//   ChunkQueue → IntentDetector (Vosk) → debounce → DetectionCallback
//
// Resampling lives in the capture thread so the queue carries compact
// 16kHz chunks and the inference thread is entirely dedicated to Vosk.

class Pipeline {
public:
    struct Config {
        std::string              alsa_device  = ::Config::ALSA_DEVICE;
        std::string              model_path;
        std::vector<std::string> object_list;
        DetectionCallback        on_detection;
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
};

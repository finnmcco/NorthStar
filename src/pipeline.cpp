#include "pipeline.hpp"
#include "audio_capture.hpp"
#include "resampler.hpp"

#include <cstdio>
#include <stdexcept>

extern std::atomic<bool> g_running;

Pipeline::Pipeline(Config cfg) : cfg_(std::move(cfg)) {
    if (cfg_.object_list.empty())
        throw std::invalid_argument("object_list cannot be empty");
    if (!cfg_.on_detection)
        throw std::invalid_argument("on_detection callback must be set");
    if (cfg_.model_path.empty())
        throw std::invalid_argument("model_path must be set");
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::start() {
    capture_thread_   = std::thread(&Pipeline::capture_loop,   this);
    inference_thread_ = std::thread(&Pipeline::inference_loop, this);
}

void Pipeline::stop() {
    queue_.shutdown();
    if (capture_thread_.joinable())   capture_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
}

// ---------------------------------------------------------------------------
// Capture thread: ALSA → left ch + gain → resample 48k→16k → queue
// ---------------------------------------------------------------------------

void Pipeline::capture_loop() {
    try {
        AudioCapture cap(cfg_.alsa_device);
        Resampler    resampler;

        cap.start();
        std::printf("[capture] running on %s\n", cfg_.alsa_device.c_str());

        std::vector<int16_t> chunk_48k, chunk_16k;

        while (g_running) {
            if (!cap.read_period(chunk_48k))
                continue;   // XRUN — drop period, keep going

            resampler.process(chunk_48k, chunk_16k);
            queue_.push(std::vector<int16_t>(chunk_16k));
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[capture] error: %s\n", e.what());
        g_running = false;
    }

    queue_.shutdown();
    std::printf("[capture] stopped\n");
}

// ---------------------------------------------------------------------------
// Inference thread: queue → Vosk → debounce → callback
// ---------------------------------------------------------------------------

void Pipeline::inference_loop() {
    try {
        IntentDetector detector(cfg_.model_path, cfg_.object_list);
        std::printf("[infer] running\n");

        std::string last_word;
        int         streak = 0;

        std::vector<int16_t> chunk;
        while (queue_.pop(chunk)) {
            auto result = detector.feed(chunk);

            if (!result) {
                last_word.clear();
                streak = 0;
                continue;
            }

            if (result->is_final) {
                last_word.clear();
                streak = 0;
                cfg_.on_detection(*result);

            } else if (cfg_.use_partials) {
                if (result->word == last_word) {
                    if (++streak == ::Config::DEBOUNCE_COUNT)
                        cfg_.on_detection(*result);
                } else {
                    last_word = result->word;
                    streak    = 1;
                }
            }
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[infer] error: %s\n", e.what());
        g_running = false;
    }

    std::printf("[infer] stopped\n");
}

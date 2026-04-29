#pragma once
#include "config.hpp"

#include <vosk_api.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct DetectionResult {
    std::string word;
    bool        is_final;  // true = utterance boundary confirmed by Vosk
};

// Wraps a Vosk grammar-restricted recogniser.
//
// Grammar: JSON array of object words + "[unk]" as a catch-all bucket.
// "[unk]" prevents force-matching background noise to the nearest object word.
//
// Two detection paths:
//   Final   — committed after Vosk detects trailing silence. High confidence.
//   Partial — mid-utterance hypothesis. Lower confidence, lower latency.
//             Debounce is applied by Pipeline before firing the callback.

class IntentDetector {
public:
    IntentDetector(const std::string&              model_path,
                   const std::vector<std::string>& object_list);
    ~IntentDetector();

    IntentDetector(const IntentDetector&)            = delete;
    IntentDetector& operator=(const IntentDetector&) = delete;

    // Feed one chunk of Config::VOSK_RATE int16 audio.
    // Returns a DetectionResult on match, std::nullopt otherwise.
    std::optional<DetectionResult> feed(const std::vector<int16_t>& audio);

    void reset();

private:
    VoskModel*               model_      = nullptr;
    VoskRecognizer*          recognizer_ = nullptr;
    std::vector<std::string> object_list_;

    static std::string build_grammar(const std::vector<std::string>& words);
    static std::string extract_field(const char* json, const char* field);
};

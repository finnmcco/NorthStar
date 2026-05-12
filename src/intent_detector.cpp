#include "intent_detector.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>

// Vosk returns small JSON blobs: {"text" : "cat"} or {"partial" : "dog"}
// Whitespace around ':' varies by build — we search both forms.
std::string IntentDetector::extract_field(const char* json, const char* field) {
    if (!json || !field) return {};
    for (const char* sep : {" : \"", ": \""}) {
        std::string key = std::string("\"") + field + "\"" + sep;
        const char* p = std::strstr(json, key.c_str());
        if (!p) continue;
        p += key.size();
        const char* e = std::strchr(p, '"');
        if (!e) continue;
        return std::string(p, e);
    }
    return {};
}

std::string IntentDetector::build_grammar(const std::vector<std::string>& words) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) oss << ", ";
        oss << '"' << words[i] << '"';
    }
    oss << ", \"[unk]\"]";
    return oss.str();
}

IntentDetector::IntentDetector(const std::string&              model_path,
                               const std::vector<std::string>& object_list)
    : object_list_(object_list)
{
    vosk_set_log_level(-1);

    model_ = vosk_model_new(model_path.c_str());
    if (!model_)
        throw std::runtime_error("Failed to load Vosk model: " + model_path);

    std::string grammar = build_grammar(object_list);
    recognizer_ = vosk_recognizer_new_grm(model_, Config::VOSK_RATE, grammar.c_str());
    if (!recognizer_)
        throw std::runtime_error("Failed to create Vosk recognizer");

    vosk_recognizer_set_words(recognizer_, 0);
}

IntentDetector::~IntentDetector() {
    if (recognizer_) vosk_recognizer_free(recognizer_);
    if (model_)      vosk_model_free(model_);
}

void IntentDetector::reset() {
    vosk_recognizer_reset(recognizer_);
}

std::optional<DetectionResult> IntentDetector::feed(const std::vector<int16_t>& audio) {
    int final_flag = vosk_recognizer_accept_waveform_s(
        recognizer_, audio.data(), static_cast<int>(audio.size()));

    auto match = [&](const std::string& text) -> std::optional<std::string> {
        if (text.empty() || text == "[unk]") return std::nullopt;
        for (const auto& obj : object_list_)
            if (text.find(obj) != std::string::npos)
                return obj;
        return std::nullopt;
    };

    if (final_flag) {
        std::string text = extract_field(vosk_recognizer_result(recognizer_), "text");
        if (auto hit = match(text))
            return DetectionResult{*hit, 0, true};
    } else {
        std::string partial = extract_field(
            vosk_recognizer_partial_result(recognizer_), "partial");
        if (auto hit = match(partial))
            return DetectionResult{*hit, 0, false};
    }

    return std::nullopt;
}

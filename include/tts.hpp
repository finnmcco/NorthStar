#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// tts.hpp  –  Piper TTS engine interface
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>
#include <vector>

// Config is a free struct so it is fully defined before TTSEngine uses it
struct TTSConfig {
    std::string voice_path;
    std::string piper_path;
    // espeak-ng phoneme data directory.  Piper bundles espeak-ng but does not
    // set this itself; without it synthesis silently fails on most distros.
    std::string espeak_data_path = "/usr/lib/aarch64-linux-gnu/espeak-ng-data";
    float length_scale     = 1.0f;
    float sentence_silence = 0.08f;
    float noise_scale      = 0.667f;
    float noise_w          = 0.8f;
};

class TTSEngine {
public:
    using Config = TTSConfig;

    explicit TTSEngine(Config cfg = Config{});

    // synthesis
    std::vector<uint8_t>  synthesise_wav(const std::string& text);
    std::vector<int16_t>  synthesise(const std::string& text);

    // format
    [[nodiscard]] uint32_t sample_rate() const noexcept { return 22'050; }
    [[nodiscard]] uint8_t  channels()    const noexcept { return 1; }

    // diagnostics
    [[nodiscard]] bool               ready()      const noexcept;
    [[nodiscard]] const std::string& last_error() const noexcept { return err_; }
    [[nodiscard]] const std::string& piper_path() const noexcept { return cfg_.piper_path; }
    [[nodiscard]] const std::string& voice_path() const noexcept { return cfg_.voice_path; }

    // static discovery
    static std::string find_piper();
    static std::string find_default_voice();

private:
    Config      cfg_;
    std::string err_;
};
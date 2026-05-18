#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// speech_aggregator.hpp  –  seam between on_filtered_pair() and the TTS stack
//
// Today: speaks the sentence from the first qualifying report in a session,
//        drops the rest until reset() is called by the next button press.
//
// Future: this is where the multi-pair agreement logic will sit.  It will
//         buffer reports, check that a few of them agree on key metrics
//         (distance, temperature, direction) within tolerance, and only then
//         trigger speech.  on_report() is given the full OutputReport for
//         exactly this reason — the agreement function needs raw numbers,
//         not the rendered sentence.
//
// Threading: on_report() is called from the DetectionFilter's emitting
//            thread.  reset() is called from the GPIO button thread.
//            spoken_ is atomic and speak() is serialised by speech_mutex_
//            so overlapping calls across rapid press/release cycles cannot
//            re-enter Piper or libpulse.
// ─────────────────────────────────────────────────────────────────────────────

#include "tts.hpp"
#include "speaker.hpp"
#include "output_struct.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

class SpeechAggregator {
public:
    explicit SpeechAggregator(double gain = 1.0, bool verbose = true)
        : gain_(gain), verbose_(verbose)
    {
        if (gain_ <= 0.0 || !std::isfinite(gain_)) gain_ = 1.0;

        // Bring up TTS.  Any failure here is logged loudly at startup but
        // is non-fatal: the integrated pipeline keeps the stdout sentence
        // working as a text-only fallback.
        if (!tts_.ready()) {
            std::fprintf(stderr,
                "[speech] DISABLED: TTS not ready — %s\n",
                tts_.last_error().c_str());
            return;
        }
        if (!spk_.open()) {
            std::fprintf(stderr,
                "[speech] DISABLED: speaker open failed — %s\n",
                spk_.last_error().c_str());
            return;
        }
        enabled_ = true;
        log("[speech] ENABLED\n");
        log("[speech]   piper : %s\n", tts_.piper_path().c_str());
        log("[speech]   voice : %s\n", tts_.voice_path().c_str());
        log("[speech]   gain  : %.2fx\n", gain_);
        std::fflush(stdout);
    }

    SpeechAggregator(const SpeechAggregator&)            = delete;
    SpeechAggregator& operator=(const SpeechAggregator&) = delete;

    // Hand the next finalised report + its rendered sentence to the
    // aggregator.  First call per session speaks; subsequent calls drop.
    void on_report(const OutputReport& report, const std::string& sentence) {
        (void)report;  // reserved for future agreement logic
        if (!enabled_)              return;
        if (spoken_.exchange(true)) {
            log("[speech] drop: already spoken this session\n");
            return;
        }
        speak(sentence);
    }

    // Called from the button press callback to begin a new session.
    // This only clears the per-session "already spoken" state.  Call
    // interrupt() first when the physical button should stop current audio.
    void reset() noexcept {
        spoken_.store(false);
    }

    // Stop any speech that is currently being synthesised/played or already
    // queued in the audio server. This is intentionally safe to call from the
    // GPIO button thread.
    void interrupt() noexcept {
        play_generation_.fetch_add(1, std::memory_order_acq_rel);
        spk_.flush();
        log("[speech] interrupted\n");
        std::fflush(stdout);
    }

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] double gain() const noexcept { return gain_; }

    [[nodiscard]] bool already_spoken() const noexcept {
        return spoken_.load(std::memory_order_acquire);
    }

private:
    static int16_t read_i16le(const uint8_t* p) {
        return static_cast<int16_t>(static_cast<uint16_t>(p[0]) |
                                    (static_cast<uint16_t>(p[1]) << 8));
    }

    static void write_i16le(uint8_t* p, int16_t v) {
        const auto u = static_cast<uint16_t>(v);
        p[0] = static_cast<uint8_t>(u & 0xFFu);
        p[1] = static_cast<uint8_t>((u >> 8) & 0xFFu);
    }

    bool apply_gain(std::vector<uint8_t>& wav) {
        if (std::abs(gain_ - 1.0) < 0.001) return true;

        WavInfo info{};
        if (!Speaker::parse_wav_header(wav.data(), wav.size(), info)) {
            std::fprintf(stderr, "[speech] gain skipped: could not parse WAV header\n");
            return false;
        }
        if (info.bits_per_sample != 16) {
            std::fprintf(stderr, "[speech] gain skipped: expected 16-bit PCM WAV\n");
            return false;
        }

        const size_t avail   = wav.size() - info.data_offset;
        const size_t nbytes  = std::min<size_t>(info.data_size, avail);
        const size_t samples = nbytes / sizeof(int16_t);
        size_t clipped = 0;

        uint8_t* pcm = wav.data() + info.data_offset;
        for (size_t i = 0; i < samples; ++i) {
            const int16_t in = read_i16le(pcm + i * 2);
            double y = static_cast<double>(in) * gain_;
            if (y > 32767.0) { y = 32767.0; ++clipped; }
            if (y < -32768.0) { y = -32768.0; ++clipped; }
            write_i16le(pcm + i * 2, static_cast<int16_t>(std::lrint(y)));
        }

        log("[speech] applied gain %.2fx to %zu samples (clipped=%zu)\n",
            gain_, samples, clipped);
        return true;
    }

    template <typename... Args>
    void log(const char* fmt, Args... args) const {
        if (!verbose_) return;
        std::printf(fmt, args...);
    }

    // Synthesise and play on the calling thread.  Mutex prevents the
    // pathological case of a rapid release→press triggering a second
    // on_report() while the first is still inside drain().
    void speak(const std::string& text) {
        std::lock_guard<std::mutex> lk(speech_mutex_);
        log("[speech] speaking: \"%s\"\n", text.c_str());
        std::fflush(stdout);

        auto wav = tts_.synthesise_wav(text);
        if (wav.empty()) {
            std::fprintf(stderr,
                "[speech] synthesise failed: %s\n",
                tts_.last_error().c_str());
            return;
        }

        apply_gain(wav);

        if (!spk_.play_wav(wav)) {
            std::fprintf(stderr,
                "[speech] play_wav failed: %s\n",
                spk_.last_error().c_str());
        }
    }

    TTSEngine         tts_{};
    Speaker           spk_{};
    double            gain_ = 1.0;
    bool              verbose_ = true;
    std::atomic<bool> spoken_{false};
    std::atomic<bool> enabled_{false};
    std::atomic<uint64_t> play_generation_{0};
    std::mutex        speech_mutex_;
};

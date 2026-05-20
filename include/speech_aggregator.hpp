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
//
// TTS change (persistent Piper):
//   TTSEngine is now constructed with explicit piper_bin / model_path /
//   sample_rate arguments passed through from main().  The subprocess is
//   spawned once at construction; synthesise() reuses it for every utterance,
//   avoiding the ~2.4 s per-call model-load cost of the old std::system()
//   design.  apply_gain() now works directly on the int16_t PCM vector
//   returned by synthesise() rather than on a WAV byte buffer.
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
    // Construct with explicit Piper paths so TTSEngine spawns the subprocess
    // exactly once here rather than per utterance.
    //
    // @param gain        Linear PCM gain applied before playback (1.0 = unity).
    // @param verbose     If false, suppress all [speech] log lines.
    // @param piper_bin   Full path to the piper executable.
    // @param model_path  Full path to the .onnx voice model.
    // @param sample_rate Expected sample rate of the model (default 22050).
    explicit SpeechAggregator(double      gain        = 1.0,
                               bool        verbose     = true,
                               const char* piper_bin   = "/home/dst5/NSJamie/NorthStar/.venv/bin/piper",
                               const char* model_path  = "/home/dst5/NSJamie/NorthStar/model_inf/en_US-lessac-medium.onnx",
                               int         sample_rate = 22050)
        : tts_(piper_bin, model_path, sample_rate)
        , gain_(gain)
        , verbose_(verbose)
        , piper_bin_(piper_bin)
        , model_path_(model_path)
    {
        if (gain_ <= 0.0 || !std::isfinite(gain_)) gain_ = 1.0;

        // Bring up TTS.  Any failure here is logged loudly at startup but
        // is non-fatal: the integrated pipeline keeps the stdout sentence
        // working as a text-only fallback.
        if (!tts_.ready()) {
            std::fprintf(stderr,
                "[speech] DISABLED: TTS not ready — %s\n",
                tts_.error_message().c_str());
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
        log("[speech]   piper : %s\n", piper_bin_.c_str());
        log("[speech]   model : %s\n", model_path_.c_str());
        log("[speech]   rate  : %d Hz\n", tts_.sample_rate());
        log("[speech]   gain  : %.2fx\n", gain_);
        std::fflush(stdout);
    }

    SpeechAggregator(const SpeechAggregator&)            = delete;
    SpeechAggregator& operator=(const SpeechAggregator&) = delete;

    // ── Status pass-throughs (used by main() for fail-fast startup check) ───
    [[nodiscard]] bool        tts_ready() const noexcept { return tts_.ready(); }
    [[nodiscard]] std::string tts_error() const          { return tts_.error_message(); }

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

    [[nodiscard]] bool   enabled()       const noexcept { return enabled_; }
    [[nodiscard]] double gain()          const noexcept { return gain_; }
    [[nodiscard]] bool   already_spoken() const noexcept {
        return spoken_.load(std::memory_order_acquire);
    }

private:
    // Apply linear gain in-place to a raw int16_t PCM vector.
    // Returns true (always succeeds; clamp handles overflow).
    bool apply_gain(std::vector<int16_t>& pcm) {
        if (std::abs(gain_ - 1.0) < 0.001) return true;

        size_t clipped = 0;
        for (int16_t& s : pcm) {
            double y = static_cast<double>(s) * gain_;
            if (y >  32767.0) { y =  32767.0; ++clipped; }
            if (y < -32768.0) { y = -32768.0; ++clipped; }
            s = static_cast<int16_t>(std::lrint(y));
        }

        log("[speech] applied gain %.2fx to %zu samples (clipped=%zu)\n",
            gain_, pcm.size(), clipped);
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

        // synthesise() returns raw int16_t PCM — no temp file, no WAV header.
        // Model stays resident in the Piper subprocess; this call costs
        // inference time only (~100-300 ms on aarch64).
        auto pcm = tts_.synthesise(text);
        if (pcm.empty()) {
            std::fprintf(stderr,
                "[speech] synthesise failed: %s\n",
                tts_.error_message().c_str());
            return;
        }

        apply_gain(pcm);

        // play() feeds raw int16_t samples directly to libpulse,
        // bypassing the WAV parse step that play_wav() requires.
        if (!spk_.play(pcm)) {
            std::fprintf(stderr,
                "[speech] play failed: %s\n",
                spk_.last_error().c_str());
        }
    }

    TTSEngine         tts_;          // persistent Piper subprocess
    Speaker           spk_{};
    double            gain_       = 1.0;
    bool              verbose_    = true;
    std::string       piper_bin_;
    std::string       model_path_;
    std::atomic<bool> spoken_{false};
    std::atomic<bool> enabled_{false};
    std::atomic<uint64_t> play_generation_{0};
    std::mutex        speech_mutex_;
};
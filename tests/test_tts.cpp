// ─────────────────────────────────────────────────────────────────────────────
// test_tts.cpp  –  standalone TTS engine test (no speaker needed)
//
// Synthesises several sentences and reports WAV metadata.
// Optionally saves the WAV to a file so you can listen to it separately.
//
// Usage:
//   ./test_tts                          # synthesise built-in sentences
//   ./test_tts "Custom sentence here"   # synthesise one custom sentence
//   ./test_tts --save /tmp/out.wav      # also save the last WAV to disk
//
// Build target: test_tts
// ─────────────────────────────────────────────────────────────────────────────
#include "speaker.hpp"   // for parse_wav_header
#include "tts.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static void print_wav_info(const std::vector<uint8_t>& wav) {
    WavInfo info{};
    if (!Speaker::parse_wav_header(wav.data(), wav.size(), info)) {
        std::cout << "    [!] Could not parse WAV header\n";
        return;
    }
    const double dur =
        static_cast<double>(info.data_size) /
        (info.sample_rate * info.channels * (info.bits_per_sample / 8));

    std::cout << "    sample_rate  : " << info.sample_rate     << " Hz\n"
              << "    channels     : " << info.channels         << "\n"
              << "    bit depth    : " << info.bits_per_sample  << " bit\n"
              << "    pcm bytes    : " << info.data_size        << "\n"
              << "    duration     : " << std::fixed << std::setprecision(3)
                                       << dur                   << " s\n"
              << "    total size   : " << wav.size()            << " bytes\n";
}

static bool save_wav(const std::vector<uint8_t>& wav, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot write: " << path << "\n"; return false; }
    f.write(reinterpret_cast<const char*>(wav.data()),
            static_cast<std::streamsize>(wav.size()));
    return f.good();
}

int main(int argc, char* argv[]) {
    std::cout << "══════════════════════════════════════\n"
              << " TTS engine test (no speaker)\n"
              << "══════════════════════════════════════\n";

    // ── argument parsing ──────────────────────────────────────────────────────
    std::vector<std::string> sentences = {
        "Hello! The text-to-speech engine is working correctly.",
        "The quick brown fox jumps over the lazy dog.",
        "Raspberry Pi five, with dual MAX98357A amplifiers.",
    };
    std::string save_path;
    std::string custom_voice;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg == "--voice" && i + 1 < argc) {
            custom_voice = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            sentences = { arg };   // single custom sentence
        }
    }

    // ── engine setup ──────────────────────────────────────────────────────────
    TTSEngine::Config cfg;
    cfg.voice_path = custom_voice;   // empty → auto-detect

    TTSEngine tts(cfg);

    std::cout << "\nPiper binary : "
              << (tts.piper_path().empty() ? "[NOT FOUND]" : tts.piper_path()) << "\n"
              << "Voice        : "
              << (tts.voice_path().empty() ? "[NOT FOUND]" : tts.voice_path()) << "\n\n";

    if (!tts.ready()) {
        std::cerr << "ERROR: " << tts.last_error() << "\n"
                  << "Run setup_tts.sh to install Piper and a voice model.\n";
        return EXIT_FAILURE;
    }

    // ── synthesise each sentence ──────────────────────────────────────────────
    bool all_ok = true;
    std::vector<uint8_t> last_wav;

    for (size_t i = 0; i < sentences.size(); ++i) {
        const auto& text = sentences[i];
        std::cout << "[" << (i+1) << "/" << sentences.size() << "] \""
                  << text << "\"\n";

        const auto t0  = std::chrono::steady_clock::now();
        const auto wav = tts.synthesise_wav(text);
        const auto t1  = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (wav.empty()) {
            std::cout << "    FAILED: " << tts.last_error() << "\n";
            all_ok = false;
        } else {
            std::cout << "    synth time   : " << std::fixed << std::setprecision(1)
                      << ms << " ms\n";
            print_wav_info(wav);

            // Compute real-time factor (RTF < 1 means faster-than-real-time)
            WavInfo info{};
            Speaker::parse_wav_header(wav.data(), wav.size(), info);
            const double audio_dur =
                static_cast<double>(info.data_size) /
                (info.sample_rate * info.channels * 2.0);
            if (audio_dur > 0.0)
                std::cout << "    RTF          : "
                          << std::setprecision(3) << (ms / 1000.0) / audio_dur
                          << "  (< 1.0 = faster than real-time ✓)\n";

            last_wav = wav;
        }
        std::cout << "\n";
    }

    // ── optionally save the last WAV ──────────────────────────────────────────
    if (!save_path.empty() && !last_wav.empty()) {
        if (save_wav(last_wav, save_path))
            std::cout << "Saved WAV → " << save_path << "\n"
                      << "Play with: aplay " << save_path << "\n\n";
        else
            all_ok = false;
    }

    std::cout << "Overall result: " << (all_ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// ─────────────────────────────────────────────────────────────────────────────
// test_speaker.cpp  –  standalone speaker driver test
//
// Mode 1 (default):  synthesise a 440 Hz sine wave and play it for 1 second.
// Mode 2:            pass a WAV file path as argv[1] to play it.
//
// Build target: test_speaker
// ─────────────────────────────────────────────────────────────────────────────
#include "speaker.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <string>
#include <vector>

// ── Generate a sine-wave tone ─────────────────────────────────────────────────
// Returns num_samples 16-bit PCM samples at sample_rate Hz.
static std::vector<int16_t> make_sine(
    double  freq_hz,
    double  duration_s,
    double  amplitude,   // 0.0 – 1.0
    uint32_t sample_rate)
{
    const size_t n = static_cast<size_t>(duration_s * sample_rate);
    std::vector<int16_t> buf(n);
    const double k  = 2.0 * M_PI * freq_hz / sample_rate;
    const double amp = amplitude * 32'767.0;
    for (size_t i = 0; i < n; ++i)
        buf[i] = static_cast<int16_t>(amp * std::sin(k * static_cast<double>(i)));
    return buf;
}

// ── Apply a simple linear fade-in / fade-out ─────────────────────────────────
static void apply_fade(std::vector<int16_t>& buf, uint32_t sample_rate,
                       double fade_s = 0.02)
{
    const size_t fade = static_cast<size_t>(fade_s * sample_rate);
    const size_t n    = buf.size();
    for (size_t i = 0; i < fade && i < n; ++i) {
        buf[i]     = static_cast<int16_t>(buf[i]     * (static_cast<double>(i)   / fade));
        buf[n-1-i] = static_cast<int16_t>(buf[n-1-i] * (static_cast<double>(i)   / fade));
    }
}

// ── Sweep: play three chromatic test tones ────────────────────────────────────
static bool play_tone_sweep(Speaker& spk) {
    const struct { double freq; const char* note; } tones[] = {
        {261.63, "C4 (middle C)"},
        {440.00, "A4 (concert pitch)"},
        {523.25, "C5"},
    };

    for (const auto& t : tones) {
        std::cout << "  → " << t.note << " (" << t.freq << " Hz, 0.5 s)\n";
        auto buf = make_sine(t.freq, 0.5, 0.7, spk.sample_rate());
        apply_fade(buf, spk.sample_rate());
        if (!spk.play(buf)) {
            std::cerr << "  [ERROR] " << spk.last_error() << "\n";
            return false;
        }
    }
    return spk.drain();
}

int main(int argc, char* argv[]) {
    std::cout << "══════════════════════════════════════\n"
              << " Speaker driver test\n"
              << "══════════════════════════════════════\n";

    // ── Mode 2: play a WAV file ───────────────────────────────────────────────
    if (argc > 1) {
        const std::string path = argv[1];
        std::cout << "Mode: play WAV file → " << path << "\n";

        // Peek at the WAV header first
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "[ERROR] Cannot open file: " << path << "\n";
            return EXIT_FAILURE;
        }
        const std::vector<uint8_t> wav_data(
            (std::istreambuf_iterator<char>(f)),
            std::istreambuf_iterator<char>()
        );

        WavInfo info{};
        if (!Speaker::parse_wav_header(wav_data.data(), wav_data.size(), info)) {
            std::cerr << "[ERROR] Not a valid 16-bit PCM WAV file\n";
            return EXIT_FAILURE;
        }
        std::cout << "  sample_rate     : " << info.sample_rate     << " Hz\n"
                  << "  channels        : " << info.channels         << "\n"
                  << "  bits_per_sample : " << info.bits_per_sample  << "\n"
                  << "  data_size       : " << info.data_size / 1024 << " KiB\n"
                  << "  duration        : "
                  << static_cast<double>(info.data_size) /
                     (info.sample_rate * info.channels * info.bits_per_sample / 8)
                  << " s\n";

        Speaker spk(info.sample_rate,
                    static_cast<uint8_t>(info.channels));
        if (!spk.open()) {
            std::cerr << "[ERROR] " << spk.last_error() << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "Playing…\n";
        if (!spk.play_wav(wav_data)) {
            std::cerr << "[ERROR] " << spk.last_error() << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "Done.\n";
        return EXIT_SUCCESS;
    }

    // ── Mode 1: tone sweep ────────────────────────────────────────────────────
    std::cout << "Mode: tone sweep (440 Hz sine + chromatic neighbours)\n"
              << "Target: PipeWire default sink\n\n";

    constexpr uint32_t RATE = 44'100;   // use 44.1k for the tone test
    Speaker spk(RATE, 1);
    if (!spk.open()) {
        std::cerr << "[ERROR] Could not open speaker: " << spk.last_error() << "\n"
                  << "  Is pipewire-pulse running?  Try: systemctl --user status pipewire-pulse\n";
        return EXIT_FAILURE;
    }
    std::cout << "Stream opened at " << spk.sample_rate() << " Hz, "
              << static_cast<int>(spk.channels()) << " ch\n\n";

    // 200 ms silence before first tone
    {
        const auto silence = std::vector<int16_t>(RATE / 5, 0);
        spk.play(silence);
    }

    const bool ok = play_tone_sweep(spk);

    // 200 ms trailing silence
    {
        const auto silence = std::vector<int16_t>(RATE / 5, 0);
        spk.play(silence);
        spk.drain();
    }

    std::cout << "\nTest " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// speaker.hpp  –  MAX98357A I²S speaker driver via PipeWire/PulseAudio API
//
// Uses the PulseAudio simple-API (libpulse-simple) which PipeWire exposes
// transparently through its PA compatibility daemon (pipewire-pulse).
// The configured default sink (platform-soc_107c000000_sound) is used unless
// an explicit sink name is passed to the constructor.
//
// Thread safety: a Speaker instance must be used from a single thread.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>
#include <vector>

// Forward-declare the PA opaque type so callers need not include pulse headers.
struct pa_simple;

// ---------------------------------------------------------------------------
// WavInfo  –  metadata extracted from a parsed WAV header
// ---------------------------------------------------------------------------
struct WavInfo {
    uint32_t sample_rate     = 0;
    uint16_t channels        = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_offset     = 0;   // byte offset of PCM data in the buffer
    uint32_t data_size       = 0;   // byte length of PCM data
};

// ---------------------------------------------------------------------------
// Speaker
// ---------------------------------------------------------------------------
class Speaker {
public:
    // Default format matches Piper TTS output: 22 050 Hz, mono, 16-bit signed LE.
    static constexpr uint32_t DEFAULT_RATE     = 22'050;
    static constexpr uint8_t  DEFAULT_CHANNELS = 1;

    // sink_name: empty string → use the PipeWire default sink.
    explicit Speaker(
        uint32_t    sample_rate = DEFAULT_RATE,
        uint8_t     channels    = DEFAULT_CHANNELS,
        std::string sink_name   = ""
    );
    ~Speaker();

    Speaker(const Speaker&)            = delete;
    Speaker& operator=(const Speaker&) = delete;

    // ── lifecycle ────────────────────────────────────────────────────────────
    // open() creates the PA stream; must be called before any play*() method.
    bool open();
    void close();
    [[nodiscard]] bool is_open() const noexcept { return pa_ != nullptr; }

    // ── raw PCM playback ─────────────────────────────────────────────────────
    // Blocks until all samples have been consumed by the audio server.
    bool play(const int16_t* pcm, size_t num_samples);
    bool play(const std::vector<int16_t>& pcm);

    // ── WAV playback ─────────────────────────────────────────────────────────
    // Parses the WAV header, reopens the PA stream if sample rate / channels
    // differ from the current configuration, then plays the PCM data.
    bool play_wav(const std::vector<uint8_t>& wav_data);
    bool play_wav_file(const std::string& path);

    // Wait until the server has played everything that was written.
    bool drain();

    // ── diagnostics ──────────────────────────────────────────────────────────
    [[nodiscard]] const std::string& last_error() const noexcept { return err_; }
    [[nodiscard]] uint32_t           sample_rate() const noexcept { return rate_; }
    [[nodiscard]] uint8_t            channels()    const noexcept { return ch_; }

    // ── static helpers ───────────────────────────────────────────────────────
    // Parse a WAV header from an in-memory buffer.  Returns false if the buffer
    // is not a valid 16-bit PCM RIFF/WAVE file.
    static bool parse_wav_header(const uint8_t* data, size_t size, WavInfo& out);

private:
    bool reopen(uint32_t sample_rate, uint8_t channels);

    uint32_t    rate_;
    uint8_t     ch_;
    std::string sink_name_;
    pa_simple*  pa_  = nullptr;
    std::string err_;
};

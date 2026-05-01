// ─────────────────────────────────────────────────────────────────────────────
// speaker.cpp  –  MAX98357A I²S speaker driver (PipeWire / PulseAudio)
// ─────────────────────────────────────────────────────────────────────────────
#include "speaker.hpp"

#include <pulse/error.h>
#include <pulse/sample.h>
#include <pulse/simple.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>

// ── WAV chunk scanner ─────────────────────────────────────────────────────────
// Reads multi-byte little-endian integers without UB.
static uint16_t read_u16le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static uint32_t read_u32le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) <<  8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

bool Speaker::parse_wav_header(const uint8_t* data, size_t size, WavInfo& out) {
    // Minimum valid WAV: RIFF header (12) + fmt chunk (24) + data chunk (8)
    if (size < 44)                              return false;
    if (std::memcmp(data,     "RIFF", 4) != 0) return false;
    if (std::memcmp(data + 8, "WAVE", 4) != 0) return false;

    bool found_fmt  = false;
    bool found_data = false;

    size_t pos = 12;
    while (pos + 8 <= size) {
        const uint32_t chunk_size = read_u32le(data + pos + 4);

        if (std::memcmp(data + pos, "fmt ", 4) == 0) {
            if (chunk_size < 16) return false;           // malformed fmt chunk
            const uint16_t audio_fmt = read_u16le(data + pos + 8);
            if (audio_fmt != 1) return false;            // not PCM
            out.channels        = read_u16le(data + pos + 10);
            out.sample_rate     = read_u32le(data + pos + 12);
            out.bits_per_sample = read_u16le(data + pos + 22);
            found_fmt = true;

        } else if (std::memcmp(data + pos, "data", 4) == 0) {
            out.data_offset = static_cast<uint32_t>(pos + 8);
            out.data_size   = chunk_size;
            found_data = true;
            break;  // data chunk is always last; stop scanning
        }

        pos += 8 + chunk_size;
        if (chunk_size & 1u) ++pos;  // RIFF chunks are word-aligned
    }

    return found_fmt && found_data;
}

// ── Speaker ───────────────────────────────────────────────────────────────────

Speaker::Speaker(uint32_t sample_rate, uint8_t channels, std::string sink_name)
    : rate_(sample_rate), ch_(channels), sink_name_(std::move(sink_name)) {}

Speaker::~Speaker() { close(); }

bool Speaker::open() {
    close();

    pa_sample_spec spec{};
    spec.format   = PA_SAMPLE_S16LE;
    spec.rate     = rate_;
    spec.channels = ch_;

    // pa_channel_map_init_auto gives sensible defaults (MONO for ch_==1)
    pa_channel_map cmap;
    pa_channel_map_init_auto(&cmap, ch_, PA_CHANNEL_MAP_DEFAULT);

    int err = 0;
    pa_ = pa_simple_new(
        nullptr,                                         // server  (null → default)
        "tts-driver",                                    // app name
        PA_STREAM_PLAYBACK,
        sink_name_.empty() ? nullptr : sink_name_.c_str(), // sink (null → default)
        "speech",                                        // stream description
        &spec,
        &cmap,
        nullptr,                                         // buffer attrs (default)
        &err
    );

    if (!pa_) {
        err_ = std::string("pa_simple_new: ") + pa_strerror(err);
        return false;
    }
    return true;
}

void Speaker::close() {
    if (pa_) {
        pa_simple_free(pa_);
        pa_ = nullptr;
    }
}

bool Speaker::reopen(uint32_t sample_rate, uint8_t channels) {
    close();
    rate_ = sample_rate;
    ch_   = channels;
    return open();
}

bool Speaker::drain() {
    if (!pa_) return true;
    int err = 0;
    if (pa_simple_drain(pa_, &err) < 0) {
        err_ = std::string("pa_simple_drain: ") + pa_strerror(err);
        return false;
    }
    return true;
}

bool Speaker::play(const int16_t* pcm, size_t num_samples) {
    if (!pa_) { err_ = "Speaker not open"; return false; }
    if (num_samples == 0) return true;

    int err = 0;
    if (pa_simple_write(pa_, pcm, num_samples * sizeof(int16_t), &err) < 0) {
        err_ = std::string("pa_simple_write: ") + pa_strerror(err);
        return false;
    }
    return true;
}

bool Speaker::play(const std::vector<int16_t>& pcm) {
    return play(pcm.data(), pcm.size());
}

bool Speaker::play_wav(const std::vector<uint8_t>& wav_data) {
    WavInfo info{};
    if (!parse_wav_header(wav_data.data(), wav_data.size(), info)) {
        err_ = "Invalid WAV: bad or unsupported header";
        return false;
    }
    if (info.bits_per_sample != 16) {
        err_ = "Only 16-bit signed PCM WAV is supported";
        return false;
    }

    // Reopen the PA stream only when format parameters change.
    if (!pa_ || info.sample_rate != rate_ || info.channels != ch_) {
        if (!reopen(info.sample_rate, info.channels)) return false;
    }

    if (info.data_offset > wav_data.size()) {
        err_ = "WAV data offset beyond buffer end";
        return false;
    }
    const size_t avail  = wav_data.size() - info.data_offset;
    const size_t nbytes = std::min<size_t>(info.data_size, avail);

    int err = 0;
    if (pa_simple_write(pa_, wav_data.data() + info.data_offset, nbytes, &err) < 0) {
        err_ = std::string("pa_simple_write: ") + pa_strerror(err);
        return false;
    }
    return drain();
}

bool Speaker::play_wav_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err_ = "Cannot open file: " + path;
        return false;
    }
    const std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );
    return play_wav(data);
}
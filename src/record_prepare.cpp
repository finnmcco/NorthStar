// record_prepare.cpp  —  Stage 1: capture → extract → resample → WAV
//
// Reads stereo S32_LE @ 48kHz from ALSA, extracts the left channel,
// downsamples to 16kHz mono int16, and writes a standard WAV file
// that Stage 2 (infer) and any external tool (aplay, Audacity, soxi) can read.
//
// Build:
//   g++ -std=c++17 -O2 -o record_prepare record_prepare.cpp -lasound -lsamplerate
//
// Usage:
//   ./record_prepare                              # default device, output.wav
//   ./record_prepare hw:2,0 capture.wav           # explicit device and file
//   ./record_prepare hw:2,0 capture.wav 2.0       # optional gain multiplier

#include <alsa/asoundlib.h>
#include <samplerate.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr unsigned int      CAP_RATE      = 48000;
static constexpr unsigned int      OUT_RATE      = 16000;
static constexpr unsigned int      CHANNELS      = 2;       // ALSA capture channels
static constexpr snd_pcm_uframes_t PERIOD_FRAMES = 4800;    // 100ms at 48kHz
static constexpr snd_pcm_uframes_t BUFFER_FRAMES = PERIOD_FRAMES * 4;
static constexpr double            SRC_RATIO     = static_cast<double>(OUT_RATE) / CAP_RATE;

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// ---------------------------------------------------------------------------
// WAV writer  (PCM, mono, 16-bit)
// ---------------------------------------------------------------------------

// WAV header — all fields little-endian
#pragma pack(push, 1)
struct WavHeader {
    // RIFF chunk
    char     riff[4]       = {'R','I','F','F'};
    uint32_t chunk_size    = 0;           // filled in at close()
    char     wave[4]       = {'W','A','V','E'};
    // fmt sub-chunk
    char     fmt[4]        = {'f','m','t',' '};
    uint32_t fmt_size      = 16;
    uint16_t audio_fmt     = 1;           // PCM
    uint16_t num_channels  = 1;
    uint32_t sample_rate   = OUT_RATE;
    uint32_t byte_rate     = OUT_RATE * 1 * 2;   // rate * ch * bytes_per_sample
    uint16_t block_align   = 2;
    uint16_t bits_per_samp = 16;
    // data sub-chunk
    char     data[4]       = {'d','a','t','a'};
    uint32_t data_size     = 0;           // filled in at close()
};
#pragma pack(pop)

class WavWriter {
public:
    explicit WavWriter(const std::string& path) : path_(path) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_)
            throw std::runtime_error("Cannot open for writing: " + path);
        // Write placeholder header — we'll overwrite it at close()
        WavHeader hdr;
        std::fwrite(&hdr, sizeof(hdr), 1, f_);
    }

    ~WavWriter() { close(); }

    // Append mono int16 samples
    void write(const std::vector<int16_t>& samples) {
        std::fwrite(samples.data(), sizeof(int16_t), samples.size(), f_);
        data_bytes_ += samples.size() * sizeof(int16_t);
    }

    void close() {
        if (!f_) return;
        // Patch the header with final sizes
        WavHeader hdr;
        hdr.data_size  = static_cast<uint32_t>(data_bytes_);
        hdr.chunk_size = static_cast<uint32_t>(sizeof(WavHeader) - 8 + data_bytes_);
        std::rewind(f_);
        std::fwrite(&hdr, sizeof(hdr), 1, f_);
        std::fclose(f_);
        f_ = nullptr;
    }

    size_t bytes_written() const { return data_bytes_; }

private:
    std::string path_;
    std::FILE*  f_          = nullptr;
    size_t      data_bytes_ = 0;
};

// ---------------------------------------------------------------------------
// ALSA helpers
// ---------------------------------------------------------------------------

static snd_pcm_t* open_capture(const char* device) {
    snd_pcm_t* handle = nullptr;
    int err = snd_pcm_open(&handle, device, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0)
        throw std::runtime_error(std::string("snd_pcm_open: ") + snd_strerror(err));

    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(handle, hw);

    auto chk = [&](int e, const char* msg) {
        if (e < 0) throw std::runtime_error(std::string(msg) + ": " + snd_strerror(e));
    };

    unsigned int      rate   = CAP_RATE;
    snd_pcm_uframes_t period = PERIOD_FRAMES;
    snd_pcm_uframes_t buf    = BUFFER_FRAMES;

    chk(snd_pcm_hw_params_set_access(handle, hw, SND_PCM_ACCESS_RW_INTERLEAVED), "set_access");
    chk(snd_pcm_hw_params_set_format(handle, hw, SND_PCM_FORMAT_S32_LE),         "set_format");
    chk(snd_pcm_hw_params_set_rate_near(handle, hw, &rate, nullptr),              "set_rate");
    chk(snd_pcm_hw_params_set_channels(handle, hw, CHANNELS),                    "set_channels");
    chk(snd_pcm_hw_params_set_period_size_near(handle, hw, &period, nullptr),    "set_period");
    chk(snd_pcm_hw_params_set_buffer_size_near(handle, hw, &buf),                "set_buffer");
    chk(snd_pcm_hw_params(handle, hw),                                           "apply_params");
    chk(snd_pcm_prepare(handle),                                                 "prepare");
    return handle;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    const char* alsa_dev  = argc > 1 ? argv[1] : "hw:2,0";
    const char* out_path  = argc > 2 ? argv[2] : "output.wav";
    float       gain      = argc > 3 ? std::stof(argv[3]) : 1.0f;

    if (gain != 1.0f)
        std::printf("Gain: %.1fx\n", gain);

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Open ALSA
    snd_pcm_t* cap = nullptr;
    try {
        std::printf("Opening capture (%s)... ", alsa_dev);
        cap = open_capture(alsa_dev);
        std::puts("ok");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }

    // Open WAV output
    WavWriter wav(out_path);
    std::printf("Output: %s  (mono, %u Hz, 16-bit PCM)\n", out_path, OUT_RATE);

    // Set up libsamplerate  (stateful — persists across chunks)
    int src_err = 0;
    SRC_STATE* src = src_new(SRC_SINC_FASTEST, 1, &src_err);
    if (!src) {
        std::fprintf(stderr, "src_new: %s\n", src_strerror(src_err));
        snd_pcm_close(cap);
        return 1;
    }

    // Working buffers
    std::vector<int32_t> raw(PERIOD_FRAMES * CHANNELS);   // stereo S32_LE from ALSA
    std::vector<float>   in_f(PERIOD_FRAMES);              // left ch as float
    std::vector<float>   out_f(PERIOD_FRAMES);             // resampled float
    std::vector<int16_t> out_s(PERIOD_FRAMES);             // resampled int16

    std::puts("\nRecording — Ctrl+C to stop.\n");
    snd_pcm_start(cap);
    auto t0 = std::chrono::steady_clock::now();

    while (!g_stop) {
        // 1. Read one period of stereo S32_LE from ALSA
        snd_pcm_sframes_t n = snd_pcm_readi(cap, raw.data(), PERIOD_FRAMES);
        if (n == -EPIPE) {
            std::fputs("\r[capture] XRUN — recovering\n", stderr);
            snd_pcm_prepare(cap);
            continue;
        }
        if (n < 0) {
            std::fprintf(stderr, "\n[capture] fatal: %s\n", snd_strerror(n));
            break;
        }

        // 2. Extract left channel: S32_LE left-justified 24-in-32
        //    raw[i*2] = left frame i.  >> 16 → top 16 of 24 significant bits.
        //    Convert to float for libsamplerate (-1.0 … +1.0 range).
        for (snd_pcm_sframes_t i = 0; i < n; ++i)
            in_f[i] = static_cast<float>(raw[i * 2] >> 16) / 32768.0f * gain;

        // 3. Resample PERIOD_FRAMES @ 48kHz → ~1600 frames @ 16kHz
        const int out_capacity = static_cast<int>(n * SRC_RATIO) + 2;
        out_f.resize(out_capacity);
        out_s.resize(out_capacity);

        SRC_DATA d{};
        d.data_in       = in_f.data();
        d.data_out      = out_f.data();
        d.input_frames  = n;
        d.output_frames = out_capacity;
        d.src_ratio     = SRC_RATIO;
        d.end_of_input  = 0;

        if ((src_err = src_process(src, &d)) != 0) {
            std::fprintf(stderr, "src_process: %s\n", src_strerror(src_err));
            break;
        }

        // 4. Convert float → int16 and write to WAV
        src_float_to_short_array(out_f.data(), out_s.data(), d.output_frames_gen);
        out_s.resize(d.output_frames_gen);
        wav.write(out_s);

        // Progress
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("\r  %.1f s  |  %.1f KB written ",
                    ms / 1000.0,
                    wav.bytes_written() / 1024.0);
        std::fflush(stdout);
    }

    // Cleanup
    snd_pcm_drop(cap);
    snd_pcm_close(cap);
    src_delete(src);
    wav.close();   // patches WAV header with final sizes

    double secs = static_cast<double>(wav.bytes_written()) /
                  (OUT_RATE * sizeof(int16_t));
    std::printf("\n\nSaved %.2f s → %s\n", secs, out_path);
    std::printf("Verify: aplay -f S16_LE -r %u -c 1 %s\n", OUT_RATE, out_path);
    std::printf("Or:     aplay %s\n", out_path);  // aplay reads WAV headers natively
    return 0;
}
/*
    audio_record.cpp  --  record raw mic audio to AudioTests/ with no gain

    Captures from the ICS43432 MEMS microphone, extracts the left channel,
    applies gain=1.0 (no amplification), resamples 48kHz->16kHz, and saves
    a 16kHz mono 16-bit PCM WAV file into AudioTests/.

    The file is named by timestamp so repeated runs don't overwrite each other.

    Usage:
        ./audio_record              -- records until Ctrl-C
        ./audio_record <seconds>    -- records for a fixed duration

    Output:
        AudioTests/YYYYMMDD_HHMMSS.wav
*/

#include "audio_config.hpp"
#include "resampler.hpp"

#include <alsa/asoundlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

// -- WAV helpers --------------------------------------------------------------

static void write_u32_le(std::ostream& f, uint32_t v) {
    uint8_t b[4] = { uint8_t(v), uint8_t(v>>8), uint8_t(v>>16), uint8_t(v>>24) };
    f.write(reinterpret_cast<char*>(b), 4);
}
static void write_u16_le(std::ostream& f, uint16_t v) {
    uint8_t b[2] = { uint8_t(v), uint8_t(v>>8) };
    f.write(reinterpret_cast<char*>(b), 2);
}

static void write_wav_header(std::ostream& f, uint32_t num_samples,
                             uint32_t sample_rate = 16000)
{
    const uint32_t data_bytes  = num_samples * 2;   // 16-bit mono
    const uint32_t riff_size   = 36 + data_bytes;

    f.write("RIFF", 4);
    write_u32_le(f, riff_size);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    write_u32_le(f, 16);                    // chunk size
    write_u16_le(f, 1);                     // PCM
    write_u16_le(f, 1);                     // mono
    write_u32_le(f, sample_rate);
    write_u32_le(f, sample_rate * 2);       // byte rate
    write_u16_le(f, 2);                     // block align
    write_u16_le(f, 16);                    // bits per sample
    f.write("data", 4);
    write_u32_le(f, data_bytes);
}

// Rewrite the WAV header with final sample count (called after recording).
static void finalise_wav(const std::string& path, uint32_t num_samples)
{
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) { std::fprintf(stderr, "[wav] could not reopen %s\n", path.c_str()); return; }
    write_wav_header(f, num_samples);
}

// -- Output path --------------------------------------------------------------

static std::string make_output_path()
{
    std::filesystem::create_directories("AudioTests");
    const auto now = std::chrono::system_clock::to_time_t(
                         std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);
    std::ostringstream oss;
    oss << "AudioTests/" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".wav";
    return oss.str();
}

// -- ALSA capture (inline, gain = 1.0) ----------------------------------------

static snd_pcm_t* open_capture()
{
    snd_pcm_t* handle = nullptr;
    int err;
    if ((err = snd_pcm_open(&handle, Config::ALSA_DEVICE,
                            SND_PCM_STREAM_CAPTURE, 0)) < 0)
        throw std::runtime_error("snd_pcm_open: " + std::string(snd_strerror(err)));

    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(handle, hw);

    auto chk = [&](int e, const char* msg) {
        if (e < 0) throw std::runtime_error(std::string(msg) + ": " + snd_strerror(e));
    };

    unsigned int      rate   = Config::CAP_RATE;
    snd_pcm_uframes_t period = Config::PERIOD_FRAMES;
    snd_pcm_uframes_t buf    = Config::BUFFER_FRAMES;

    chk(snd_pcm_hw_params_set_access(handle, hw, SND_PCM_ACCESS_RW_INTERLEAVED), "set_access");
    chk(snd_pcm_hw_params_set_format(handle, hw, SND_PCM_FORMAT_S32_LE),         "set_format");
    chk(snd_pcm_hw_params_set_rate_near(handle, hw, &rate, nullptr),              "set_rate");
    chk(snd_pcm_hw_params_set_channels(handle, hw, Config::CAP_CHANNELS),        "set_channels");
    chk(snd_pcm_hw_params_set_period_size_near(handle, hw, &period, nullptr),    "set_period");
    chk(snd_pcm_hw_params_set_buffer_size_near(handle, hw, &buf),                "set_buffer");
    chk(snd_pcm_hw_params(handle, hw),                                           "apply_params");
    chk(snd_pcm_prepare(handle),                                                 "prepare");
    chk(snd_pcm_start(handle),                                                   "start");
    return handle;
}

// -- main ---------------------------------------------------------------------

int main(int argc, char* argv[])
{
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    int duration_s = -1;  // -1 = unlimited
    if (argc >= 2) {
        duration_s = std::atoi(argv[1]);
        if (duration_s <= 0) {
            std::fprintf(stderr, "Usage: %s [seconds]\n", argv[0]);
            return 1;
        }
    }

    const std::string output_path = make_output_path();
    std::printf("[record] output:    %s\n", output_path.c_str());
    std::printf("[record] gain:      1.0 (none)\n");
    std::printf("[record] device:    %s\n", Config::ALSA_DEVICE);
    std::printf("[record] format:    16kHz mono 16-bit PCM\n");
    if (duration_s > 0)
        std::printf("[record] duration:  %d s\n", duration_s);
    else
        std::printf("[record] duration:  until Ctrl-C\n");
    std::printf("[record] recording...\n\n");

    // Open WAV with placeholder header (will be finalised at end)
    std::ofstream wav(output_path, std::ios::binary);
    if (!wav) {
        std::fprintf(stderr, "[record] FATAL: cannot open %s\n", output_path.c_str());
        return 1;
    }
    write_wav_header(wav, 0);   // placeholder

    snd_pcm_t* handle = nullptr;
    try { handle = open_capture(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "[record] FATAL: %s\n", e.what());
        return 1;
    }

    Resampler resampler;
    std::vector<int32_t> raw_buf(Config::PERIOD_FRAMES * Config::CAP_CHANNELS);
    std::vector<int16_t> mono_48k(Config::PERIOD_FRAMES);
    std::vector<int16_t> mono_16k;

    uint32_t total_samples = 0;
    const auto start_time  = std::chrono::steady_clock::now();

    while (g_running) {
        // Check duration limit
        if (duration_s > 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= duration_s) break;
        }

        snd_pcm_sframes_t n = snd_pcm_readi(handle,
                                             raw_buf.data(),
                                             Config::PERIOD_FRAMES);
        if (n == -EPIPE) {
            std::fprintf(stderr, "[record] XRUN -- recovering\n");
            snd_pcm_prepare(handle);
            continue;
        }
        if (n < 0) {
            std::fprintf(stderr, "[record] read error: %s\n", snd_strerror(n));
            break;
        }

        // Extract left channel, >> 16, gain = 1.0 (no multiplication)
        for (snd_pcm_sframes_t i = 0; i < n; ++i) {
            const int32_t raw = raw_buf[i * Config::CAP_CHANNELS];   // left ch
            mono_48k[i] = static_cast<int16_t>(raw >> 16);
        }
        mono_48k.resize(n);

        // Resample 48k -> 16k
        resampler.process(mono_48k, mono_16k);

        // Write samples
        wav.write(reinterpret_cast<const char*>(mono_16k.data()),
                  mono_16k.size() * sizeof(int16_t));
        total_samples += static_cast<uint32_t>(mono_16k.size());

        const float secs = static_cast<float>(total_samples) / 16000.f;
        std::printf("\r[record] %.1f s  (%u samples)", secs, total_samples);
        std::fflush(stdout);
    }

    std::printf("\n[record] stopping...\n");
    snd_pcm_drop(handle);
    snd_pcm_close(handle);
    wav.close();

    // Rewrite WAV header with correct sample count
    finalise_wav(output_path, total_samples);

    const float secs = static_cast<float>(total_samples) / 16000.f;
    std::printf("[record] saved %.2f s -> %s\n", secs, output_path.c_str());
    return 0;
}

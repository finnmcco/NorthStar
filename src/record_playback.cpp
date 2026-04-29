// record_playback.cpp
// Opens capture (hw:2,0) and playback (plughw:3,0) simultaneously,
// records until Ctrl+C, then plays back.
//
// Build:
//   g++ -std=c++17 -O2 -o record_playback record_playback.cpp -lasound
//
// Usage:
//   ./record_playback                          # both channels
//   ./record_playback hw:2,0 plughw:3,0 L     # left channel only
//   ./record_playback hw:2,0 plughw:3,0 R     # right channel only
//   ./record_playback hw:2,0 plughw:3,0 both  # explicit both

#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr unsigned int      SAMPLE_RATE    = 48000;
static constexpr unsigned int      CHANNELS       = 2;
static constexpr snd_pcm_uframes_t PERIOD_FRAMES  = 4800;
static constexpr snd_pcm_uframes_t BUFFER_FRAMES  = PERIOD_FRAMES * 4;
static constexpr size_t            PERIOD_SAMPLES = PERIOD_FRAMES * CHANNELS;

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// ---------------------------------------------------------------------------
// ALSA helpers
// ---------------------------------------------------------------------------

static snd_pcm_t* open_pcm(const char* device, snd_pcm_stream_t stream) {
    snd_pcm_t* handle = nullptr;
    int err = snd_pcm_open(&handle, device, stream, 0);
    if (err < 0)
        throw std::runtime_error(std::string("snd_pcm_open (") + device +
                                 "): " + snd_strerror(err));
    return handle;
}

static void configure_pcm(snd_pcm_t* handle, const char* label) {
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(handle, hw);

    auto chk = [&](int e, const char* msg) {
        if (e < 0)
            throw std::runtime_error(std::string(label) + " — " +
                                     msg + ": " + snd_strerror(e));
    };

    unsigned int      rate   = SAMPLE_RATE;
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
}

// Zero out one channel in an interleaved stereo period before playback.
// mute_ch: 0 = zero left (play right only), 1 = zero right (play left only)
static void mute_channel(std::vector<int32_t>& period, int mute_ch) {
    for (size_t i = mute_ch; i < period.size(); i += CHANNELS)
        period[i] = 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    const char* cap_dev = argc > 1 ? argv[1] : "hw:2,0";
    const char* pb_dev  = argc > 2 ? argv[2] : "plughw:3,0";
    const char* ch_arg  = argc > 3 ? argv[3] : "both";

    // mute_ch: which interleaved index to zero out during playback.
    // Interleaved layout: index 0 = left, index 1 = right.
    //   "L" → keep left,  silence right → mute_ch = 1
    //   "R" → keep right, silence left  → mute_ch = 0
    //   "both" (default)                → mute_ch = -1 (no muting)
    int mute_ch = -1;
    if (ch_arg[0] == 'L' || ch_arg[0] == 'l') {
        mute_ch = 1;
        std::puts("Channel: LEFT only");
    } else if (ch_arg[0] == 'R' || ch_arg[0] == 'r') {
        mute_ch = 0;
        std::puts("Channel: RIGHT only");
    } else if (ch_arg[0] != 'b') {
        std::fprintf(stderr, "Unknown channel '%s' — using both.\n", ch_arg);
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    snd_pcm_t* cap_handle = nullptr;
    snd_pcm_t* pb_handle  = nullptr;

    // -----------------------------------------------------------------------
    // Step 1 — open and configure both handles before doing anything else
    // -----------------------------------------------------------------------
    try {
        std::printf("Opening capture  (%s)... ", cap_dev);
        cap_handle = open_pcm(cap_dev, SND_PCM_STREAM_CAPTURE);
        configure_pcm(cap_handle, "capture");
        std::puts("ok");

        std::printf("Opening playback (%s)... ", pb_dev);
        pb_handle = open_pcm(pb_dev, SND_PCM_STREAM_PLAYBACK);
        configure_pcm(pb_handle, "playback");
        std::puts("ok");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILED\n%s\n", e.what());
        if (cap_handle) snd_pcm_close(cap_handle);
        if (pb_handle)  snd_pcm_close(pb_handle);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Step 2 — record
    // -----------------------------------------------------------------------
    std::vector<std::vector<int32_t>> recording;
    std::vector<int32_t> period_buf(PERIOD_SAMPLES);

    std::puts("\nRecording — press Ctrl+C to stop and play back.");

    snd_pcm_start(cap_handle);
    auto t0 = std::chrono::steady_clock::now();

    while (!g_stop) {
        snd_pcm_sframes_t n = snd_pcm_readi(
            cap_handle, period_buf.data(), PERIOD_FRAMES);

        if (n == -EPIPE) {
            std::fputs("\r[capture] XRUN — recovering\n", stderr);
            snd_pcm_prepare(cap_handle);
            continue;
        }
        if (n < 0) {
            std::fprintf(stderr, "\n[capture] fatal: %s\n", snd_strerror(n));
            break;
        }

        recording.push_back(period_buf);

        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("\r  %.1f s recorded ", ms / 1000.0);
        std::fflush(stdout);
    }

    snd_pcm_drop(cap_handle);
    snd_pcm_close(cap_handle);
    cap_handle = nullptr;

    double duration = static_cast<double>(recording.size()) *
                      PERIOD_FRAMES / SAMPLE_RATE;
    std::printf("\n\nCaptured %.2f s (%zu periods).\n", duration, recording.size());

    if (recording.empty()) {
        std::puts("Nothing recorded.");
        snd_pcm_close(pb_handle);
        return 0;
    }

    // -----------------------------------------------------------------------
    // Step 3 — play back
    // -----------------------------------------------------------------------
    std::printf("Playing back on %s...\n", pb_dev);
    g_stop = false;   // allow Ctrl+C to abort playback too

    for (auto period : recording) {          // copy — mute_channel modifies in place
        if (g_stop) {
            std::puts("Playback interrupted.");
            break;
        }

        if (mute_ch >= 0)
            mute_channel(period, mute_ch);

        const int32_t*    ptr    = period.data();
        snd_pcm_uframes_t frames = PERIOD_FRAMES;

        while (frames > 0) {
            snd_pcm_sframes_t n = snd_pcm_writei(pb_handle, ptr, frames);
            if (n == -EPIPE) {
                std::fputs("[playback] underrun — recovering\n", stderr);
                snd_pcm_prepare(pb_handle);
                continue;
            }
            if (n < 0) {
                std::fprintf(stderr, "[playback] fatal: %s\n", snd_strerror(n));
                goto done;
            }
            ptr    += n * CHANNELS;
            frames -= n;
        }
    }

done:
    snd_pcm_drain(pb_handle);
    snd_pcm_close(pb_handle);
    std::puts("Done.");
    return 0;
}
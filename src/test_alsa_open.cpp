// test_alsa_open.cpp
// Verifies that the capture and playback ALSA handles can both be opened
// and configured simultaneously, before any audio is transferred.
//
// Build:
//   g++ -std=c++17 -O2 -o test_alsa_open test_alsa_open.cpp -lasound
//
// Usage:
//   ./test_alsa_open
//   ./test_alsa_open hw:2,0 plughw:3,0

#include <alsa/asoundlib.h>
#include <cstdio>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// Constants — must match the rest of the driver
// ---------------------------------------------------------------------------

static constexpr unsigned int SAMPLE_RATE   = 48000;
static constexpr unsigned int CHANNELS      = 2;
static constexpr snd_pcm_uframes_t PERIOD   = 4800;
static constexpr snd_pcm_uframes_t BUF      = PERIOD * 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_sep()  { std::puts("─────────────────────────────────────"); }
static void print_ok()   { std::puts("  [OK]"); }
static void print_fail() { std::puts("  [FAIL]"); }

// Configure a PCM handle to our standard parameters.
// Returns true on success, prints a reason on failure.
static bool configure_pcm(snd_pcm_t* handle, const char* label) {
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(handle, hw);

    struct Step { int rc; const char* name; };
    unsigned int rate = SAMPLE_RATE;
    snd_pcm_uframes_t period = PERIOD, buf = BUF;

    Step steps[] = {
        { snd_pcm_hw_params_set_access(handle, hw, SND_PCM_ACCESS_RW_INTERLEAVED),
          "access=RW_INTERLEAVED" },
        { snd_pcm_hw_params_set_format(handle, hw, SND_PCM_FORMAT_S32_LE),
          "format=S32_LE" },
        { snd_pcm_hw_params_set_rate_near(handle, hw, &rate, nullptr),
          "rate=48000" },
        { snd_pcm_hw_params_set_channels(handle, hw, CHANNELS),
          "channels=2" },
        { snd_pcm_hw_params_set_period_size_near(handle, hw, &period, nullptr),
          "period_size=4800" },
        { snd_pcm_hw_params_set_buffer_size_near(handle, hw, &buf),
          "buffer_size=19200" },
        { snd_pcm_hw_params(handle, hw),
          "apply hw_params" },
        { snd_pcm_prepare(handle),
          "prepare" },
    };

    bool ok = true;
    for (auto& s : steps) {
        std::printf("    %-30s", s.name);
        if (s.rc < 0) {
            std::printf("FAIL  (%s)\n", snd_strerror(s.rc));
            ok = false;
            break;   // later steps depend on earlier ones
        }
        std::puts("ok");
    }

    // Report what the driver actually negotiated (may differ from what we asked)
    if (ok) {
        snd_pcm_hw_params_t* hw2;
        snd_pcm_hw_params_alloca(&hw2);
        snd_pcm_hw_params_current(handle, hw2);

        unsigned int actual_rate; int dir;
        snd_pcm_uframes_t actual_period, actual_buf;
        snd_pcm_format_t  actual_fmt;
        unsigned int      actual_ch;

        snd_pcm_hw_params_get_rate(hw2, &actual_rate, &dir);
        snd_pcm_hw_params_get_period_size(hw2, &actual_period, &dir);
        snd_pcm_hw_params_get_buffer_size(hw2, &actual_buf);
        snd_pcm_hw_params_get_format(hw2, &actual_fmt);
        snd_pcm_hw_params_get_channels(hw2, &actual_ch);

        std::printf("    ── negotiated: %s, %u Hz, %u ch, "
                    "period=%lu frames, buf=%lu frames\n",
                    snd_pcm_format_name(actual_fmt),
                    actual_rate, actual_ch,
                    actual_period, actual_buf);

        // Warn if the driver silently changed anything critical
        if (actual_rate != SAMPLE_RATE)
            std::printf("    !! WARNING: rate mismatch — got %u, wanted %u\n",
                        actual_rate, SAMPLE_RATE);
        if (actual_ch != CHANNELS)
            std::printf("    !! WARNING: channel mismatch — got %u, wanted %u\n",
                        actual_ch, CHANNELS);
        if (actual_fmt != SND_PCM_FORMAT_S32_LE)
            std::printf("    !! WARNING: format mismatch — got %s, wanted S32_LE\n",
                        snd_pcm_format_name(actual_fmt));
    }

    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    const char* cap_dev = argc > 1 ? argv[1] : "hw:2,0";
    const char* pb_dev  = argc > 2 ? argv[2] : "plughw:3,0";

    std::printf("\nALSA dual-handle open test\n");
    std::printf("  capture  → %s\n", cap_dev);
    std::printf("  playback → %s\n\n", pb_dev);

    snd_pcm_t* cap_handle = nullptr;
    snd_pcm_t* pb_handle  = nullptr;
    bool cap_ok = false, pb_ok = false;

    // -----------------------------------------------------------------------
    // Step 1a — open capture
    // -----------------------------------------------------------------------
    print_sep();
    std::printf("Step 1a: open capture (%s)\n", cap_dev);
    {
        int err = snd_pcm_open(&cap_handle, cap_dev,
                               SND_PCM_STREAM_CAPTURE, 0);
        if (err < 0) {
            std::printf("  snd_pcm_open FAILED: %s\n", snd_strerror(err));
            print_fail();
        } else {
            print_ok();
            std::puts("  configure:");
            cap_ok = configure_pcm(cap_handle, "capture");
            cap_ok ? print_ok() : print_fail();
        }
    }

    // -----------------------------------------------------------------------
    // Step 1b — open playback (while capture handle is still open)
    // -----------------------------------------------------------------------
    print_sep();
    std::printf("Step 1b: open playback (%s) — capture still open\n", pb_dev);
    {
        int err = snd_pcm_open(&pb_handle, pb_dev,
                               SND_PCM_STREAM_PLAYBACK, 0);
        if (err < 0) {
            std::printf("  snd_pcm_open FAILED: %s\n", snd_strerror(err));
            print_fail();
        } else {
            print_ok();
            std::puts("  configure:");
            pb_ok = configure_pcm(pb_handle, "playback");
            pb_ok ? print_ok() : print_fail();
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    print_sep();
    std::puts("Summary:");
    std::printf("  capture  (%s):  %s\n", cap_dev,  cap_ok ? "PASS" : "FAIL");
    std::printf("  playback (%s): %s\n",  pb_dev,   pb_ok  ? "PASS" : "FAIL");

    if (cap_ok && pb_ok)
        std::puts("\n  Both handles open simultaneously — step 1 is good.\n");
    else
        std::puts("\n  One or more handles failed — fix before proceeding.\n");

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    if (cap_handle) snd_pcm_close(cap_handle);
    if (pb_handle)  snd_pcm_close(pb_handle);

    return (cap_ok && pb_ok) ? 0 : 1;
}
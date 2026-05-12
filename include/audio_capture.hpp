#pragma once
#include "audio_config.hpp"

#include <alsa/asoundlib.h>
#include <cstdint>
#include <string>
#include <vector>

// Opens an ALSA capture device and reads one period at a time.
//
// Each read_period() call:
//   1. Blocks until Config::PERIOD_FRAMES stereo S32_LE frames are ready
//   2. Extracts the left channel  (right is noise — R/L pin tied to GND)
//   3. Shifts right by 16 to get the top 16 of the 24 significant bits
//   4. Applies Config::MIC_GAIN in float domain before quantising to int16
//   5. Returns Config::PERIOD_FRAMES samples @ Config::CAP_RATE (48kHz)
//
// Output is fed to Resampler in the capture thread.

class AudioCapture {
public:
    explicit AudioCapture(const std::string& device = Config::ALSA_DEVICE);
    ~AudioCapture();

    AudioCapture(const AudioCapture&)            = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    void start();
    void stop();

    // Returns false on XRUN — caller should skip the chunk and continue.
    bool read_period(std::vector<int16_t>& out);

private:
    snd_pcm_t*           handle_  = nullptr;
    std::vector<int32_t> raw_buf_; // length = PERIOD_FRAMES * CAP_CHANNELS

    void configure();
};

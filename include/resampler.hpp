#pragma once
#include "config.hpp"

#include <samplerate.h>
#include <cstdint>
#include <vector>

// Downsamples mono int16 audio from Config::CAP_RATE (48kHz) to
// Config::VOSK_RATE (16kHz) using SRC_SINC_FASTEST.
//
// SRC_STATE is preserved across calls — filter history carries over
// between chunks, avoiding discontinuities at chunk boundaries.
//
// Gain is applied upstream in AudioCapture so the float conversion
// happens only once across the full gain + resample chain.

class Resampler {
public:
    Resampler();
    ~Resampler();

    Resampler(const Resampler&)            = delete;
    Resampler& operator=(const Resampler&) = delete;

    // in:  int16 @ Config::CAP_RATE   (Config::PERIOD_FRAMES samples)
    // out: int16 @ Config::VOSK_RATE  (~PERIOD_FRAMES / 3 samples)
    // Returns the number of output samples produced.
    int process(const std::vector<int16_t>& in, std::vector<int16_t>& out);

private:
    SRC_STATE*         state_ = nullptr;
    std::vector<float> in_f_, out_f_;
};

#include "audio_capture.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

AudioCapture::AudioCapture(const std::string& device) {
    int err;
    if ((err = snd_pcm_open(&handle_, device.c_str(),
                            SND_PCM_STREAM_CAPTURE, 0)) < 0)
        throw std::runtime_error("snd_pcm_open: " + std::string(snd_strerror(err)));

    configure();
    raw_buf_.resize(Config::PERIOD_FRAMES * Config::CAP_CHANNELS);
}

void AudioCapture::configure() {
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(handle_, hw);

    auto chk = [&](int e, const char* msg) {
        if (e < 0)
            throw std::runtime_error(std::string(msg) + ": " + snd_strerror(e));
    };

    unsigned int      rate   = Config::CAP_RATE;
    snd_pcm_uframes_t period = Config::PERIOD_FRAMES;
    snd_pcm_uframes_t buf    = Config::BUFFER_FRAMES;

    chk(snd_pcm_hw_params_set_access(handle_, hw, SND_PCM_ACCESS_RW_INTERLEAVED), "set_access");
    chk(snd_pcm_hw_params_set_format(handle_, hw, SND_PCM_FORMAT_S32_LE),         "set_format");
    chk(snd_pcm_hw_params_set_rate_near(handle_, hw, &rate, nullptr),              "set_rate");
    chk(snd_pcm_hw_params_set_channels(handle_, hw, Config::CAP_CHANNELS),        "set_channels");
    chk(snd_pcm_hw_params_set_period_size_near(handle_, hw, &period, nullptr),    "set_period");
    chk(snd_pcm_hw_params_set_buffer_size_near(handle_, hw, &buf),                "set_buffer");
    chk(snd_pcm_hw_params(handle_, hw),                                           "apply_params");
    chk(snd_pcm_prepare(handle_),                                                 "prepare");
}

AudioCapture::~AudioCapture() {
    if (handle_) {
        snd_pcm_drop(handle_);
        snd_pcm_close(handle_);
    }
}

void AudioCapture::start() {
    int err = snd_pcm_start(handle_);
    if (err < 0)
        throw std::runtime_error("snd_pcm_start: " + std::string(snd_strerror(err)));
}

void AudioCapture::stop() {
    snd_pcm_drop(handle_);
}

bool AudioCapture::read_period(std::vector<int16_t>& out) {
    out.resize(Config::PERIOD_FRAMES);

    snd_pcm_sframes_t n = snd_pcm_readi(handle_, raw_buf_.data(), Config::PERIOD_FRAMES);

    if (n == -EPIPE) {
        std::fprintf(stderr, "[audio] XRUN — recovering\n");
        snd_pcm_prepare(handle_);
        return false;
    }
    if (n < 0)
        throw std::runtime_error("snd_pcm_readi: " + std::string(snd_strerror(n)));

    // Left channel extraction + gain in float domain.
    //
    // raw_buf_[i * 2] = left frame i (interleaved, so every other int32).
    // The ICS43432 is left-justified in the 32-bit word — bits [31:8] hold
    // the 24-bit sample, bits [7:0] are zero padding.
    // >> 16 takes the top 16 of those 24 bits → standard int16 range.
    //
    // Gain is applied in float to avoid integer overflow, then clamped
    // before converting back to int16 for the Resampler.
    for (snd_pcm_sframes_t i = 0; i < n; ++i) {
        float f = static_cast<float>(raw_buf_[i * 2] >> 16) * Config::MIC_GAIN;
        f = std::clamp(f, -32768.0f, 32767.0f);
        out[i] = static_cast<int16_t>(f);
    }

    return true;
}

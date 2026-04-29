#include "resampler.hpp"

#include <stdexcept>

Resampler::Resampler() {
    int err;
    // SRC_SINC_FASTEST: adequate quality for speech, low CPU on Pi 5.
    // The integer ratio 48000/16000 = 3 is the best case for a sinc filter.
    state_ = src_new(SRC_SINC_FASTEST, 1, &err);
    if (!state_)
        throw std::runtime_error("src_new: " + std::string(src_strerror(err)));
}

Resampler::~Resampler() {
    if (state_) src_delete(state_);
}

int Resampler::process(const std::vector<int16_t>& in, std::vector<int16_t>& out) {
    const int in_n  = static_cast<int>(in.size());
    const int out_n = static_cast<int>(in_n * Config::RESAMPLE_RATIO) + 8;

    in_f_.resize(in_n);
    out_f_.resize(out_n);

    src_short_to_float_array(in.data(), in_f_.data(), in_n);

    SRC_DATA d{};
    d.data_in       = in_f_.data();
    d.data_out      = out_f_.data();
    d.input_frames  = in_n;
    d.output_frames = out_n;
    d.src_ratio     = Config::RESAMPLE_RATIO;
    d.end_of_input  = 0;

    int err = src_process(state_, &d);
    if (err)
        throw std::runtime_error("src_process: " + std::string(src_strerror(err)));

    const int produced = static_cast<int>(d.output_frames_gen);
    out.resize(produced);
    src_float_to_short_array(out_f_.data(), out.data(), produced);
    return produced;
}

#include "inference/hailo8_inference.hpp"
#include <hailo/hailort.h>
#include <iostream>

using namespace hailort;

Hailo8Inference::Hailo8Inference(const std::string& hef_path)
    : hef_path_(hef_path)
{
}

Hailo8Inference::~Hailo8Inference()
{
    stop();
}

bool Hailo8Inference::initialize()
{
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice\n";
        return false;
    }
    vdevice_ = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path_);
    if (!hef_exp) {
        std::cerr << "Failed to load HEF\n";
        return false;
    }
    auto hef = std::move(hef_exp.value());

    auto network_groups_exp = vdevice_->configure(hef);
    if (!network_groups_exp || network_groups_exp->empty()) {
        std::cerr << "Failed to configure network group\n";
        return false;
    }

    network_group_ = network_groups_exp.value().at(0);

    const uint32_t timeout_ms = HAILO_DEFAULT_VSTREAM_TIMEOUT_MS;

    // Pipeline depth: how many frames can be in-flight inside Hailo at once.
    // 2 means we can be writing frame N+1 while reading frame N.
    const uint32_t queue_size = 1;

    auto input_params_exp =
        network_group_->make_input_vstream_params(
            false,
            static_cast<hailo_format_type_t>(HAILO_FORMAT_TYPE_AUTO),
            timeout_ms,
            queue_size,
            "");

    auto output_params_exp =
        network_group_->make_output_vstream_params(
            false,
            static_cast<hailo_format_type_t>(HAILO_FORMAT_TYPE_AUTO),
            timeout_ms,
            queue_size,
            "");

    if (!input_params_exp || !output_params_exp) {
        std::cerr << "Failed to create vstream params\n";
        return false;
    }

    auto input_vstreams_exp =
        VStreamsBuilder::create_input_vstreams(
            *network_group_,
            input_params_exp.value());

    auto output_vstreams_exp =
        VStreamsBuilder::create_output_vstreams(
            *network_group_,
            output_params_exp.value());

    if (!input_vstreams_exp || !output_vstreams_exp) {
        std::cerr << "Failed to create vstreams\n";
        return false;
    }

    input_streams_  = std::move(input_vstreams_exp.value());
    output_streams_ = std::move(output_vstreams_exp.value());

    
    input_frame_size_  = input_streams_[0].get_frame_size();
    output_frame_size_ = output_streams_[0].get_frame_size();

    std::cout << "Hailo initialized\n";
    std::cout << "Input frame size:  " << input_frame_size_  << "\n";
    std::cout << "Output frame size: " << output_frame_size_ << "\n";

    std::cout << "Input format type: " << input_streams_[0].get_info().format.type << "\n";
    std::cout << "Input format order: " << input_streams_[0].get_info().format.order << "\n";

    // ── Start async read thread ──────────────────────────────────────────
    // This thread continuously reads completed results from Hailo and
    // places them into result_queue_ so run() can pick them up.


    return true;
}

void Hailo8Inference::stop()
{
    if (!input_streams_.empty())  input_streams_[0].abort();
    if (!output_streams_.empty()) output_streams_[0].abort();
}
bool Hailo8Inference::run(uint8_t* input_buffer)
{
    if (!input_buffer) return false;

    auto status = input_streams_[0].write(
        MemoryView(input_buffer, input_frame_size_));
    if (status != HAILO_SUCCESS) return false;

    output_buffer_.resize(output_frame_size_);
    status = output_streams_[0].read(
        MemoryView(output_buffer_.data(), output_frame_size_));
    if (status != HAILO_SUCCESS) return false;

    return true;
}

const std::vector<uint8_t>& Hailo8Inference::get_output() const
{
    return output_buffer_;
}
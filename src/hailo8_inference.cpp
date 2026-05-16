#include "hailo8_inference.hpp"
#include <iostream>

using namespace hailort;

Hailo8Inference::Hailo8Inference(const std::string& hef_path)
    : hef_path_(hef_path)
{}

Hailo8Inference::~Hailo8Inference()
{
    stop();
}

void Hailo8Inference::register_callback(ResultCallback cb)
{
    callback_ = std::move(cb);
}

bool Hailo8Inference::initialize()
{
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) { std::cerr << "[Hailo] Failed to create VDevice\n"; return false; }
    vdevice_ = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path_);
    if (!hef_exp) { std::cerr << "[Hailo] Failed to load HEF: " << hef_path_ << "\n"; return false; }
    auto hef = std::move(hef_exp.value());

    auto network_groups_exp = vdevice_->configure(hef);
    if (!network_groups_exp || network_groups_exp->empty()) { std::cerr << "[Hailo] Failed to configure\n"; return false; }
    network_group_ = network_groups_exp.value().at(0);

    const uint32_t timeout_ms = HAILO_DEFAULT_VSTREAM_TIMEOUT_MS;
    const uint32_t queue_size = 2;

    auto in_params_exp  = network_group_->make_input_vstream_params(false, static_cast<hailo_format_type_t>(HAILO_FORMAT_TYPE_AUTO), timeout_ms, queue_size, "");
    auto out_params_exp = network_group_->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_INFINITE, queue_size, "");

    if (!in_params_exp || !out_params_exp) { std::cerr << "[Hailo] Failed to create vstream params\n"; return false; }

    auto in_exp  = VStreamsBuilder::create_input_vstreams (*network_group_, in_params_exp.value());
    auto out_exp = VStreamsBuilder::create_output_vstreams(*network_group_, out_params_exp.value());

    if (!in_exp || !out_exp) { std::cerr << "[Hailo] Failed to create vstreams\n"; return false; }

    input_streams_  = std::move(in_exp.value());
    output_streams_ = std::move(out_exp.value());

    input_frame_size_ = input_streams_[0].get_frame_size();

    output_frame_sizes_.clear();
    for (const auto& s : output_streams_)
        output_frame_sizes_.push_back(s.get_frame_size());

    std::cout << "[Hailo] Initialized — " << output_streams_.size() << " output stream(s)\n";
    std::cout << "        input frame size : " << input_frame_size_ << " bytes\n";
    for (std::size_t i = 0; i < output_streams_.size(); ++i)
        std::cout << "        output[" << i << "] " << output_streams_[i].get_info().name << " : " << output_frame_sizes_[i] << " bytes\n";

    running_ = true;
    read_thread_ = std::thread([this]()
    {
        const std::size_t n = output_streams_.size();
        std::vector<std::vector<uint8_t>> bufs(n);
        for (std::size_t i = 0; i < n; ++i)
            bufs[i].resize(output_frame_sizes_[i]);

        while (running_)
        {
            bool ok = true;
            for (std::size_t i = 0; i < n && ok; ++i) {
                auto status = output_streams_[i].read(MemoryView(bufs[i].data(), bufs[i].size()));
                if (status != HAILO_SUCCESS) ok = false;
            }
            if (!ok) break;

            FrameContext ctx{};
            {
                std::lock_guard<std::mutex> lk(ctx_mutex_);
                if (!ctx_queue_.empty()) { ctx = ctx_queue_.front(); ctx_queue_.pop(); }
            }

            if (callback_)
                callback_(ctx.camera_id, ctx.timestamp_ns, bufs);
        }
    });

    return true;
}

bool Hailo8Inference::write_frame(const uint8_t* buffer, uint8_t camera_id, uint64_t timestamp_ns)
{
    if (!buffer || !running_) return false;

    { std::lock_guard<std::mutex> lk(ctx_mutex_); ctx_queue_.push({camera_id, timestamp_ns}); }

    auto status = input_streams_[0].write(MemoryView(const_cast<uint8_t*>(buffer), input_frame_size_));

    if (status != HAILO_SUCCESS) {
        std::lock_guard<std::mutex> lk(ctx_mutex_);
        if (!ctx_queue_.empty()) ctx_queue_.pop();
        return false;
    }
    return true;
}

void Hailo8Inference::stop()
{
    if (!running_.exchange(false)) return;
    if (!input_streams_.empty()) input_streams_[0].abort();
    for (auto& s : output_streams_) s.abort();
    if (read_thread_.joinable()) read_thread_.join();
}

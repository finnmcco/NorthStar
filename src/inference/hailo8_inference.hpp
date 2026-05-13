#pragma once

#include <hailo/hailort.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class Hailo8Inference
{
public:
    using ResultCallback = std::function<void(
        uint8_t                              camera_id,
        uint64_t                             timestamp_ns,
        std::vector<std::vector<uint8_t>>    outputs)>;

    explicit Hailo8Inference(const std::string& hef_path);
    ~Hailo8Inference();

    Hailo8Inference(const Hailo8Inference&)            = delete;
    Hailo8Inference& operator=(const Hailo8Inference&) = delete;

    void register_callback(ResultCallback cb);
    bool initialize();

    bool write_frame(const uint8_t* buffer,
                     uint8_t        camera_id,
                     uint64_t       timestamp_ns);

    void stop();

    std::size_t input_frame_size()   const { return input_frame_size_; }
    std::size_t num_output_streams() const { return output_frame_sizes_.size(); }

    std::size_t output_frame_size(std::size_t idx = 0) const
    {
        return idx < output_frame_sizes_.size() ? output_frame_sizes_[idx] : 0;
    }

private:
    struct FrameContext {
        uint8_t  camera_id;
        uint64_t timestamp_ns;
    };

    std::string hef_path_;

    std::unique_ptr<hailort::VDevice>                vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::vector<hailort::InputVStream>               input_streams_;
    std::vector<hailort::OutputVStream>              output_streams_;

    std::size_t              input_frame_size_ = 0;
    std::vector<std::size_t> output_frame_sizes_;

    ResultCallback callback_;

    std::mutex               ctx_mutex_;
    std::queue<FrameContext> ctx_queue_;

    std::thread       read_thread_;
    std::atomic<bool> running_{false};
};

#pragma once

#include <hailo/hailort.hpp>

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>

#include "util/queue.hpp"

class Hailo8Inference
{
public:
    explicit Hailo8Inference(const std::string& hef_path);
    ~Hailo8Inference();

    Hailo8Inference(const Hailo8Inference&)            = delete;
    Hailo8Inference& operator=(const Hailo8Inference&) = delete;

    bool initialize();
    void stop();
    bool run(uint8_t* input_buffer);

    const std::vector<uint8_t>& get_output() const;

    std::size_t input_frame_size()  const { return input_frame_size_;  }
    std::size_t output_frame_size() const { return output_frame_size_; }

private:
    std::string hef_path_;

    std::unique_ptr<hailort::VDevice>                vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;

    std::vector<hailort::InputVStream>  input_streams_;
    std::vector<hailort::OutputVStream> output_streams_;

    std::size_t input_frame_size_  = 0;
    std::size_t output_frame_size_ = 0;

    std::vector<uint8_t>          output_buffer_;

    std::thread                   read_thread_;
    std::atomic<bool>             running_{false};
    queue<std::vector<uint8_t>>   result_queue_{4};
};
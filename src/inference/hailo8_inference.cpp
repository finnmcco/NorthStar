#include "inference/hailo8_inference.hpp"

#include <iostream>

using namespace hailort;

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

Hailo8Inference::Hailo8Inference(const std::string& hef_path)
    : hef_path_(hef_path)
{
}

Hailo8Inference::~Hailo8Inference()
{
    stop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

void Hailo8Inference::register_callback(ResultCallback cb)
{
    callback_ = std::move(cb);
}

bool Hailo8Inference::initialize()
{
    // ── Create VDevice ───────────────────────────────────────────────────────
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "[Hailo] Failed to create VDevice\n";
        return false;
    }
    vdevice_ = std::move(vdevice_exp.value());

    // ── Load HEF ─────────────────────────────────────────────────────────────
    auto hef_exp = Hef::create(hef_path_);
    if (!hef_exp) {
        std::cerr << "[Hailo] Failed to load HEF: " << hef_path_ << "\n";
        return false;
    }
    auto hef = std::move(hef_exp.value());

    // ── Configure network ────────────────────────────────────────────────────
    auto network_groups_exp = vdevice_->configure(hef);
    if (!network_groups_exp || network_groups_exp->empty()) {
        std::cerr << "[Hailo] Failed to configure network group\n";
        return false;
    }
    network_group_ = network_groups_exp.value().at(0);

    // ── Build VStream params ─────────────────────────────────────────────────
    // queue_size=2 allows one frame to be in-flight inside Hailo while the
    // next write is being accepted — enabling double-buffered pipelining.
    const uint32_t timeout_ms  = HAILO_DEFAULT_VSTREAM_TIMEOUT_MS;
    const uint32_t queue_size  = 2;

    auto in_params_exp =
        network_group_->make_input_vstream_params(
            false,
            static_cast<hailo_format_type_t>(HAILO_FORMAT_TYPE_AUTO),
            timeout_ms, queue_size, "");

    auto out_params_exp =
        network_group_->make_output_vstream_params(
            false,
            static_cast<hailo_format_type_t>(HAILO_FORMAT_TYPE_AUTO),
            timeout_ms, queue_size, "");

    if (!in_params_exp || !out_params_exp) {
        std::cerr << "[Hailo] Failed to create vstream params\n";
        return false;
    }

    // ── Create VStreams ───────────────────────────────────────────────────────
    auto in_exp  = VStreamsBuilder::create_input_vstreams (*network_group_, in_params_exp.value());
    auto out_exp = VStreamsBuilder::create_output_vstreams(*network_group_, out_params_exp.value());

    if (!in_exp || !out_exp) {
        std::cerr << "[Hailo] Failed to create vstreams\n";
        return false;
    }

    input_streams_  = std::move(in_exp.value());
    output_streams_ = std::move(out_exp.value());

    input_frame_size_  = input_streams_[0].get_frame_size();
    output_frame_size_ = output_streams_[0].get_frame_size();

    // ── Activate network group ────────────────────────────────────────────────
    // The ActivatedNetworkGroup keeps the hardware pipeline live.  It must stay
    // in scope for the entire inference session — stored as a member for that
    // reason.  Without activation, writes are accepted but inference never runs
    // and both streams time out.
    auto activated_exp = network_group_->activate();
    if (!activated_exp) {
        std::cerr << "[Hailo] Failed to activate network group\n";
        return false;
    }
    auto activated_network_group = std::move(activated_exp.value());

    std::cout << "[Hailo] Initialized\n"
              << "        input  frame size : " << input_frame_size_  << " bytes\n"
              << "        output frame size : " << output_frame_size_ << " bytes\n"
              << "        input  format     : "
                  << input_streams_[0].get_info().format.type  << " / "
                  << input_streams_[0].get_info().format.order << "\n";

    // ── Start async read thread ───────────────────────────────────────────────
    // This thread blocks on output_streams_[0].read() — the same blocking-I/O
    // pattern described in the course notes §3.3.3.  It wakes up each time
    // Hailo completes one inference, fires the registered callback with the raw
    // NMS buffer and the context from ctx_queue_, then loops back to wait.
    running_ = true;
    read_thread_ = std::thread([this]()
    {
        std::vector<uint8_t> buf(output_frame_size_);

        while (running_)
        {
            // Blocking read — thread sleeps here until Hailo produces a result.
            auto status = output_streams_[0].read(
                MemoryView(buf.data(), buf.size()));

            if (status != HAILO_SUCCESS)
                break; // stream aborted (stop() called) — exit cleanly

            // Retrieve the context for this result.
            FrameContext ctx{};
            {
                std::lock_guard<std::mutex> lk(ctx_mutex_);
                if (!ctx_queue_.empty()) {
                    ctx = ctx_queue_.front();
                    ctx_queue_.pop();
                }
            }

            if (callback_)
                callback_(ctx.camera_id, ctx.timestamp_ns, buf);
        }
    });

    return true;
}

bool Hailo8Inference::write_frame(const uint8_t* buffer,
                                   uint8_t        camera_id,
                                   uint64_t       timestamp_ns)
{
    if (!buffer || !running_)
        return false;

    // Push context before the write so it is always present when the read
    // thread wakes up for this frame.  Only one thread calls write_frame()
    // so there is no concurrent push to worry about.
    {
        std::lock_guard<std::mutex> lk(ctx_mutex_);
        ctx_queue_.push({camera_id, timestamp_ns});
    }

    auto status = input_streams_[0].write(
        MemoryView(const_cast<uint8_t*>(buffer), input_frame_size_));

    if (status != HAILO_SUCCESS) {
        // Write failed; remove the context we just pushed so the queue
        // doesn't get out of sync with the Hailo pipeline.
        std::lock_guard<std::mutex> lk(ctx_mutex_);
        if (!ctx_queue_.empty())
            ctx_queue_.pop();
        return false;
    }

    return true;
}

void Hailo8Inference::stop()
{
    if (!running_.exchange(false))
        return; // already stopped

    // Abort both streams so the blocking read() in the read thread unblocks
    // and the blocking write() in write_frame() (if mid-call) returns an error.
    if (!input_streams_.empty())  input_streams_[0].abort();
    if (!output_streams_.empty()) output_streams_[0].abort();

    // Per course notes §3.3.2: always join before the thread goes out of scope.
    if (read_thread_.joinable())
        read_thread_.join();
}
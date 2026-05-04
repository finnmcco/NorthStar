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

/*
    Hailo8Inference
    ═══════════════
    Async wrapper around one Hailo VStream pipeline.

    Thread model
    ─────────────
    write_frame()  — called from the consumer thread.  Pushes the frame into
                     the Hailo input stream (blocks only until Hailo accepts
                     the data, typically < 1 ms) then returns immediately.

    read_thread_   — internal thread that blocks on output_streams_[0].read().
                     When Hailo completes an inference the read unblocks, the
                     registered callback is fired with the raw NMS buffer, then
                     the thread loops back to wait for the next result.

    Because write and read are independent Hailo can double-buffer: writing
    frame N+1 while frame N is still being processed.

    Context tracking
    ─────────────────
    Hailo processes frames in FIFO order so the Nth read result is always the
    Nth written frame.  write_frame() pushes {camera_id, timestamp} onto a
    ctx_queue_ before each write; the read thread pops the matching context to
    attach it to the callback invocation.

    Usage
    ──────
        Hailo8Inference hailo(DEFAULT_HEF_PATH);
        hailo.register_callback([](uint8_t cam, uint64_t ts, std::vector<uint8_t> out) {
            auto detections = parse_nms_output(out);
            // dispatch downstream ...
        });
        hailo.initialize();

        // consumer loop:
        while (queue.pop(pkt))
            hailo.write_frame(pkt.data, pkt.camera_id, pkt.timestamp);

        hailo.stop();
*/
class Hailo8Inference
{
public:
    /*
        ResultCallback

        Fired on the internal read thread each time one inference completes.

            camera_id    — forwarded from the matching write_frame() call
            timestamp_ns — forwarded from the matching write_frame() call
            output       — raw NMS output buffer; pass to parse_nms_output()
    */
    using ResultCallback = std::function<void(uint8_t              camera_id,
                                               uint64_t             timestamp_ns,
                                               std::vector<uint8_t> output)>;

    explicit Hailo8Inference(const std::string& hef_path);
    ~Hailo8Inference();

    Hailo8Inference(const Hailo8Inference&)            = delete;
    Hailo8Inference& operator=(const Hailo8Inference&) = delete;

    // Register the result callback.  Call before initialize().
    void register_callback(ResultCallback cb);

    // Create VDevice, configure network, start the read thread.
    // Returns false on any Hailo error.
    bool initialize();

    // Push one 640×640 RGB frame into Hailo.
    //   buffer       — kWidth*kHeight*kChannels bytes, RGB interleaved
    //   camera_id    — forwarded verbatim to the callback
    //   timestamp_ns — forwarded verbatim to the callback
    // Blocks only until Hailo accepts the data (typically < 1 ms).
    // Returns false if the stream has been aborted (stop() was called).
    bool write_frame(const uint8_t* buffer,
                     uint8_t        camera_id,
                     uint64_t       timestamp_ns);

    // Abort streams and join the read thread.  Idempotent.
    void stop();

    std::size_t input_frame_size()  const { return input_frame_size_;  }
    std::size_t output_frame_size() const { return output_frame_size_; }

private:
    // Metadata carried alongside each in-flight frame.
    struct FrameContext {
        uint8_t  camera_id;
        uint64_t timestamp_ns;
    };

    std::string hef_path_;

    std::unique_ptr<hailort::VDevice>                vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::vector<hailort::InputVStream>               input_streams_;
    std::vector<hailort::OutputVStream>              output_streams_;

    std::size_t input_frame_size_  = 0;
    std::size_t output_frame_size_ = 0;

    ResultCallback callback_;

    // FIFO context queue: write_frame() pushes, read_thread_ pops.
    // Protected by ctx_mutex_.
    std::mutex               ctx_mutex_;
    std::queue<FrameContext> ctx_queue_;

    std::thread       read_thread_;
    std::atomic<bool> running_{false};
};
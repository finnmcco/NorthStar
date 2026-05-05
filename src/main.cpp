/*
    main.cpp  —  NorthStar perception pipeline
    ════════════════════════════════════════════

    Full pipeline (see system context):
      1. Both cameras push 640×640 RGB frames into a shared frameQueue.
      2. The consumer thread pops frames and writes them to Hailo for inference.
      3. Hailo's internal read thread fires on_inference_result() with detections
         and the originating camera_id / timestamp.
      4. StereoMatcher pairs cam0 and cam1 results that share a close timestamp.
      5. on_stereo_pair() receives the matched pair for depth estimation,
         thermal lookup, and TTS (stubs — to be filled in as those subsystems
         are implemented).

    Thread layout
    ─────────────
        libcamera (cam0 & cam1)  ──┐
                                   ├─→  frameQueue  →  consumer thread  →  Hailo write
        libcamera (cam1)         ──┘                                            │
                                                              Hailo read thread─┘
                                                                        │
                                                              on_inference_result()
                                                                        │
                                                              StereoMatcher
                                                                        │
                                                              on_stereo_pair()
*/

#include "config.hpp"
#include <atomic>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <optional>

#include "capture_controller.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"
#include "camera_queue.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Graceful shutdown
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// ─────────────────────────────────────────────────────────────────────────────
//  StereoMatcher
//
//  Pairs inference results from cam0 and cam1 that originated from frames
//  captured within MATCH_TOLERANCE_US microseconds of each other.
//
//  Called exclusively from the Hailo read thread (single-threaded) so it
//  needs no internal locking.
// ─────────────────────────────────────────────────────────────────────────────
class StereoMatcher
{
public:
    struct Result {
        uint8_t              camera_id;
        uint64_t             timestamp_ns;
        std::vector<Detection> detections;
    };

    struct StereoPair {
        Result cam0;
        Result cam1;
    };

    using PairCallback = std::function<void(StereoPair)>;

    explicit StereoMatcher(PairCallback cb)
        : cb_(std::move(cb))
    {}

    // Feed one inference result.  If it completes a pair the PairCallback fires.
    void on_result(uint8_t camera_id,
                   uint64_t timestamp_ns,
                   std::vector<Detection> detections)
    {
        Result r{ camera_id, timestamp_ns, std::move(detections) };

        if (camera_id == 0) {
            if (pending_cam1_ && timestamps_match(timestamp_ns, pending_cam1_->timestamp_ns)) {
                cb_(StereoPair{ std::move(r), std::move(*pending_cam1_) });
                pending_cam1_.reset();
            } else {
                pending_cam0_ = std::move(r);
            }
        } else {
            if (pending_cam0_ && timestamps_match(timestamp_ns, pending_cam0_->timestamp_ns)) {
                cb_(StereoPair{ std::move(*pending_cam0_), std::move(r) });
                pending_cam0_.reset();
            } else {
                pending_cam1_ = std::move(r);
            }
        }
    }

private:
    // Two frames are a stereo pair if they are within half a frame period at
    // 30 fps (~16 ms).  Timestamps are nanoseconds.
    static constexpr uint64_t kMatchToleranceNs = 16'000'000ULL; // 16 ms

    static bool timestamps_match(uint64_t a, uint64_t b)
    {
        return (a > b ? a - b : b - a) < kMatchToleranceNs;
    }

    PairCallback         cb_;
    std::optional<Result> pending_cam0_;
    std::optional<Result> pending_cam1_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Downstream stubs
//  Replace these as the depth / thermal / TTS subsystems are implemented.
// ─────────────────────────────────────────────────────────────────────────────
static void on_stereo_pair(StereoMatcher::StereoPair pair)
{
    // TODO: stereo depth estimation using fixed baseline + bounding box centroids
    // TODO: thermal camera lookup (MLX90640) for object region
    // TODO: TTS output via MAX98357A

    // For now: log the paired detections.
    std::cout << "[stereo pair] cam0 ts=" << pair.cam0.timestamp_ns
              << "  cam1 ts=" << pair.cam1.timestamp_ns << "\n";

    for (const auto& d : pair.cam0.detections)
        std::cout << "  [cam0] " << COCO_CLASSES[d.class_id]
                  << " conf=" << d.score << "\n";

    for (const auto& d : pair.cam1.detections)
        std::cout << "  [cam1] " << COCO_CLASSES[d.class_id]
                  << " conf=" << d.score << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    // ── Hailo ────────────────────────────────────────────────────────────────
    const std::string hef_path = DEFAULT_HEF_PATH;
    Hailo8Inference hailo(hef_path);

    StereoMatcher matcher(on_stereo_pair);

    hailo.register_callback(
        [&matcher](uint8_t camera_id,
                   uint64_t timestamp_ns,
                   std::vector<std::vector<uint8_t>> raw_output)
        {
            // Parse NMS on the Hailo read thread, then hand off to matcher.
            // Matcher is only ever called from this thread so no locking needed.
            auto detections = parse_detections(raw_output);
            matcher.on_result(camera_id, timestamp_ns, std::move(detections));
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    // ── Camera pipeline ───────────────────────────────────────────────────────
    // Both cameras push into the same queue.  CameraCapture sets camera_id on
    // each FramePacket so downstream code can tell them apart.

    queue<FramePacket> frameQueue(8);

    // TODO: replace with a two-camera CaptureController once libcamera dual-
    //       stream configuration is validated.  For now a single CameraCapture
    //       exercises the full downstream pipeline.
    CameraCapture capture(frameQueue, /*fps=*/30);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "NorthStar pipeline running — press Ctrl-C to quit\n";

    // ── Consumer loop ─────────────────────────────────────────────────────────
    // Single thread: pops FramePacket → write_frame() → release pool buffer.
    // write_frame() returns as soon as Hailo accepts the data (< 1 ms).
    // Results arrive asynchronously via the registered callback.
    FramePacket pkt{};

    while (g_running && frameQueue.pop(pkt))
    {
        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);

        // Pool buffer can be released immediately; Hailo has its own copy.
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    capture.stop();
    frameQueue.stop();
    hailo.stop(); // aborts streams and joins read thread

    std::cout << "Pipeline stopped.\n";
    return 0;
}
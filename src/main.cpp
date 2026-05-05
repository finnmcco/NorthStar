/*
    main.cpp  —  NorthStar perception pipeline
    ════════════════════════════════════════════

    Full pipeline:
      1. Camera pushes 640×640 BGR frames into a shared frameQueue.
      2. Consumer thread pops frames and writes them to Hailo for inference.
      3. Hailo read thread fires callback with detections + camera_id + timestamp.
      4. StereoMatcher pairs cam0 and cam1 results by timestamp.
      5. on_stereo_pair() receives the matched pair for depth / thermal / TTS.

    Thread layout
    ─────────────
        libcamera (cam0 & cam1)  ──┐
                                   ├─→  frameQueue  →  consumer  →  Hailo write
        libcamera (cam1)         ──┘                                      │
                                                        Hailo read thread─┘
                                                                  │
                                                        StereoMatcher → on_stereo_pair()
*/

#include "config.hpp"
#include "camera_config.hpp"
#include <atomic>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <optional>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "inference/hailo8_inference.hpp"
#include "detection_utils.hpp"

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// ─────────────────────────────────────────────────────────────────────────────
//  StereoMatcher
//  Called exclusively from the Hailo read thread — no locking needed.
// ─────────────────────────────────────────────────────────────────────────────
class StereoMatcher
{
public:
    struct Result {
        uint8_t                camera_id;
        uint64_t               timestamp_ns;
        std::vector<Detection> detections;
    };

    struct StereoPair {
        Result cam0;
        Result cam1;
    };

    using PairCallback = std::function<void(StereoPair)>;

    explicit StereoMatcher(PairCallback cb) : cb_(std::move(cb)) {}

    void on_result(uint8_t camera_id, uint64_t timestamp_ns,
                   std::vector<Detection> detections)
    {
        Result r{ camera_id, timestamp_ns, std::move(detections) };

        if (camera_id == 0) {
            if (pending_cam1_ && match(timestamp_ns, pending_cam1_->timestamp_ns)) {
                cb_(StereoPair{ std::move(r), std::move(*pending_cam1_) });
                pending_cam1_.reset();
            } else {
                pending_cam0_ = std::move(r);
            }
        } else {
            if (pending_cam0_ && match(timestamp_ns, pending_cam0_->timestamp_ns)) {
                cb_(StereoPair{ std::move(*pending_cam0_), std::move(r) });
                pending_cam0_.reset();
            } else {
                pending_cam1_ = std::move(r);
            }
        }
    }

private:
    static bool match(uint64_t a, uint64_t b)
    {
        return (a > b ? a - b : b - a) < STEREO_MATCH_TOLERANCE_NS;
    }

    PairCallback          cb_;
    std::optional<Result> pending_cam0_;
    std::optional<Result> pending_cam1_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Downstream stubs — replace as subsystems are implemented
// ─────────────────────────────────────────────────────────────────────────────
static void on_stereo_pair(StereoMatcher::StereoPair pair)
{
    // TODO: stereo depth estimation using fixed baseline + bounding box centroids
    // TODO: thermal camera lookup (MLX90640) for object region
    // TODO: TTS output via MAX98357A

    std::cout << "[stereo pair] cam0 ts=" << pair.cam0.timestamp_ns
              << "  cam1 ts=" << pair.cam1.timestamp_ns << "\n";

    for (const auto& d : pair.cam0.detections)
        std::cout << "  [cam0] " << COCO_CLASSES[d.object_id]
                  << " conf=" << d.confidence << "\n";

    for (const auto& d : pair.cam1.detections)
        std::cout << "  [cam1] " << COCO_CLASSES[d.object_id]
                  << " conf=" << d.confidence << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    Hailo8Inference hailo(DEFAULT_HEF_PATH);
    StereoMatcher   matcher(on_stereo_pair);

    hailo.register_callback(
        [&matcher](uint8_t camera_id,
                   uint64_t timestamp_ns,
                   std::vector<std::vector<uint8_t>> raw_output)
        {
            auto detections = parse_detections(raw_output);
            matcher.on_result(camera_id, timestamp_ns, std::move(detections));
        });

    if (!hailo.initialize()) {
        std::cerr << "Failed to initialise Hailo\n";
        return 1;
    }

    queue<FramePacket> frameQueue(FRAME_QUEUE_DEPTH);

    // TODO: replace with two-camera CaptureController once dual-stream is validated.
    CameraCapture capture(frameQueue);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "NorthStar pipeline running — press Ctrl-C to quit\n";

    FramePacket pkt{};
    while (g_running && frameQueue.pop(pkt))
        hailo.write_frame(pkt.data.data(), pkt.camera_id, pkt.timestamp);

    capture.stop();
    frameQueue.stop();
    hailo.stop();

    std::cout << "Pipeline stopped.\n";
    return 0;
}
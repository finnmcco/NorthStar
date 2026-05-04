#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "capture/frame_packet.hpp"
#include "util/buffer_pool.hpp"
#include "util/queue.hpp"

/*
    ImageCapture
    ════════════
    Simulates a stereo camera pair by reading pre-captured image files from
    disk.  Drop-in replacement for CameraCapture in test and benchmark builds.

    Folder layout
    ─────────────
        root/
            cam0/   ← images for camera 0, sorted alphanumerically
            cam1/   ← images for camera 1, sorted alphanumerically

    If cam1/ does not exist or is empty, all frames are assigned camera_id = 0.

    Image requirements
    ──────────────────
    Each image must be 640 × 640 RGB (PNG or JPEG).  Images that fail to
    decode or have the wrong dimensions are skipped with a warning.

    Interleaving
    ─────────────
    The worker emits cam0[0], cam1[0], cam0[1], cam1[1], ...
    When loop = true  it wraps around indefinitely (good for sustained benchmarks).
    When loop = false it stops after one pass through the shorter list.

    Timing
    ──────
    Frames are pushed at the configured fps using sleep_for.  Timestamps
    are monotonic nanoseconds from CLOCK_MONOTONIC_RAW, just like the real
    camera — so end-to-end latency measurements remain meaningful.

    Memory
    ──────
    Each frame is copied into a BufferPool buffer.  The consumer MUST call
    pool.release(pkt.data) when it has finished with the frame, exactly as
    it would with frames from CameraCapture.
*/
class ImageCapture
{
public:
    static constexpr int kWidth    = 640;
    static constexpr int kHeight   = 640;
    static constexpr int kChannels = 3;

    /*
        Parameters
        ──────────
        root_dir  — directory containing cam0/ and (optionally) cam1/ subfolders.
        outQueue  — queue to push FramePacket objects into.
        pool      — BufferPool supplying kWidth*kHeight*kChannels byte buffers.
        fps       — push rate (best-effort, default 30).
        loop      — true: cycle through images indefinitely.
                    false: stop after one pass through the shorter list.
    */
    ImageCapture(const std::string&  root_dir,
                 queue<FramePacket>& outQueue,
                 BufferPool&         pool,
                 int                 fps  = 30,
                 bool                loop = false);

    ImageCapture(const ImageCapture&)            = delete;
    ImageCapture& operator=(const ImageCapture&) = delete;

    // Scan root_dir for images and start the worker thread.
    // Returns false if no images were found or the thread fails to start.
    bool start();

    // Signal the worker to stop and join it.
    void stop();

    std::size_t count_cam0() const { return paths_cam0_.size(); }
    std::size_t count_cam1() const { return paths_cam1_.size(); }

private:
    void run();

    static std::uint64_t now_ns();

    // Return sorted image paths (png/jpg/jpeg) from a directory.
    // Returns an empty vector if the directory does not exist.
    static std::vector<std::filesystem::path>
    load_image_paths(const std::filesystem::path& dir);

    // Decode one image file into a pre-allocated 640×640×3 RGB buffer.
    // Returns false and logs a warning if the image cannot be loaded or has
    // the wrong dimensions.
    static bool decode_into(const std::filesystem::path& path, uint8_t* dst);

private:
    std::string         root_dir_;
    queue<FramePacket>& outQueue_;
    BufferPool&         pool_;
    int                 fps_;
    bool                loop_;

    std::vector<std::filesystem::path> paths_cam0_;
    std::vector<std::filesystem::path> paths_cam1_;

    std::atomic<bool> running_{false};
    std::thread       worker_;
};

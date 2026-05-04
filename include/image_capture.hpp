#pragma once
#include <atomic>
#include <string>
#include <thread>

#include "frame_packet.hpp"
#include "camera_queue.hpp"

/*
    ImageCapture
    ════════════
    Feeds pre-captured PNG/JPEG images into the pipeline queue instead of a
    live camera.  Designed as a drop-in replacement for CameraCapture when
    libcamera is unavailable (e.g. on a development machine) or when you want
    a reproducible dataset for benchmarking.

    Directory layout
    ─────────────────
        image_root/
            cam0/   ← 640×640 images for camera id 0
            cam1/   ← 640×640 images for camera id 1  (optional)

    Images are sorted alphabetically within each subfolder so capture order is
    deterministic.  Non-image files are silently skipped.

    Frame interleaving
    ───────────────────
    If cam1/ exists, frames are interleaved: cam0[0], cam1[0], cam0[1], cam1[1],
    … with matching timestamps so StereoMatcher can pair them.  If cam1/ is
    absent, only cam0 frames are pushed.

    Parameters
    ──────────
    fps   — simulated capture rate; controls the sleep between frames.
    loop  — if true, cycle through images forever; if false, stop after one pass.

    Images that are not already 640×640 are resized automatically via OpenCV.
    Images are converted to RGB (matching the live camera pipeline output).
*/
class ImageCapture
{
public:
    ImageCapture(const std::string&  image_root,
                 queue<FramePacket>& q,
                 BufferPool&         pool,
                 int                 fps  = 30,
                 bool                loop = false);

    ~ImageCapture();

    ImageCapture(const ImageCapture&)            = delete;
    ImageCapture& operator=(const ImageCapture&) = delete;

    // Starts the worker thread.  Returns false if no images are found.
    bool start();

    // Signals the worker to stop and joins the thread.
    void stop();

private:
    std::string         image_root_;
    queue<FramePacket>& queue_;
    BufferPool&         pool_;
    int                 fps_;
    bool                loop_;

    std::thread         worker_;
    std::atomic<bool>   running_{false};

    void run();
};
#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

// Your project headers (adjust paths if needed)
#include "capture/frame_packet.hpp"
#include "util/queue.hpp"
#include "util/buffer_pool.hpp"

/*
    FakeCapture

    Purpose:
    --------
    A stand-in for the real libcamera-based capture thread.

    It simulates a camera by:
      - acquiring an output buffer from BufferPool
      - filling it with a simple RGB test pattern
      - timestamping it (monotonic clock)
      - pushing a FramePacket into a BoundedQueue
      - repeating at a fixed frame rate (e.g., 30 FPS)

    Why it’s useful:
    ----------------
    - Validates your threading and queue behavior without camera complexity.
    - Validates BufferPool acquire/release lifetimes.
    - Gives you a stable base before integrating libcamera.

    Threading:
    ----------
    - start() launches an internal worker thread.
    - stop() requests shutdown and joins the worker thread.
*/

class FakeCapture
{
public:
    // Output format: 640x640 RGB
    static constexpr int kWidth    = 640;
    static constexpr int kHeight   = 640;
    static constexpr int kChannels = 3;

    /*
        Constructor

        Parameters:
          - outQueue: queue that the fake capture thread will push FramePackets into
          - pool: BufferPool that provides the output buffers (owned elsewhere)
          - fps: how fast to "produce" frames (default 30)
    */
    FakeCapture(queue<FramePacket>& outQueue,
                BufferPool& pool,
                int fps = 30);

    // Disable copying (owns a thread, and references shared objects)
    FakeCapture(const FakeCapture&) = delete;
    FakeCapture& operator=(const FakeCapture&) = delete;

    // Start the worker thread. Returns false if already running or fps invalid.
    bool start();

    // Request shutdown and join the worker thread (safe to call multiple times).
    void stop();

private:
    // Worker loop run on the background thread
    void run();

    // Helper: monotonic timestamp in nanoseconds
    static std::uint64_t now_ns();

    // Helper: fill the buffer with a simple RGB pattern
    static void fill_test_pattern_rgb(std::uint8_t* dst,
                                      int width,
                                      int height,
                                      int strideBytes,
                                      std::uint32_t frameIndex);

private:
    queue<FramePacket>& outQueue_;
    BufferPool& pool_;
    int fps_ = 30;

    std::atomic<bool> running_{false};
    std::thread worker_;

    std::uint32_t frameCounter_ = 0;
};
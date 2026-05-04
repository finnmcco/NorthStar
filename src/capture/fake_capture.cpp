#include "capture/fake_capture.hpp"

#include <chrono>
#include <ctime>    // clock_gettime
#include <cstring>  // memset (optional)
#include <thread>

FakeCapture::FakeCapture(queue<FramePacket>& outQueue,
                         BufferPool& pool,
                         int fps)
    : outQueue_(outQueue), pool_(pool), fps_(fps)
{
}

bool FakeCapture::start()
{
    // Prevent starting twice
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false; // already running
    }

    if (fps_ <= 0) {
        running_ = false;
        return false;
    }

    // Launch the worker thread
    worker_ = std::thread(&FakeCapture::run, this);
    return true;
}

void FakeCapture::stop()
{
    // If already stopped, do nothing
    bool wasRunning = running_.exchange(false);

    // Join if the thread exists
    if (worker_.joinable()) {
        worker_.join();
    }

    (void)wasRunning;
}

void FakeCapture::run()
{
    // Target period per frame
    const auto framePeriod = std::chrono::microseconds(1'000'000 / fps_);

    // Our output is tightly packed RGB by default:
    // strideBytes = width * 3
    const int outStrideBytes = kWidth * kChannels;

    while (running_.load())
    {
        const auto tStart = std::chrono::steady_clock::now();

        // 1) Acquire an output buffer (non-blocking)
        std::uint8_t* out = pool_.try_acquire();
        if (!out) {
            // No free buffer available:
            // - This means downstream is not releasing fast enough.
            // - Policy: drop this frame and continue (don’t stall capture).
            std::this_thread::sleep_for(framePeriod);
            continue;
        }

        // 2) Fill it with a simple test pattern (so you can see it changes)
        fill_test_pattern_rgb(out, kWidth, kHeight, outStrideBytes, frameCounter_);

        // 3) Timestamp it (monotonic, ns)
        const std::uint64_t ts = now_ns();

        // 4) Push packet into the queue
        FramePacket pkt;
        pkt.timestamp = ts;
        pkt.data = out;

        // If queue is stopped, push() returns false.
        // In that case, release buffer back to pool and exit.
        if (!outQueue_.push(pkt)) {
            pool_.release(out);
            break;
        }

        frameCounter_++;

        // 5) Sleep to maintain FPS (best-effort)
        const auto tEnd = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);

        if (elapsed < framePeriod) {
            std::this_thread::sleep_for(framePeriod - elapsed);
        }
    }
}

std::uint64_t FakeCapture::now_ns()
{
    // CLOCK_MONOTONIC_RAW is good for consistent timestamps (not affected by NTP)
    // If your system doesn't support it, CLOCK_MONOTONIC is also fine.
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    const std::uint64_t sec  = static_cast<std::uint64_t>(ts.tv_sec);
    const std::uint64_t nsec = static_cast<std::uint64_t>(ts.tv_nsec);
    return sec * 1'000'000'000ULL + nsec;
}

void FakeCapture::fill_test_pattern_rgb(std::uint8_t* dst,
                                        int width,
                                        int height,
                                        int strideBytes,
                                        std::uint32_t frameIndex)
{
    // A simple pattern:
    // - R varies with x
    // - G varies with y
    // - B varies with frame index (so it changes over time)
    //
    // This makes it easy to verify frames are updating.

    const std::uint8_t blue = static_cast<std::uint8_t>(frameIndex & 0xFF);

    for (int y = 0; y < height; ++y)
    {
        std::uint8_t* row = dst + y * strideBytes;

        const std::uint8_t green = static_cast<std::uint8_t>((y * 255) / (height - 1));

        for (int x = 0; x < width; ++x)
        {
            const std::uint8_t red = static_cast<std::uint8_t>((x * 255) / (width - 1));

            const int idx = x * 3;
            row[idx + 0] = red;    // R
            row[idx + 1] = green;  // G
            row[idx + 2] = blue;   // B
        }
    }
}
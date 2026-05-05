/*
    camera_throughput.cpp  —  Raw camera frame rate test (no Hailo).
    Ctrl-C to stop.
*/

#include "camera_config.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <numeric>
#include <vector>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main()
{
    std::signal(SIGINT, on_sigint);

    queue<FramePacket> frameQueue(FRAME_QUEUE_DEPTH);
    CameraCapture capture(frameQueue);

    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "Camera throughput test — Ctrl-C to stop\n\n";

    FramePacket pkt{};
    int frame_count = 0;
    std::vector<double> inter_ms;
    inter_ms.reserve(4096);

    auto last       = Clock::now();
    auto wall_start = Clock::now();
    bool first      = true;

    while (g_running && frameQueue.pop(pkt))
    {
        auto now = Clock::now();
        if (!first)
            inter_ms.push_back(Ms(now - last).count());
        first = false;
        last  = now;
        ++frame_count;

        if (frame_count % CAMERA_FPS == 0 && inter_ms.size() > 1) {
            double wall_s = Ms(now - wall_start).count() / 1000.0;
            double mean   = std::accumulate(inter_ms.begin(), inter_ms.end(), 0.0)
                            / inter_ms.size();
            double mn     = *std::min_element(inter_ms.begin(), inter_ms.end());
            double mx     = *std::max_element(inter_ms.begin(), inter_ms.end());
            auto   s      = inter_ms; std::sort(s.begin(), s.end());

            std::cout << "Frames: "  << frame_count
                      << "  wall: "  << wall_s << "s"
                      << "  fps: "   << (frame_count / wall_s)
                      << "  mean="   << mean << "ms"
                      << "  min="    << mn   << "ms"
                      << "  max="    << mx   << "ms"
                      << "  p50="    << s[s.size()*50/100] << "ms"
                      << "  p95="    << s[s.size()*95/100] << "ms\n";
        }
    }

    capture.stop();
    frameQueue.stop();

    std::cout << "\nDone. Total frames: " << frame_count << "\n";
    return 0;
}
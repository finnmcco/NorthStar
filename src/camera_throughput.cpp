#include <iostream>
#include <atomic>
#include <csignal>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

#include "capture_controller.hpp"
#include "frame_packet.hpp"
#include "camera_queue.hpp"

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main()
{
    std::signal(SIGINT, on_sigint);


    queue<FramePacket> frameQueue(4);

    CameraCapture capture(frameQueue, 30);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    std::cout << "Camera throughput test — Ctrl-C to stop\n\n";

    FramePacket pkt{};
    int frame_count = 0;
    std::vector<double> inter_ms;
    inter_ms.reserve(4096);

    auto last      = Clock::now();
    auto wall_start = Clock::now();
    bool first     = true;

    while (g_running && frameQueue.pop(pkt))
    {
        auto now = Clock::now();

        if (!first) {
            double dt = Ms(now - last).count();
            inter_ms.push_back(dt);
        }
        first = false;
        last  = now;

        ++frame_count;

        // Print rolling stats every 30 frames
        if (frame_count % 30 == 0 && inter_ms.size() > 1) {
            double wall_s = Ms(now - wall_start).count() / 1000.0;
            double mean   = std::accumulate(inter_ms.begin(), inter_ms.end(), 0.0)
                            / inter_ms.size();
            double mn     = *std::min_element(inter_ms.begin(), inter_ms.end());
            double mx     = *std::max_element(inter_ms.begin(), inter_ms.end());

            std::vector<double> s = inter_ms;
            std::sort(s.begin(), s.end());
            double p50 = s[s.size() * 50 / 100];
            double p95 = s[s.size() * 95 / 100];

            std::cout << "Frames: "   << frame_count
                      << "  wall: "   << wall_s    << "s"
                      << "  fps: "    << (frame_count / wall_s)
                      << "  inter-frame mean=" << mean << "ms"
                      << "  min="     << mn    << "ms"
                      << "  max="     << mx    << "ms"
                      << "  p50="     << p50   << "ms"
                      << "  p95="     << p95   << "ms"
                      << "\n";
        }
    }

    capture.stop();
    frameQueue.stop();

    std::cout << "\nDone. Total frames: " << frame_count << "\n";
    return 0;
}
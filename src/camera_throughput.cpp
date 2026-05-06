#include <iostream>
#include <atomic>
#include <csignal>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"

static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr)
        g_controller_ptr->stop_capture();
}

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main()
{
    std::signal(SIGINT, on_sigint);

    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    std::cout << "Camera throughput test — Ctrl-C to stop\n\n";

    auto& cam_q = controller.get_cam_queue();

    int frame_count = 0;
    std::vector<double> inter_ms;
    inter_ms.reserve(4096);

    auto wall_start = Clock::now();
    auto last       = Clock::now();
    bool first      = true;

    while (g_running) {
        auto pkt = cam_q.pop();
        if (!pkt) break;

        auto now = Clock::now();

        if (!first)
            inter_ms.push_back(Ms(now - last).count());
        first = false;
        last  = now;

        ++frame_count;

        // Print rolling stats every 30 frames.
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

            std::cout << "Frames: "  << frame_count
                      << "  wall: "  << wall_s   << "s"
                      << "  fps: "   << (frame_count / wall_s)
                      << "  inter-frame mean=" << mean << "ms"
                      << "  min="    << mn   << "ms"
                      << "  max="    << mx   << "ms"
                      << "  p50="    << p50  << "ms"
                      << "  p95="    << p95  << "ms"
                      << "\n";
        }
    }

    std::cout << "\nDone. Total frames: " << frame_count << "\n";
    return 0;
}
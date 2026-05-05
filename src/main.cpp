
/*
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <vector>
#include "capture_controller.hpp"
#include "frame_saver.hpp"


int main() {
    std::cout << "Initialising capture controller..." << std::endl;
    CaptureController controller;

    std::cout << "Starting capture..." << std::endl;
    controller.start_capture();
    
    // Start saving frames concurrently on a separate thread
    std::thread saver_thread([&]() {
        FrameSaver saver(controller.get_cam_queue(), "./capture_output");
        saver.save_all();
    });

    // Block main thread for slightly longer than capture duration
    std::this_thread::sleep_for(std::chrono::seconds(6));
    
    // Join the saver thread
    if (saver_thread.joinable()) {
        saver_thread.join();
    }

    std::cout << "Capture complete." << std::endl;

    return 0;
}
*/

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include "capture_controller.hpp"
#include "button-driver.h"
#include <unistd.h>
#include <csignal>

static volatile sig_atomic_t g_should_exit = 0;
static void on_sigint(int) { g_should_exit = 1; }

struct CamFrameInfo {
    uint8_t  camera_id;
    uint64_t timestamp_us;
};

struct IRFrameInfo {
    uint64_t timestamp_us;
    float    ambient;
};

void print_stats(const std::string& name, const std::vector<uint64_t>& timestamps_us) {
    if (timestamps_us.size() < 2) {
        std::cout << name << ": " << timestamps_us.size() << " frames (not enough for stats)\n";
        return;
    }

    std::vector<int64_t> intervals;
    intervals.reserve(timestamps_us.size() - 1);
    for (size_t i = 1; i < timestamps_us.size(); ++i) {
        intervals.push_back(static_cast<int64_t>(timestamps_us[i] - timestamps_us[i-1]));
    }

    int64_t sum = std::accumulate(intervals.begin(), intervals.end(), int64_t{0});
    double  mean_us = static_cast<double>(sum) / intervals.size();
    int64_t min_us  = *std::min_element(intervals.begin(), intervals.end());
    int64_t max_us  = *std::max_element(intervals.begin(), intervals.end());

    uint64_t span_us = timestamps_us.back() - timestamps_us.front();
    double   fps     = (timestamps_us.size() - 1) * 1e6 / static_cast<double>(span_us);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << name << ": " << timestamps_us.size() << " frames, "
              << fps << " fps, "
              << "interval mean=" << mean_us / 1000.0 << "ms "
              << "min=" << min_us / 1000.0 << "ms "
              << "max=" << max_us / 1000.0 << "ms "
              << "jitter=" << (max_us - min_us) / 1000.0 << "ms\n";
}

int main() {
    std::signal(SIGINT, on_sigint);
    std::cout << "Initialising capture controller...\n";
    CaptureController controller;
    ButtonDriver btn;

    std::vector<CamFrameInfo> cam_frames;
    std::vector<IRFrameInfo>  ir_frames;
    cam_frames.reserve(500);
    ir_frames.reserve(50);

    btn.registerPressCallback([&](){
        std::cout << "Starting capture...\n";
        controller.start_capture();

    });

    btn.registerReleaseCallback([&](){
        std::cout << "Stopping capture...\n";
        controller.stop_capture();
    });

    std::thread cam_drain([&]() {
        auto& q = controller.get_cam_queue();
        while (auto pkt = q.pop()) {
            cam_frames.push_back({pkt->camera_id, pkt->timestamp_us});
        }
    });

    std::thread ir_drain([&]() {
        auto& q = controller.get_ir_queue();
        while (auto pkt = q.pop()) {
            ir_frames.push_back({pkt->timestamp_us, pkt->ambientTemp});
        }
    });

    // Wait for SIGINT to exit
    std::cout << "Press button to capture. Ctrl+C to exit.\n";
    pause();  // or signal handling

    // On exit:
    controller.stop_capture();  //ensure queues are stopped so drain threads exit
    controller.shutdown();
    cam_drain.join();
    ir_drain.join();

    std::cout << "\n=== Capture summary ===\n";

    // Split camera frames by ID
    std::vector<uint64_t> cam0_ts, cam1_ts;
    for (const auto& f : cam_frames) {
        if (f.camera_id == 0) cam0_ts.push_back(f.timestamp_us);
        else                  cam1_ts.push_back(f.timestamp_us);
    }
    print_stats("cam0", cam0_ts);
    print_stats("cam1", cam1_ts);

    // IR timestamps are in nanoseconds — convert to us for the same helper
    std::vector<uint64_t> ir_ts_us;
    ir_ts_us.reserve(ir_frames.size());
    for (const auto& f : ir_frames) ir_ts_us.push_back(f.timestamp_us);
    print_stats("IR  ", ir_ts_us);

    if (!ir_frames.empty()) {
        std::cout << "IR ambient: first=" << ir_frames.front().ambient
                  << "C last=" << ir_frames.back().ambient << "C\n";
    }

    std::cout << "Capture complete.\n";
    return 0;
}

/*
    capture_frames.cpp  --  save camera frames while button is held

    Starts capture on button press, stops on release.
    Every frame from both cameras is saved as a PNG into a timestamped
    session directory.

    File naming:
        capture_frames/<session>/frame_cam<id>_<timestamp_ns>.png

    Run as:
        ./capture_frames
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "button-driver.h"
#include "gpio.h"

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

// -- Globals ------------------------------------------------------------------
std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;
static std::string g_output_dir;
static std::atomic<uint64_t> g_frames_saved{0};

static void on_signal(int)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}

static std::string make_session_dir()
{
    using namespace std::chrono;
    const auto now = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);

    std::ostringstream oss;
    oss << "../" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    const std::string dir = oss.str();

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "[main] FATAL: could not create %s -- %s\n",
                     dir.c_str(), ec.message().c_str());
        return {};
    }
    return dir;
}

// -- main ---------------------------------------------------------------------
int main(int /*argc*/, char* /*argv*/[])
{
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    g_output_dir = make_session_dir();
    if (g_output_dir.empty()) return 1;
    std::printf("[main] output dir: %s\n", g_output_dir.c_str());

    gpio::setupGpio();

    CaptureController controller;
    g_controller_ptr = &controller;

    // -- Camera consumer thread
    std::thread cam_thread([&]() {
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;

            // Build filename: frame_cam<id>_<timestamp_ns>.png
            char filename[256];
            std::snprintf(filename, sizeof(filename),
                          "%s/frame_cam%d_%llu.png",
                          g_output_dir.c_str(),
                          pkt->camera_id,
                          static_cast<unsigned long long>(ts_ns));

            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            cv::imwrite(filename, frame);

            const uint64_t n = ++g_frames_saved;
            std::printf("[cam] saved frame #%llu  cam=%d  ts=%llu\n",
                        static_cast<unsigned long long>(n),
                        pkt->camera_id,
                        static_cast<unsigned long long>(ts_ns));
            std::fflush(stdout);
        }
        std::printf("[cam] thread exiting -- %llu frames saved\n",
                    static_cast<unsigned long long>(g_frames_saved.load()));
    });

    // -- IR drain thread (must be consumed to avoid back-pressure)
    std::thread ir_thread([&]() {
        auto& ir_q = controller.get_ir_queue();
        while (auto pkt = ir_q.pop()) { (void)pkt; }
    });

    // -- Button
    button_driver::ButtonDriver btn;

    btn.registerPressCallback([&]() {
        std::printf("\n[button] PRESS -- starting capture\n");
        std::fflush(stdout);
        controller.start_capture();
    });

    btn.registerReleaseCallback([&]() {
        std::printf("\n[button] RELEASE -- stopping capture\n");
        std::printf("[button] %llu frames saved so far\n",
                    static_cast<unsigned long long>(g_frames_saved.load()));
        std::fflush(stdout);
        controller.stop_capture();
    });

    std::printf("\n----------------------------------------------------\n");
    std::printf(" capture_frames ready.\n");
    std::printf("  - Hold button to capture frames.\n");
    std::printf("  - Release to stop.\n");
    std::printf("  - Ctrl-C to exit.\n");
    std::printf("  - Output: %s\n", g_output_dir.c_str());
    std::printf("----------------------------------------------------\n\n");
    std::fflush(stdout);

    while (g_running)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    controller.stop_capture();
    controller.shutdown();
    cam_thread.join();
    ir_thread.join();
    gpio::teardownGpio();

    std::printf("\n[main] done -- %llu frames saved to %s\n",
                static_cast<unsigned long long>(g_frames_saved.load()),
                g_output_dir.c_str());
    return 0;
}
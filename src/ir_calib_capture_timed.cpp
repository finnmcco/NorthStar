/*
    ir_calib_capture.cpp  —  cam0 + IR calibration capture tool

    Captures synchronised cam0 + IR frame pairs automatically, one per second.
    Run the binary, hold the target object steady, and it will save pairs
    continuously until you press Ctrl-C.

    Usage:
        ./ir_calib_capture [output_dir]

        output_dir  — optional, defaults to "ir_calib_50cm"

    Output per capture:
        <output_dir>/pair_NN_cam0.png    — 640x640 BGR from camera 0
        <output_dir>/pair_NN_ir.png      — IR rendered as false-colour heatmap
        <output_dir>/pair_NN_ir.csv      — raw temperatures, °C, sensor-native order

    Indices continue from where a previous session left off, so you can
    re-run without overwriting existing pairs.
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "ir_queue.hpp"
#include "frame_packet.hpp"
#include "MLX90640.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// ─── Capture interval ────────────────────────────────────────────────────────
static constexpr auto CAPTURE_INTERVAL = std::chrono::milliseconds(1000);

// ─── Globals for signal handling ─────────────────────────────────────────────
std::atomic<bool> g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_signal(int)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}

// ─── Latest-frame holders (mutex-protected) ──────────────────────────────────
static std::mutex                 g_cam_mtx;
static std::optional<FramePacket> g_latest_cam0;

static std::mutex                 g_ir_mtx;
static std::optional<IRPacket>    g_latest_ir;

// ─── IR rendering ─────────────────────────────────────────────────────────────
static constexpr int IR_WIDTH        = 32;
static constexpr int IR_HEIGHT       = 24;
static constexpr int IR_DISPLAY_SCALE = 16;

static cv::Mat render_ir(const std::array<float, MLX90640::PIXEL_COUNT>& temps)
{
    cv::Mat raw(IR_HEIGHT, IR_WIDTH, CV_32FC1);
    for (int y = 0; y < IR_HEIGHT; ++y)
        for (int x = 0; x < IR_WIDTH; ++x)
            raw.at<float>(y, x) = temps[y * IR_WIDTH + x];

    double lo, hi;
    cv::minMaxLoc(raw, &lo, &hi);
    const double span = std::max(hi - lo, 0.1);

    cv::Mat norm;
    raw.convertTo(norm, CV_8UC1, 255.0 / span, -lo * 255.0 / span);

    cv::Mat coloured;
    cv::applyColorMap(norm, coloured, cv::COLORMAP_INFERNO);

    cv::Mat upscaled;
    cv::resize(coloured, upscaled,
               cv::Size(IR_WIDTH * IR_DISPLAY_SCALE, IR_HEIGHT * IR_DISPLAY_SCALE),
               0, 0, cv::INTER_NEAREST);

    char label[64];
    std::snprintf(label, sizeof(label), "min=%.1fC max=%.1fC", lo, hi);
    cv::putText(upscaled, label, {8, 24},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);

    return upscaled;
}

// ─── Find next pair index ─────────────────────────────────────────────────────
static int next_pair_index(const fs::path& dir)
{
    if (!fs::exists(dir)) return 1;
    int max_idx = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.size() < 10 || name.substr(0, 5) != "pair_") continue;
        const auto under_pos = name.find('_', 5);
        if (under_pos == std::string::npos) continue;
        try {
            int idx = std::stoi(name.substr(5, under_pos - 5));
            if (idx > max_idx) max_idx = idx;
        } catch (...) {}
    }
    return max_idx + 1;
}

// ─── Save one synchronised pair ──────────────────────────────────────────────
static bool save_pair(int idx,
                      const fs::path& out_dir,
                      const FramePacket& cam_pkt,
                      const IRPacket& ir_pkt)
{
    char base[32];
    std::snprintf(base, sizeof(base), "pair_%02d", idx);

    // cam0 PNG
    cv::Mat cam_frame(640, 640, CV_8UC3,
                      const_cast<uint8_t*>(cam_pkt.data.data()));
    const fs::path cam_path = out_dir / (std::string(base) + "_cam0.png");
    if (!cv::imwrite(cam_path.string(), cam_frame)) {
        std::fprintf(stderr, "[save] failed to write %s\n", cam_path.string().c_str());
        return false;
    }

    // IR PNG (false-colour heatmap for viewing)
    const fs::path ir_png = out_dir / (std::string(base) + "_ir.png");
    cv::imwrite(ir_png.string(), render_ir(ir_pkt.temps));

    // IR CSV (raw temperatures)
    const fs::path ir_csv = out_dir / (std::string(base) + "_ir.csv");
    std::ofstream f(ir_csv);
    if (!f) {
        std::fprintf(stderr, "[save] failed to write %s\n", ir_csv.string().c_str());
        return false;
    }
    for (int y = 0; y < IR_HEIGHT; ++y) {
        for (int x = 0; x < IR_WIDTH; ++x) {
            if (x > 0) f << ',';
            f << std::fixed << ir_pkt.temps[y * IR_WIDTH + x];
        }
        f << '\n';
    }

    // Log summary
    float t_min = ir_pkt.temps[0], t_max = ir_pkt.temps[0];
    for (float t : ir_pkt.temps) {
        if (t < t_min) t_min = t;
        if (t > t_max) t_max = t;
    }
    std::printf("[saved] pair %02d  ambient=%.1fC  IR min=%.1fC max=%.1fC\n",
                idx, ir_pkt.ambientTemp, t_min, t_max);
    std::fflush(stdout);
    return true;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Output directory from command line, or default
    const fs::path out_dir = (argc > 1) ? argv[1] : "ir_calib_50cm";
    fs::create_directories(out_dir);
    int pair_idx = next_pair_index(out_dir);

    std::printf("ir_calib_capture  (automatic, 1 pair/second)\n");
    std::printf("Output dir  : %s/\n", fs::absolute(out_dir).string().c_str());
    std::printf("Next pair   : #%02d\n", pair_idx);
    std::printf("Interval    : %lld ms\n",
                static_cast<long long>(CAPTURE_INTERVAL.count()));
    std::printf("Initialising sensors...\n");

    CaptureController controller;
    g_controller_ptr = &controller;

    // Drain cam0 frames — keep newest
    std::thread cam_thread([&]() {
        auto& q = controller.get_cam_queue();
        while (auto pkt = q.pop()) {
            if (pkt->camera_id != 0) continue;
            std::lock_guard<std::mutex> lk(g_cam_mtx);
            g_latest_cam0 = std::move(*pkt);
        }
    });

    // Drain IR frames — keep newest
    std::thread ir_thread([&]() {
        auto& q = controller.get_ir_queue();
        while (auto pkt = q.pop()) {
            std::lock_guard<std::mutex> lk(g_ir_mtx);
            g_latest_ir = std::move(*pkt);
        }
    });

    controller.start_capture();

    // Brief settle period so AE/AWB converge and the IR sensor has at least
    // one complete frame before the first capture fires.
    std::printf("Settling for 3 seconds before first capture...\n");
    for (int i = 3; i > 0 && g_running; --i) {
        std::printf("  %d...\n", i);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::printf("\nCapturing — Ctrl-C to stop.\n\n");

    // Timed capture loop
    while (g_running) {
        auto next_tick = std::chrono::steady_clock::now() + CAPTURE_INTERVAL;

        // Snapshot latest frames under their respective locks
        std::optional<FramePacket> cam_snap;
        std::optional<IRPacket>    ir_snap;
        {
            std::lock_guard<std::mutex> lk(g_cam_mtx);
            cam_snap = g_latest_cam0;
        }
        {
            std::lock_guard<std::mutex> lk(g_ir_mtx);
            ir_snap = g_latest_ir;
        }

        if (!cam_snap) {
            std::printf("[skip] no cam0 frame yet\n");
        } else if (!ir_snap) {
            std::printf("[skip] no IR frame yet\n");
        } else {
            if (save_pair(pair_idx, out_dir, *cam_snap, *ir_snap)) {
                ++pair_idx;
            }
        }

        // Sleep until the next tick, checking g_running so Ctrl-C feels instant
        while (g_running && std::chrono::steady_clock::now() < next_tick) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // Shutdown
    std::printf("\nShutting down...\n");
    controller.stop_capture();
    controller.shutdown();
    cam_thread.join();
    ir_thread.join();
    std::printf("Done. Saved up to pair %02d in %s/\n",
                pair_idx - 1, out_dir.string().c_str());
    return 0;
}
/*
    ir_calib_capture.cpp  —  cam0 + IR calibration capture tool

    Captures synchronised cam0 + IR frame pairs on demand. Each capture
    is triggered by pressing ENTER in the terminal. Outputs go into
    ir_calib/ alongside auto-numbered filenames.

    Usage:
        ./ir_calib_capture
        (then press ENTER to capture, Ctrl-C to exit)

    Output per capture:
        ir_calib/pair_NN_cam0.png    — 640x640 BGR from camera 0
        ir_calib/pair_NN_ir.png      — IR rendered as false-colour heatmap
        ir_calib/pair_NN_ir.csv      — raw temperatures, °C, sensor-native order

    Tip for finding the next pair index: the script scans existing files
    in ir_calib/ at startup. You can re-run across sessions and indices
    will continue from where you left off.
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

// ─── Constants ───────────────────────────────────────────────────────────────
static constexpr const char* OUTPUT_DIR    = "ir_calib_50cm";
static constexpr int         IR_WIDTH      = 32;
static constexpr int         IR_HEIGHT     = 24;
static constexpr int         IR_DISPLAY_SCALE = 16;

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

// ─── Render IR as a false-coloured upscaled preview ──────────────────────────
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

// ─── Find next pair index ────────────────────────────────────────────────────
static int next_pair_index(const fs::path& dir)
{
    if (!fs::exists(dir)) return 1;
    int max_idx = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.size() < 10 || name.substr(0, 5) != "pair_") continue;
        // Parse the number between "pair_" and the next "_"
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
static bool save_pair(int idx, const FramePacket& cam_pkt, const IRPacket& ir_pkt)
{
    char base[32];
    std::snprintf(base, sizeof(base), "pair_%02d", idx);

    // cam0 PNG
    cv::Mat cam_frame(640, 640, CV_8UC3,
                      const_cast<uint8_t*>(cam_pkt.data.data()));
    const fs::path cam_path = fs::path(OUTPUT_DIR) / (std::string(base) + "_cam0.png");
    if (!cv::imwrite(cam_path.string(), cam_frame)) {
        std::fprintf(stderr, "[save] failed to write %s\n", cam_path.string().c_str());
        return false;
    }

    // IR PNG (false-colour heatmap)
    cv::Mat ir_preview = render_ir(ir_pkt.temps);
    const fs::path ir_png = fs::path(OUTPUT_DIR) / (std::string(base) + "_ir.png");
    cv::imwrite(ir_png.string(), ir_preview);

    // IR CSV (raw temperatures, sensor-native pixel order)
    const fs::path ir_csv = fs::path(OUTPUT_DIR) / (std::string(base) + "_ir.csv");
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

    // Compute ambient + hot-spot temperatures for logging
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

int main()
{
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    fs::create_directories(OUTPUT_DIR);
    int pair_idx = next_pair_index(OUTPUT_DIR);

    std::printf("ir_calib_capture\n");
    std::printf("Output dir: %s/\n", fs::absolute(OUTPUT_DIR).string().c_str());
    std::printf("Next pair will be #%02d\n", pair_idx);
    std::printf("Initialising sensors...\n");

    CaptureController controller;
    g_controller_ptr = &controller;

    // ── Drain cam0 frames: keep only camera_id==0, keep newest ───────────────
    std::thread cam_thread([&]() {
        auto& q = controller.get_cam_queue();
        while (auto pkt = q.pop()) {
            if (pkt->camera_id != 0) continue;  // discard cam1, not needed here
            std::lock_guard<std::mutex> lk(g_cam_mtx);
            g_latest_cam0 = std::move(*pkt);
        }
    });

    // ── Drain IR frames: keep newest ─────────────────────────────────────────
    std::thread ir_thread([&]() {
        auto& q = controller.get_ir_queue();
        while (auto pkt = q.pop()) {
            std::lock_guard<std::mutex> lk(g_ir_mtx);
            g_latest_ir = std::move(*pkt);
        }
    });

    controller.start_capture();
    std::printf("Capture started — let sensors settle for a moment.\n");
    std::printf("\nPress ENTER to capture a pair. Ctrl-C to exit.\n\n");

    // ── Capture loop: blocks on stdin getline ────────────────────────────────
    std::string line;
    while (g_running && std::getline(std::cin, line)) {
        // Snapshot the latest cam0 and IR frames
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
            std::printf("[skip] no cam0 frame yet — wait a moment\n");
            continue;
        }
        if (!ir_snap) {
            std::printf("[skip] no IR frame yet — wait a moment\n");
            continue;
        }

        if (save_pair(pair_idx, *cam_snap, *ir_snap)) {
            ++pair_idx;
        }
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    std::printf("\nShutting down...\n");
    controller.stop_capture();
    controller.shutdown();
    cam_thread.join();
    ir_thread.join();
    std::printf("Done. Captured up to pair %02d in %s/\n",
                pair_idx - 1, OUTPUT_DIR);
    return 0;
}
#include "capture/image_capture.hpp"

#include <algorithm>   // std::sort
#include <chrono>
#include <ctime>       // clock_gettime
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

ImageCapture::ImageCapture(const std::string&  root_dir,
                            queue<FramePacket>& outQueue,
                            BufferPool&         pool,
                            int                 fps,
                            bool                loop)
    : root_dir_(root_dir)
    , outQueue_(outQueue)
    , pool_(pool)
    , fps_(fps)
    , loop_(loop)
{}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

bool ImageCapture::start()
{
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
        return false; // already running

    if (fps_ <= 0) {
        running_ = false;
        return false;
    }

    // Scan cam0/ and cam1/ under root_dir.
    fs::path root(root_dir_);
    paths_cam0_ = load_image_paths(root / "cam0");
    paths_cam1_ = load_image_paths(root / "cam1");

    // Fall back to loading root itself as cam0 only if no subfolders.
    if (paths_cam0_.empty() && paths_cam1_.empty())
        paths_cam0_ = load_image_paths(root);

    if (paths_cam0_.empty()) {
        std::cerr << "[ImageCapture] No images found under: " << root_dir_ << "\n";
        running_ = false;
        return false;
    }

    std::cout << "[ImageCapture] cam0: " << paths_cam0_.size() << " images\n";
    if (!paths_cam1_.empty())
        std::cout << "[ImageCapture] cam1: " << paths_cam1_.size() << " images\n";

    worker_ = std::thread(&ImageCapture::run, this);
    return true;
}

void ImageCapture::stop()
{
    running_ = false;
    if (worker_.joinable())
        worker_.join();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Worker
// ─────────────────────────────────────────────────────────────────────────────

void ImageCapture::run()
{
    const auto frame_period =
        std::chrono::microseconds(1'000'000 / fps_);

    const bool stereo = !paths_cam1_.empty();

    std::size_t idx = 0; // index into cam0 (and cam1) lists

    while (running_)
    {
        // ── Determine which images to push this iteration ─────────────────────
        const std::size_t cam0_total = paths_cam0_.size();
        const std::size_t cam1_total = stereo ? paths_cam1_.size() : 0;

        // Number of pairs we can form
        const std::size_t pair_count = stereo
            ? std::min(cam0_total, cam1_total)
            : cam0_total;

        if (idx >= pair_count) {
            if (loop_) {
                idx = 0;         // wrap around
            } else {
                break;           // single pass done
            }
        }

        const auto t_frame_start = std::chrono::steady_clock::now();

        // ── cam0 ─────────────────────────────────────────────────────────────
        {
            uint8_t* buf = pool_.try_acquire();
            if (!buf) {
                // Pool exhausted — downstream too slow, skip this frame.
                std::cerr << "[ImageCapture] pool exhausted, dropping frame " << idx << "\n";
            } else if (decode_into(paths_cam0_[idx], buf)) {
                FramePacket pkt;
                pkt.camera_id = 0;
                pkt.timestamp = now_ns();
                pkt.data      = buf;

                if (!outQueue_.push(pkt)) {
                    pool_.release(buf);
                    break; // queue stopped
                }
            } else {
                pool_.release(buf); // decode failed — buffer back to pool
            }
        }

        // ── cam1 (if present) ────────────────────────────────────────────────
        if (stereo) {
            uint8_t* buf = pool_.try_acquire();
            if (!buf) {
                std::cerr << "[ImageCapture] pool exhausted, dropping cam1 frame " << idx << "\n";
            } else if (decode_into(paths_cam1_[idx], buf)) {
                FramePacket pkt;
                pkt.camera_id = 1;
                pkt.timestamp = now_ns(); // slightly after cam0 — within tolerance
                pkt.data      = buf;

                if (!outQueue_.push(pkt)) {
                    pool_.release(buf);
                    break;
                }
            } else {
                pool_.release(buf);
            }
        }

        ++idx;

        // ── Pace to target FPS ────────────────────────────────────────────────
        const auto elapsed = std::chrono::steady_clock::now() - t_frame_start;
        if (elapsed < frame_period)
            std::this_thread::sleep_for(frame_period - elapsed);
    }

    // Signal the queue that no more frames are coming.
    outQueue_.stop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

std::uint64_t ImageCapture::now_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec)  * 1'000'000'000ULL
         + static_cast<std::uint64_t>(ts.tv_nsec);
}

std::vector<fs::path>
ImageCapture::load_image_paths(const fs::path& dir)
{
    std::vector<fs::path> paths;

    if (!fs::exists(dir) || !fs::is_directory(dir))
        return paths;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg"
                          || ext == ".PNG" || ext == ".JPG" || ext == ".JPEG")
            paths.push_back(entry.path());
    }

    std::sort(paths.begin(), paths.end()); // deterministic order
    return paths;
}

bool ImageCapture::decode_into(const fs::path& path, uint8_t* dst)
{
    // OpenCV loads as BGR
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "[ImageCapture] Failed to decode: " << path << "\n";
        return false;
    }

    if (bgr.cols != kWidth || bgr.rows != kHeight) {
        std::cerr << "[ImageCapture] Wrong dimensions (" << bgr.cols << "×" << bgr.rows
                  << "), expected " << kWidth << "×" << kHeight
                  << ": " << path << "\n";
        return false;
    }

    // Convert BGR → RGB to match the format coming from CameraCapture
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    if (!rgb.isContinuous())
        rgb = rgb.clone();

    std::memcpy(dst, rgb.data, static_cast<std::size_t>(kWidth * kHeight * kChannels));
    return true;
}

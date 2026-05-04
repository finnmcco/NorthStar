#include "image_capture.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool is_image_file(const fs::path& p)
{
    const std::string ext = p.extension().string();
    for (char& c : const_cast<std::string&>(ext))
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

// Collect image paths from a directory, sorted alphabetically.
static std::vector<fs::path> collect_images(const fs::path& dir)
{
    std::vector<fs::path> paths;
    if (!fs::is_directory(dir)) return paths;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_image_file(entry.path()))
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Load, resize to 640×640, convert to RGB.  Returns empty Mat on failure.
static cv::Mat load_frame(const fs::path& p)
{
    cv::Mat bgr = cv::imread(p.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "[ImageCapture] Could not load: " << p << "\n";
        return {};
    }
    if (bgr.cols != 640 || bgr.rows != 640)
        cv::resize(bgr, bgr, cv::Size(640, 640));

    // Convert to RGB — matches the live camera pipeline output.
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    return rgb;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

ImageCapture::ImageCapture(const std::string&  image_root,
                           queue<FramePacket>& q,
                           BufferPool&         pool,
                           int                 fps,
                           bool                loop)
    : image_root_(image_root), queue_(q), pool_(pool), fps_(fps), loop_(loop)
{}

ImageCapture::~ImageCapture()
{
    stop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  start / stop
// ─────────────────────────────────────────────────────────────────────────────

bool ImageCapture::start()
{
    // Quick check: is there at least one image to send?
    const auto cam0_images = collect_images(fs::path(image_root_) / "cam0");
    if (cam0_images.empty()) {
        std::cerr << "[ImageCapture] No images found in "
                  << image_root_ << "/cam0/\n";
        return false;
    }

    running_ = true;
    worker_  = std::thread(&ImageCapture::run, this);
    return true;
}

void ImageCapture::stop()
{
    running_ = false;
    if (worker_.joinable())
        worker_.join();
}

// ─────────────────────────────────────────────────────────────────────────────
//  run  (worker thread)
// ─────────────────────────────────────────────────────────────────────────────

void ImageCapture::run()
{
    const auto cam0_images = collect_images(fs::path(image_root_) / "cam0");
    const auto cam1_images = collect_images(fs::path(image_root_) / "cam1");
    const bool dual = !cam1_images.empty();

    const std::size_t n = cam0_images.size();
    std::cout << "[ImageCapture] " << n << " cam0 images"
              << (dual ? ", " + std::to_string(cam1_images.size()) + " cam1 images" : "")
              << " — " << fps_ << " fps"
              << (loop_ ? " (loop)" : "") << "\n";

    const auto frame_period =
        std::chrono::microseconds(1'000'000 / std::max(fps_, 1));

    uint64_t fake_ts_ns = 0;
    constexpr uint64_t ts_step_ns = 33'333'333ULL; // ~30 fps in ns

    do {
        for (std::size_t i = 0; i < n && running_; ++i)
        {
            // ── Push cam0 frame ───────────────────────────────────────────────
            {
                cv::Mat rgb = load_frame(cam0_images[i]);
                if (!rgb.empty())
                {
                    uint8_t* buf = nullptr;
                    // Spin-wait for a pool buffer (avoid blocking libcamera thread).
                    while (running_ && !(buf = pool_.try_acquire()))
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));

                    if (buf) {
                        std::memcpy(buf, rgb.data, 640 * 640 * 3);

                        FramePacket pkt;
                        pkt.camera_id = 0;
                        pkt.timestamp = fake_ts_ns;
                        pkt.data      = buf;

                        if (!queue_.push(pkt))
                            pool_.release(buf); // queue stopped
                    }
                }
            }

            // ── Push matching cam1 frame (if available) ───────────────────────
            if (dual && i < cam1_images.size())
            {
                cv::Mat rgb = load_frame(cam1_images[i]);
                if (!rgb.empty())
                {
                    uint8_t* buf = nullptr;
                    while (running_ && !(buf = pool_.try_acquire()))
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));

                    if (buf) {
                        std::memcpy(buf, rgb.data, 640 * 640 * 3);

                        FramePacket pkt;
                        pkt.camera_id = 1;
                        pkt.timestamp = fake_ts_ns; // same ts → StereoMatcher pairs them
                        pkt.data      = buf;

                        if (!queue_.push(pkt))
                            pool_.release(buf);
                    }
                }
            }

            fake_ts_ns += ts_step_ns;
            std::this_thread::sleep_for(frame_period);
        }
    } while (loop_ && running_);

    // Signal that no more frames will be pushed.
    queue_.stop();
    std::cout << "[ImageCapture] Done\n";
}
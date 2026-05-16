/*
    stereo_debug_main.cpp  --  one-shot stereo depth diagnostic (button-only)

    Hold the button to arm the filter for the COCO object passed as argument.
    Release to stop capture and reset the filter.

    On each FilteredInferencePair the program:
      1. Computes stereo depth.
      2. Saves three diagnostic PNGs into stereo_debug/<session>/:
           pair_0001_unrectified.png
           pair_0001_rectified.png
           pair_0001_disparity.png
      3. Exits after SAVE_PAIRS pairs.

    Run as:
      ./stereo_debug_main <word>
      e.g. ./stereo_debug_main cup
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "coco_lookup.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"
#include "button-driver.h"
#include "gpio.h"
#include "hailo8_inference.hpp"
#include "frame_buffer.hpp"
#include "stereo_distance.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

static constexpr int SAVE_PAIRS = 20;

// -- Globals ------------------------------------------------------------------
std::atomic<bool>     g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static std::atomic<uint64_t> g_cam_frames_consumed{0};
static std::atomic<uint64_t> g_ir_frames_consumed{0};
static std::atomic<uint64_t> g_hailo_callbacks{0};
static std::atomic<uint64_t> g_pairs_emitted{0};
static std::atomic<int>      g_pairs_saved{0};

static FrameBuffer g_frame_buf_cam0;
static FrameBuffer g_frame_buf_cam1;
static StereoDepthEstimator* g_depth_ptr = nullptr;

static std::string g_output_dir;
static std::mutex  g_save_mutex;

// The single target object ID resolved from the command-line word.
static uint8_t g_target_id = 0;

static void on_signal(int)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}

// -- Helpers ------------------------------------------------------------------

static std::string make_session_dir()
{
    using namespace std::chrono;
    const auto now = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);

    std::ostringstream oss;
    oss << "stereo_debug/" << std::put_time(&tm, "%Y%m%d_%H%M%S");
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

static void draw_norm_bbox(cv::Mat& img, const BoundingBox& box,
                           const cv::Scalar& colour, const std::string& label)
{
    const int W = img.cols, H = img.rows;
    const int x0 = std::clamp(static_cast<int>(box.x_min * W), 0, W - 1);
    const int y0 = std::clamp(static_cast<int>(box.y_min * H), 0, H - 1);
    const int x1 = std::clamp(static_cast<int>(box.x_max * W), 0, W - 1);
    const int y1 = std::clamp(static_cast<int>(box.y_max * H), 0, H - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, colour, 2);
    if (!label.empty())
        cv::putText(img, label, {x0 + 4, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv::LINE_AA);
}

static void draw_pixel_bbox(cv::Mat& img, int x0, int y0, int x1, int y1,
                            const cv::Scalar& colour, const std::string& label)
{
    x0 = std::clamp(x0, 0, img.cols - 1);
    y0 = std::clamp(y0, 0, img.rows - 1);
    x1 = std::clamp(x1, 0, img.cols - 1);
    y1 = std::clamp(y1, 0, img.rows - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, colour, 2);
    if (!label.empty())
        cv::putText(img, label, {x0 + 4, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv::LINE_AA);
}

static cv::Mat side_by_side(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat out;
    cv::hconcat(left, right, out);
    return out;
}

static cv::Rect rectify_bbox(const BoundingBox& box,
                             const cv::Mat& K, const cv::Mat& D,
                             const cv::Mat& R, const cv::Mat& P,
                             const cv::Size& image_size)
{
    const float W = static_cast<float>(image_size.width);
    const float H = static_cast<float>(image_size.height);

    std::vector<cv::Point2f> corners_raw = {
        { box.x_min * W, box.y_min * H },
        { box.x_max * W, box.y_min * H },
        { box.x_max * W, box.y_max * H },
        { box.x_min * W, box.y_max * H },
    };
    std::vector<cv::Point2f> corners_rect;
    cv::undistortPoints(corners_raw, corners_rect, K, D, R, P);

    float min_x =  std::numeric_limits<float>::infinity();
    float min_y =  std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    for (const auto& p : corners_rect) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    if (max_x < 0 || max_y < 0 ||
        min_x > image_size.width  - 1 ||
        min_y > image_size.height - 1)
        return {};

    int rx0 = std::max(0, static_cast<int>(std::floor(min_x)));
    int ry0 = std::max(0, static_cast<int>(std::floor(min_y)));
    int rx1 = std::min(image_size.width  - 1, static_cast<int>(std::ceil(max_x)));
    int ry1 = std::min(image_size.height - 1, static_cast<int>(std::ceil(max_y)));
    if (rx1 <= rx0 || ry1 <= ry0) return {};
    return cv::Rect(cv::Point(rx0, ry0), cv::Point(rx1 + 1, ry1 + 1));
}

static cv::Mat colourise_disparity(const cv::Mat& disparity_px)
{
    cv::Mat valid_mask = disparity_px > 0;
    double dmin = 0, dmax = 0;
    cv::minMaxLoc(disparity_px, &dmin, &dmax, nullptr, nullptr, valid_mask);
    if (dmax <= dmin) dmax = dmin + 1.0;

    cv::Mat normed;
    disparity_px.convertTo(normed, CV_8U,
                           255.0 / (dmax - dmin),
                           -255.0 * dmin / (dmax - dmin));
    cv::Mat coloured;
    cv::applyColorMap(normed, coloured, cv::COLORMAP_JET);
    coloured.setTo(cv::Scalar(0, 0, 0), ~valid_mask);
    return coloured;
}

// -- Diagnostic PNG save ------------------------------------------------------
static void save_diagnostic_pngs(int pair_idx,
                                 const cv::Mat& left_raw,
                                 const cv::Mat& right_raw,
                                 const Detection& best_cam0,
                                 const Detection* best_cam1,
                                 std::optional<float> depth_m)
{
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "%s/pair_%04d",
                  g_output_dir.c_str(), pair_idx);

    // 1. Unrectified side-by-side
    {
        cv::Mat l = left_raw.clone();
        cv::Mat r = right_raw.clone();
        draw_norm_bbox(l, best_cam0.box, cv::Scalar(0, 255, 0), "cam0 raw");
        if (best_cam1)
            draw_norm_bbox(r, best_cam1->box, cv::Scalar(0, 255, 0), "cam1 raw");
        else
            cv::putText(r, "(no cam1 detection)", {8, 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        const std::string path = std::string(prefix) + "_unrectified.png";
        cv::imwrite(path, side_by_side(l, r))
            ? std::printf("[debug] saved %s\n", path.c_str())
            : std::printf("[debug] FAILED to save %s\n", path.c_str());
    }

    // 2. Rectified side-by-side
    {
        const cv::Mat& lrect = g_depth_ptr->last_left_rect();
        const cv::Mat& rrect = g_depth_ptr->last_right_rect();
        if (lrect.empty() || rrect.empty()) {
            std::printf("[debug] WARNING: rectified frames not available\n");
        } else {
            cv::Mat l = lrect.clone();
            cv::Mat r = rrect.clone();

            cv::Rect rb_l = rectify_bbox(best_cam0.box,
                g_depth_ptr->K1(), g_depth_ptr->D1(),
                g_depth_ptr->R1(), g_depth_ptr->P1(), l.size());
            if (rb_l.area() > 0)
                draw_pixel_bbox(l, rb_l.x, rb_l.y,
                                rb_l.x + rb_l.width - 1,
                                rb_l.y + rb_l.height - 1,
                                cv::Scalar(0, 255, 0), "cam0 rect");

            if (best_cam1) {
                cv::Rect rb_r = rectify_bbox(best_cam1->box,
                    g_depth_ptr->K2(), g_depth_ptr->D2(),
                    g_depth_ptr->R2(), g_depth_ptr->P2(), r.size());
                if (rb_r.area() > 0)
                    draw_pixel_bbox(r, rb_r.x, rb_r.y,
                                    rb_r.x + rb_r.width - 1,
                                    rb_r.y + rb_r.height - 1,
                                    cv::Scalar(0, 255, 0), "cam1 rect");
            }

            const std::string path = std::string(prefix) + "_rectified.png";
            cv::imwrite(path, side_by_side(l, r))
                ? std::printf("[debug] saved %s\n", path.c_str())
                : std::printf("[debug] FAILED to save %s\n", path.c_str());
        }
    }

    // 3. Disparity map
    {
        const cv::Mat& disp = g_depth_ptr->last_disparity();
        if (disp.empty()) {
            std::printf("[debug] WARNING: disparity not available\n");
        } else {
            cv::Mat coloured = colourise_disparity(disp);

            cv::Rect rb_l = rectify_bbox(best_cam0.box,
                g_depth_ptr->K1(), g_depth_ptr->D1(),
                g_depth_ptr->R1(), g_depth_ptr->P1(), coloured.size());

            std::string label = "depth: ";
            if (depth_m) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.2f m", *depth_m);
                label += buf;
            } else {
                label += "n/a";
            }

            if (rb_l.area() > 0)
                draw_pixel_bbox(coloured, rb_l.x, rb_l.y,
                                rb_l.x + rb_l.width - 1,
                                rb_l.y + rb_l.height - 1,
                                cv::Scalar(255, 255, 255), label);
            else
                cv::putText(coloured, label, {8, 24},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            const std::string path = std::string(prefix) + "_disparity.png";
            cv::imwrite(path, coloured)
                ? std::printf("[debug] saved %s\n", path.c_str())
                : std::printf("[debug] FAILED to save %s\n", path.c_str());
        }
    }
}

// -- Filter callback ----------------------------------------------------------
static void on_filtered_pair(FilteredInferencePair pair)
{
    const uint64_t n = ++g_pairs_emitted;

    std::printf("\n[filter] PAIR #%llu  object_id=%-3d  ts=%-12llu  "
                "cam0=%zu det  cam1=%zu det\n",
                static_cast<unsigned long long>(n),
                pair.object_id,
                static_cast<unsigned long long>(pair.timestamp_avg),
                pair.cam0_detections.size(),
                pair.cam1_detections.size());

    if (pair.cam0_detections.empty() || pair.cam1_detections.empty()) {
        std::printf("[filter] pair missing detections on a side; skipping\n");
        std::fflush(stdout);
        return;
    }

    const Detection* best_cam0 = &pair.cam0_detections.front();
    for (const auto& d : pair.cam0_detections)
        if (d.confidence > best_cam0->confidence) best_cam0 = &d;

    const Detection* best_cam1 = &pair.cam1_detections.front();
    for (const auto& d : pair.cam1_detections)
        if (d.confidence > best_cam1->confidence) best_cam1 = &d;

    std::printf("[filter]   cam0  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                best_cam0->confidence,
                best_cam0->box.x_min, best_cam0->box.y_min,
                best_cam0->box.x_max, best_cam0->box.y_max);
    std::printf("[filter]   cam1  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                best_cam1->confidence,
                best_cam1->box.x_min, best_cam1->box.y_min,
                best_cam1->box.x_max, best_cam1->box.y_max);

    cv::Mat left  = g_frame_buf_cam0.find_closest(pair.timestamp_avg);
    cv::Mat right = g_frame_buf_cam1.find_closest(pair.timestamp_avg);
    if (left.empty() || right.empty()) {
        std::printf("[depth] no frame in buffer for pair ts=%llu\n",
                    static_cast<unsigned long long>(pair.timestamp_avg));
        std::fflush(stdout);
        return;
    }
    if (!g_depth_ptr) {
        std::printf("[depth] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    std::lock_guard<std::mutex> save_lock(g_save_mutex);
    if (g_pairs_saved.load() >= SAVE_PAIRS) return;

    auto depth_m = g_depth_ptr->compute(left, right, best_cam0->box);

    if (depth_m)
        std::printf("[depth] Z = %.2f m  (object_id=%d)\n",
                    *depth_m, pair.object_id);
    else
        std::printf("[depth] could not compute (textureless / out of range / "
                    "bbox outside rectified image)\n");

    std::fflush(stdout);

    const int saved_idx = g_pairs_saved.fetch_add(1) + 1;
    save_diagnostic_pngs(saved_idx, left, right,
                         *best_cam0, best_cam1, depth_m);

    if (g_pairs_saved.load() >= SAVE_PAIRS) {
        std::printf("\n[debug] saved %d pair(s); shutting down.\n", SAVE_PAIRS);
        std::fflush(stdout);
        g_running = false;
        if (g_controller_ptr) g_controller_ptr->stop_capture();
    }
}

// -- main ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::fprintf(stderr,
            "Usage: %s <word>\n"
            "  e.g. %s cup\n",
            argv[0], argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Resolve word to COCO id
    const std::string target_word = argv[1];
    g_target_id = coco_id_for_word(target_word);
    if (g_target_id == 255) {
        std::fprintf(stderr, "[main] ERROR: \"%s\" is not a COCO class name\n",
                     target_word.c_str());
        return 1;
    }
    std::printf("[main] target: \"%s\" -> COCO id %d\n",
                target_word.c_str(), g_target_id);

    g_output_dir = make_session_dir();
    if (g_output_dir.empty()) return 1;
    std::printf("[main] output dir: %s\n", g_output_dir.c_str());

    gpio::setupGpio();

    // -- DetectionFilter
    DetectionFilter filter(on_filtered_pair);

    // -- Hailo
    std::printf("[main] creating Hailo inference (hef: %s)\n", DEFAULT_HEF_PATH);
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    hailo.register_callback(
        [&filter](uint8_t camera_id,
                  uint64_t timestamp_ns,
                  std::vector<std::vector<uint8_t>> raw_output)
        {
            const uint64_t n = ++g_hailo_callbacks;

            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_output);

            if (filter.is_armed()) {
                for (const auto& d : pkt.detections) {
                    if (d.object_id == g_target_id) {
                        std::printf("[hailo] HIT cb#%llu  cam=%d  conf=%.2f  "
                                    "box=[%.2f %.2f %.2f %.2f]\n",
                                    static_cast<unsigned long long>(n),
                                    camera_id, d.confidence,
                                    d.box.x_min, d.box.y_min,
                                    d.box.x_max, d.box.y_max);
                        std::fflush(stdout);
                    }
                }
            }

            filter.on_inference_packet(std::move(pkt));
        });

    if (!hailo.initialize()) {
        std::fprintf(stderr, "[main] FATAL: Hailo initialise failed\n");
        gpio::teardownGpio();
        return 1;
    }
    std::printf("[main] Hailo initialised\n");

    // -- Capture controller + stereo calibration
    CaptureController controller;
    g_controller_ptr = &controller;

    StereoDepthEstimator depth("../cam_calibration/stereo_calib_640_3.yaml");
    g_depth_ptr = &depth;
    std::printf("[main] stereo calibration loaded: baseline=%.3f m\n",
                depth.baseline_m());

    // -- Camera consumer thread
    std::thread cam_thread([&]() {
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;
            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            if (pkt->camera_id == 0) g_frame_buf_cam0.push(ts_ns, frame.clone());
            else                     g_frame_buf_cam1.push(ts_ns, frame.clone());

            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(), pkt->camera_id, ts_ns);

            const uint64_t n = ++g_cam_frames_consumed;
            if (n % 60 == 1) {
                std::printf("[cam] frame #%llu  cam=%d\n",
                            static_cast<unsigned long long>(n),
                            pkt->camera_id);
                std::fflush(stdout);
            }
        }
        std::printf("[cam] thread exiting -- %llu frames\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()));
    });

    // -- IR drain thread
    std::thread ir_thread([&]() {
        auto& ir_q = controller.get_ir_queue();
        while (auto pkt = ir_q.pop()) { ++g_ir_frames_consumed; (void)pkt; }
    });

    // -- Button: press arms filter, release resets
    button_driver::ButtonDriver btn;

    btn.registerPressCallback([&]() {
        std::printf("\n[button] PRESS -- starting capture, arming filter for \"%s\" (id=%d)\n",
                    target_word.c_str(), g_target_id);
        std::fflush(stdout);
        controller.start_capture();
        filter.on_intent(g_target_id);
    });

    btn.registerReleaseCallback([&]() {
        std::printf("\n[button] RELEASE -- stopping capture, resetting filter\n");
        std::fflush(stdout);
        controller.stop_capture();
        filter.reset();
    });

    std::printf("\n----------------------------------------------------\n");
    std::printf(" stereo_debug ready.\n");
    std::printf("  - Hold button to capture and filter for \"%s\".\n",
                target_word.c_str());
    std::printf("  - Program saves %d pair(s) then exits.\n", SAVE_PAIRS);
    std::printf("  - Output: %s\n", g_output_dir.c_str());
    std::printf("----------------------------------------------------\n\n");
    std::fflush(stdout);

    while (g_running)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // -- Shutdown
    std::printf("\n[main] shutting down\n");
    controller.stop_capture();
    controller.shutdown();
    cam_thread.join();
    ir_thread.join();
    hailo.stop();
    gpio::teardownGpio();

    std::printf("\n[main] cam_frames=%llu  ir_frames=%llu  hailo_cb=%llu  "
                "pairs=%llu  saved=%d\n",
                static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                static_cast<unsigned long long>(g_hailo_callbacks.load()),
                static_cast<unsigned long long>(g_pairs_emitted.load()),
                g_pairs_saved.load());
    return 0;
}
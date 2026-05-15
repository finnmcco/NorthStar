/*
    stereo_debug_main.cpp  —  one-shot stereo depth diagnostic

    Same upstream pipeline as integrated_main (button, capture, inference,
    mic intent, filter), but on the first FilteredInferencePair it:

      1. Computes stereo depth.
      2. Saves three diagnostic PNGs into stereo_debug/<session>/:
           pair_0001_unrectified.png   — cam0 | cam1 raw, with raw bbox drawn
           pair_0001_rectified.png     — cam0 | cam1 rectified, with rectified
                                          bbox drawn on each
           pair_0001_disparity.png     — colourised disparity, with rectified
                                          cam0 bbox drawn
      3. Prints the depth value and exits.

    Run as:
      ./stereo_debug_main <vosk_model_dir> <word1> [word2 ...]

    Output directory is created relative to CWD. If you want multiple captures
    per run, change SAVE_PAIRS below.
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "coco_lookup.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"
#include "pipeline.hpp"
#include "button-driver.h"
#include "gpio.h"
#include "inference/hailo8_inference.hpp"
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


// How many pairs to save before exiting. Set to a larger number if you want
// several captures per run; the program will stop the moment it has saved this
// many pair-bundles.
static constexpr int SAVE_PAIRS = 5;

// ─── Globals for signal handling ─────────────────────────────────────────────
std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;
std::atomic<bool> mic_armed{false};

// ─── Counters ────────────────────────────────────────────────────────────────
static std::atomic<uint64_t> g_cam_frames_consumed{0};
static std::atomic<uint64_t> g_ir_frames_consumed{0};
static std::atomic<uint64_t> g_hailo_callbacks{0};
static std::atomic<uint64_t> g_pairs_emitted{0};
static std::atomic<int>      g_pairs_saved{0};

// ─── Frame buffers + depth estimator ─────────────────────────────────────────
static FrameBuffer g_frame_buf_cam0;
static FrameBuffer g_frame_buf_cam1;
static StereoDepthEstimator* g_depth_ptr = nullptr;

// ─── Output directory ────────────────────────────────────────────────────────
static std::string g_output_dir;

// Guards the save path: only one pair at a time goes through the saving code,
// because save_diagnostic_pngs() is non-trivial and we don't want overlapping
// PNG writes for the same pair index.
static std::mutex g_save_mutex;

static void on_signal(int /*sig*/)
{
    g_running = false;
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build a timestamped session directory inside ./stereo_debug/ and return it.
// Creates the directory tree. Returns empty string on failure.
static std::string make_session_dir()
{
    using namespace std::chrono;
    const auto now    = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);

    std::ostringstream oss;
    oss << "stereo_debug/"
        << std::put_time(&tm, "%Y%m%d_%H%M%S");

    const std::string dir = oss.str();

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "[main] FATAL: could not create %s — %s\n",
                     dir.c_str(), ec.message().c_str());
        return {};
    }
    return dir;
}

// Draw a normalised bbox onto a 640x640 BGR image and label it.
static void draw_norm_bbox(cv::Mat& img,
                           const BoundingBox& box,
                           const cv::Scalar& colour,
                           const std::string& label)
{
    const int W = img.cols, H = img.rows;
    const int x0 = std::clamp(static_cast<int>(box.x_min * W), 0, W - 1);
    const int y0 = std::clamp(static_cast<int>(box.y_min * H), 0, H - 1);
    const int x1 = std::clamp(static_cast<int>(box.x_max * W), 0, W - 1);
    const int y1 = std::clamp(static_cast<int>(box.y_max * H), 0, H - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, colour, 2);
    if (!label.empty()) {
        cv::putText(img, label, {x0 + 4, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv::LINE_AA);
    }
}

// Draw a pixel-space bbox onto an image.
static void draw_pixel_bbox(cv::Mat& img,
                            int x0, int y0, int x1, int y1,
                            const cv::Scalar& colour,
                            const std::string& label)
{
    x0 = std::clamp(x0, 0, img.cols - 1);
    y0 = std::clamp(y0, 0, img.rows - 1);
    x1 = std::clamp(x1, 0, img.cols - 1);
    y1 = std::clamp(y1, 0, img.rows - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, colour, 2);
    if (!label.empty()) {
        cv::putText(img, label, {x0 + 4, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv::LINE_AA);
    }
}

// Concatenate two same-sized BGR images side-by-side and return the result.
static cv::Mat side_by_side(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat out;
    cv::hconcat(left, right, out);
    return out;
}

// Project a normalised bbox through rectification using the same K/D/R/P
// passed in. Returns the inclusive integer pixel rect, clamped to the image,
// or an empty cv::Rect if the projected box is entirely outside.
static cv::Rect rectify_bbox(const BoundingBox& box,
                             const cv::Mat& K, const cv::Mat& D,
                             const cv::Mat& R, const cv::Mat& P,
                             const cv::Size& image_size)
{
    const float W = static_cast<float>(image_size.width);
    const float H = static_cast<float>(image_size.height);

    // Use all four corners, not just two, because rectification is non-linear.
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
        min_x > image_size.width - 1 ||
        min_y > image_size.height - 1) {
        return {};
    }

    int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
    int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
    int x1 = std::min(image_size.width  - 1,
                      static_cast<int>(std::ceil(max_x)));
    int y1 = std::min(image_size.height - 1,
                      static_cast<int>(std::ceil(max_y)));
    if (x1 <= x0 || y1 <= y0) return {};
    return cv::Rect(cv::Point(x0, y0), cv::Point(x1 + 1, y1 + 1));
}

// Convert a float32 disparity map (in pixels) into a BGR colourised image
// suitable for saving. Invalid (negative) disparities map to black.
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

    // Mask out invalid pixels to black so they're visually obvious.
    coloured.setTo(cv::Scalar(0, 0, 0), ~valid_mask);

    return coloured;
}


// ─── The interesting bit: save all three diagnostic PNGs for one pair ────────
static void save_diagnostic_pngs(int pair_idx,
                                 const cv::Mat& left_raw,
                                 const cv::Mat& right_raw,
                                 const Detection& best_cam0,
                                 const Detection* best_cam1,   // may be null
                                 std::optional<float> depth_m)
{
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "%s/pair_%04d",
                  g_output_dir.c_str(), pair_idx);

    // ── 1. Unrectified side-by-side with raw bboxes ─────────────────────────
    {
        cv::Mat l = left_raw.clone();
        cv::Mat r = right_raw.clone();

        draw_norm_bbox(l, best_cam0.box,
                       cv::Scalar(0, 255, 0),
                       "cam0 raw");
        if (best_cam1) {
            draw_norm_bbox(r, best_cam1->box,
                           cv::Scalar(0, 255, 0),
                           "cam1 raw");
        } else {
            cv::putText(r, "(no cam1 detection)", {8, 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        const std::string path = std::string(prefix) + "_unrectified.png";
        if (cv::imwrite(path, side_by_side(l, r)))
            std::printf("[debug] saved %s\n", path.c_str());
        else
            std::printf("[debug] FAILED to save %s\n", path.c_str());
    }

    // ── 2. Rectified side-by-side with rectified bboxes ─────────────────────
    //
    //  Trigger compute() to populate last_left_rect / last_right_rect /
    //  last_disparity. We use the cam0 bbox (the one used for depth) so the
    //  estimator's state matches what we annotate.
    {
        const cv::Mat& lrect = g_depth_ptr->last_left_rect();
        const cv::Mat& rrect = g_depth_ptr->last_right_rect();
        if (lrect.empty() || rrect.empty()) {
            std::printf("[debug] WARNING: rectified frames not available "
                        "(did compute() fail?)\n");
        } else {
            cv::Mat l = lrect.clone();
            cv::Mat r = rrect.clone();

            cv::Rect rect_box_left = rectify_bbox(
                best_cam0.box,
                g_depth_ptr->K1(), g_depth_ptr->D1(),
                g_depth_ptr->R1(), g_depth_ptr->P1(),
                l.size());
            if (rect_box_left.area() > 0) {
                draw_pixel_bbox(l,
                                rect_box_left.x,
                                rect_box_left.y,
                                rect_box_left.x + rect_box_left.width  - 1,
                                rect_box_left.y + rect_box_left.height - 1,
                                cv::Scalar(0, 255, 0),
                                "cam0 rect");
            }

            if (best_cam1) {
                cv::Rect rect_box_right = rectify_bbox(
                    best_cam1->box,
                    g_depth_ptr->K2(), g_depth_ptr->D2(),
                    g_depth_ptr->R2(), g_depth_ptr->P2(),
                    r.size());
                if (rect_box_right.area() > 0) {
                    draw_pixel_bbox(r,
                                    rect_box_right.x,
                                    rect_box_right.y,
                                    rect_box_right.x + rect_box_right.width  - 1,
                                    rect_box_right.y + rect_box_right.height - 1,
                                    cv::Scalar(0, 255, 0),
                                    "cam1 rect");
                }
            }

            const std::string path = std::string(prefix) + "_rectified.png";
            if (cv::imwrite(path, side_by_side(l, r)))
                std::printf("[debug] saved %s\n", path.c_str());
            else
                std::printf("[debug] FAILED to save %s\n", path.c_str());
        }
    }

    // ── 3. Disparity map with rectified cam0 bbox overlay ──────────────────
    {
        const cv::Mat& disp = g_depth_ptr->last_disparity();
        if (disp.empty()) {
            std::printf("[debug] WARNING: disparity not available\n");
        } else {
            cv::Mat coloured = colourise_disparity(disp);

            cv::Rect rect_box_left = rectify_bbox(
                best_cam0.box,
                g_depth_ptr->K1(), g_depth_ptr->D1(),
                g_depth_ptr->R1(), g_depth_ptr->P1(),
                coloured.size());

            std::string label = "depth: ";
            if (depth_m) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.2f m", *depth_m);
                label += buf;
            } else {
                label += "n/a";
            }

            if (rect_box_left.area() > 0) {
                draw_pixel_bbox(coloured,
                                rect_box_left.x,
                                rect_box_left.y,
                                rect_box_left.x + rect_box_left.width  - 1,
                                rect_box_left.y + rect_box_left.height - 1,
                                cv::Scalar(255, 255, 255),
                                label);
            } else {
                cv::putText(coloured, label, {8, 24},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }

            const std::string path = std::string(prefix) + "_disparity.png";
            if (cv::imwrite(path, coloured))
                std::printf("[debug] saved %s\n", path.c_str());
            else
                std::printf("[debug] FAILED to save %s\n", path.c_str());
        }
    }
}


// ─── Filter callback ─────────────────────────────────────────────────────────
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

    // Highest-confidence cam0 and cam1 detections.
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
        std::printf("[depth] no frame in buffer for pair ts=%llu  "
                    "(cam0_empty=%d cam1_empty=%d)\n",
                    static_cast<unsigned long long>(pair.timestamp_avg),
                    left.empty(), right.empty());
        std::fflush(stdout);
        return;
    }
    if (!g_depth_ptr) {
        std::printf("[depth] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    // Serialise the save path. Two pairs landing back-to-back from the read
    // thread would otherwise share last_disparity_ etc.
    std::lock_guard<std::mutex> save_lock(g_save_mutex);

    if (g_pairs_saved.load() >= SAVE_PAIRS) {
        // Already done — ignore later pairs while we tear down.
        return;
    }

    auto depth_m = g_depth_ptr->compute(left, right, best_cam0->box);

    if (depth_m) {
        std::printf("[depth] Z = %.2f m  (object_id=%d)\n",
                    *depth_m, pair.object_id);
    } else {
        std::printf("[depth] could not compute (textureless / out of range / "
                    "bbox outside rectified image)\n");
    }
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


// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <vosk_model_dir> <word1> [word2 ...]\n"
            "  e.g. %s ../model_inf/vosk-model-small-en-us-0.15 cup person\n",
            argv[0], argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[main] stereo_debug — initialising\n");

    g_output_dir = make_session_dir();
    if (g_output_dir.empty()) return 1;
    std::printf("[main] output dir: %s\n", g_output_dir.c_str());

    std::printf("[main] setting up GPIO\n");
    gpio::setupGpio();

    // ── Filter ───────────────────────────────────────────────────────────────
    std::printf("[main] creating DetectionFilter\n");
    DetectionFilter filter(on_filtered_pair);

    // ── Mic pipeline config ──────────────────────────────────────────────────
    std::printf("[main] configuring mic pipeline (vosk model: %s)\n", argv[1]);
    Pipeline::Config mic_cfg;
    mic_cfg.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string word = argv[i];
        const uint8_t     id   = coco_id_for_word(word);
        if (id == 255) {
            std::fprintf(stderr,
                "[main] WARNING: \"%s\" is not a COCO class — will never match.\n",
                word.c_str());
        } else {
            std::printf("[main]   registered: \"%s\" -> COCO id %d\n",
                        word.c_str(), id);
        }
        mic_cfg.object_list.push_back(word);
        mic_cfg.object_ids.push_back(id);
    }

    mic_cfg.on_detection = [](const DetectionResult& r) {
        if (!mic_armed.load()) return;
        std::printf("[mic] heard \"%-12s\"  id=%-3d  (%s)\n",
                    r.word.c_str(), r.object_id,
                    r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    mic_cfg.on_intent = [&filter](uint8_t id) {
        if (!mic_armed.load()) {
            std::printf("[mic] intent ignored (button not held): id=%d\n", id);
            std::fflush(stdout);
            return;
        }
        std::printf("[mic] arming filter -> COCO id %d\n", id);
        std::fflush(stdout);
        filter.on_intent(id);
    };

    // ── Hailo inference ──────────────────────────────────────────────────────
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
                const uint8_t target = filter.target_object_id();
                int matches = 0;
                for (const auto& d : pkt.detections) {
                    if (d.object_id == target) {
                        ++matches;
                        std::printf("[hailo] HIT cb#%llu  cam=%d  target=%d  "
                                    "conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                                    static_cast<unsigned long long>(n),
                                    camera_id, target, d.confidence,
                                    d.box.x_min, d.box.y_min,
                                    d.box.x_max, d.box.y_max);
                    }
                }
                if (matches > 0) std::fflush(stdout);
            }

            filter.on_inference_packet(std::move(pkt));
        });

    std::printf("[main] initialising Hailo...\n");
    if (!hailo.initialize()) {
        std::fprintf(stderr, "[main] FATAL: Hailo initialise failed\n");
        gpio::teardownGpio();
        return 1;
    }
    std::printf("[main] Hailo initialised: input=%zu bytes, output streams=%zu\n",
                hailo.input_frame_size(), hailo.num_output_streams());

    // ── Capture controller ──────────────────────────────────────────────────
    std::printf("[main] creating CaptureController\n");
    CaptureController controller;
    g_controller_ptr = &controller;

    std::printf("[main] loading stereo calibration\n");
    StereoDepthEstimator depth("../cam_calibration/stereo_calib_640.yaml");
    g_depth_ptr = &depth;
    std::printf("[main] stereo calibration loaded: baseline=%.3f m\n",
                depth.baseline_m());

    // ── Camera consumer thread ───────────────────────────────────────────────
    std::printf("[main] spawning cam consumer thread\n");
    std::thread cam_thread([&]() {
        std::printf("[cam] thread started\n");
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            const uint64_t n = ++g_cam_frames_consumed;
            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;

            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            if (pkt->camera_id == 0) g_frame_buf_cam0.push(ts_ns, frame.clone());
            else                     g_frame_buf_cam1.push(ts_ns, frame.clone());

            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(),
                              pkt->camera_id,
                              ts_ns);

            if (n % 60 == 1) {
                std::printf("[cam] consumed frame #%llu  cam=%d  qsize=%zu\n",
                            static_cast<unsigned long long>(n),
                            pkt->camera_id,
                            controller.cam_queue_size());
                std::fflush(stdout);
            }
        }
        std::printf("[cam] thread exiting (queue stopped) — %llu frames total\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()));
    });

    // ── IR consumer thread ───────────────────────────────────────────────────
    // We don't need IR for this diagnostic, but the queue still has to be
    // drained or it'll back-pressure the producer.
    std::printf("[main] spawning IR consumer thread (drain only)\n");
    std::thread ir_thread([&]() {
        std::printf("[ir] thread started (drain only)\n");
        auto& ir_q = controller.get_ir_queue();
        while (auto pkt = ir_q.pop()) {
            ++g_ir_frames_consumed;
            (void)pkt;
        }
        std::printf("[ir] thread exiting — %llu frames drained\n",
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()));
    });

    // ── Mic pipeline (always-on) ─────────────────────────────────────────────
    std::printf("[main] starting mic pipeline\n");
    Pipeline mic_pipeline(std::move(mic_cfg));
    mic_pipeline.start();

    // ── Button ───────────────────────────────────────────────────────────────
    std::printf("[main] registering button callbacks\n");
    button_driver::ButtonDriver btn;

    btn.registerPressCallback([&]() {
        std::printf("\n[button] PRESS  — capture ON, mic active\n");
        std::fflush(stdout);
        mic_armed.store(true);
        controller.start_capture();
        std::printf("[cap] capture started\n");
        std::fflush(stdout);
    });

    btn.registerReleaseCallback([&]() {
        std::printf("\n[button] RELEASE  — capture OFF, mic ignored, filter reset\n");
        std::fflush(stdout);
        mic_armed.store(false);
        controller.stop_capture();
        filter.reset();
        std::printf("[cap] capture stopped\n");
        std::fflush(stdout);
    });

    std::printf("\n────────────────────────────────────────────────────\n");
    std::printf(" stereo_debug ready.\n");
    std::printf("  - Hold button to capture frames.\n");
    std::printf("  - Speak a registered object name while held.\n");
    std::printf("  - Program will save %d pair(s) then exit.\n", SAVE_PAIRS);
    std::printf("  - Output: %s\n", g_output_dir.c_str());
    std::printf("────────────────────────────────────────────────────\n\n");
    std::fflush(stdout);

    // ── Main idle loop ───────────────────────────────────────────────────────
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    std::printf("\n[main] shutdown initiated\n");

    controller.stop_capture();
    mic_pipeline.stop();
    controller.shutdown();
    cam_thread.join();
    ir_thread.join();
    hailo.stop();
    gpio::teardownGpio();

    std::printf("\n[main] FINAL: cam_frames=%llu  ir_frames=%llu  "
                "hailo_cb=%llu  pairs=%llu  saved=%d  mic_drops=%zu\n",
                static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                static_cast<unsigned long long>(g_hailo_callbacks.load()),
                static_cast<unsigned long long>(g_pairs_emitted.load()),
                g_pairs_saved.load(),
                mic_pipeline.queue_drops());

    std::printf("[main] goodbye\n");
    return 0;
}

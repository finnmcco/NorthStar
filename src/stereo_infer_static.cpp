/*
    stereo_infer_static.cpp  --  offline YOLO + stereo depth diagnostic

    Runs Hailo YOLO inference on a pair of static PNG images, filters
    detections to a user-specified COCO class, then produces:

      <out>/left_annotated.png          left frame with bounding box
      <out>/right_annotated.png         right frame with bounding box
      <out>/rectified.png               rectified pair side-by-side with bbox
      <out>/disparity.png               disparity map with bbox + depth label
      <out>/depth.png                   metric depth map with bbox + depth label

    Run as:
      ./stereo_infer_static <calib.yaml> <left.png> <right.png> <object> [out_dir]
      e.g.
      ./stereo_infer_static stereo_calib_640.yaml left.png right.png person

    The program is fully synchronous — no threads, no queues, no hardware.
    It feeds both frames into Hailo via write_frame(), waits for both
    inference callbacks to fire, then proceeds with the stereo pipeline.
*/

#include "coco_lookup.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"
#include "stereo_distance.hpp"
#include "hailo8_inference.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Pipeline state shared between the Hailo callback and main
// ---------------------------------------------------------------------------
struct InferenceState
{
    std::mutex              mtx;
    std::condition_variable cv;

    // Filled in by the Hailo callback, one entry per camera
    std::vector<Detection>  cam0_detections;
    std::vector<Detection>  cam1_detections;
    bool cam0_done{false};
    bool cam1_done{false};
};

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
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
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
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
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
}

static cv::Mat side_by_side(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat out;
    cv::hconcat(left, right, out);
    return out;
}

// ---------------------------------------------------------------------------
// Rectify a normalised bounding box into pixel coordinates in the rectified
// image, using the calibration maps for that camera.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Colourise helpers
// ---------------------------------------------------------------------------
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

static cv::Mat colourise_depth(const cv::Mat& disparity_px,
                               double fx_px, double base_m)
{
    cv::Mat depth_m(disparity_px.size(), CV_32F, 0.0f);
    const float fb = static_cast<float>(fx_px * base_m);
    for (int r = 0; r < disparity_px.rows; ++r) {
        const float* d_row = disparity_px.ptr<float>(r);
        float*       z_row = depth_m.ptr<float>(r);
        for (int c = 0; c < disparity_px.cols; ++c) {
            const float d = d_row[c];
            z_row[c] = (d > 0.0f) ? (fb / d) : 0.0f;
        }
    }
    cv::Mat valid_mask = depth_m > 0;
    double zmin = 0, zmax = 0;
    cv::minMaxLoc(depth_m, &zmin, &zmax, nullptr, nullptr, valid_mask);
    if (zmax <= zmin) zmax = zmin + 1.0;

    cv::Mat normed;
    depth_m.convertTo(normed, CV_8U,
                      255.0 / (zmax - zmin),
                      -255.0 * zmin / (zmax - zmin));
    cv::Mat coloured;
    cv::applyColorMap(normed, coloured, cv::COLORMAP_PLASMA);
    coloured.setTo(cv::Scalar(0, 0, 0), ~valid_mask);
    return coloured;
}

// ---------------------------------------------------------------------------
// Annotate a colourised map (disparity or depth) with the rectified bbox
// and a depth label.
// ---------------------------------------------------------------------------
static void annotate_map(cv::Mat& map,
                         const BoundingBox& box,
                         const StereoDepthEstimator& depth_est,
                         std::optional<float> depth_m)
{
    cv::Rect rb = rectify_bbox(box,
        depth_est.K1(), depth_est.D1(),
        depth_est.R1(), depth_est.P1(), map.size());

    std::string label = "depth: ";
    if (depth_m) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f m", *depth_m);
        label += buf;
    } else {
        label += "n/a";
    }

    if (rb.area() > 0)
        draw_pixel_bbox(map,
                        rb.x, rb.y,
                        rb.x + rb.width  - 1,
                        rb.y + rb.height - 1,
                        cv::Scalar(255, 255, 255), label);
    else
        cv::putText(map, label, {8, 24},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr,
            "Usage: %s <calib.yaml> <left.png> <right.png> <object> [out_dir]\n"
            "  e.g. %s stereo_calib_640.yaml left.png right.png person\n",
            argv[0], argv[0]);
        return 1;
    }

    const std::string calib_path  = argv[1];
    const std::string left_path   = argv[2];
    const std::string right_path  = argv[3];
    const std::string target_word = argv[4];
    const std::string output_dir  = (argc == 6) ? argv[5] : "stereo_infer_out";

    // -- Resolve COCO class name
    const uint8_t target_id = coco_id_for_word(target_word);
    if (target_id == 255) {
        std::fprintf(stderr, "[main] ERROR: \"%s\" is not a COCO class name\n",
                     target_word.c_str());
        return 1;
    }
    std::printf("[main] target: \"%s\" -> COCO id %d\n",
                target_word.c_str(), target_id);

    // -- Load images
    cv::Mat left_raw  = cv::imread(left_path,  cv::IMREAD_COLOR);
    cv::Mat right_raw = cv::imread(right_path, cv::IMREAD_COLOR);
    if (left_raw.empty()) {
        std::fprintf(stderr, "[main] ERROR: could not load: %s\n", left_path.c_str());
        return 1;
    }
    if (right_raw.empty()) {
        std::fprintf(stderr, "[main] ERROR: could not load: %s\n", right_path.c_str());
        return 1;
    }
    std::printf("[main] left  : %s (%dx%d)\n",
                left_path.c_str(),  left_raw.cols,  left_raw.rows);
    std::printf("[main] right : %s (%dx%d)\n",
                right_path.c_str(), right_raw.cols, right_raw.rows);

    // -- Output directory
    {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            std::fprintf(stderr, "[main] FATAL: could not create %s -- %s\n",
                         output_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }
    std::printf("[main] output dir: %s\n", output_dir.c_str());

    // -- Stereo calibration
    StereoDepthEstimator depth_est(calib_path);
    std::printf("[main] calibration loaded: baseline=%.3f m\n",
                depth_est.baseline_m());

    // -- Hailo inference
    std::printf("[main] initialising Hailo (hef: %s)\n", DEFAULT_HEF_PATH);
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    // Shared state for the two async callbacks
    InferenceState state;

    hailo.register_callback(
        [&state, target_id](uint8_t camera_id,
                            uint64_t /*timestamp_ns*/,
                            std::vector<std::vector<uint8_t>> raw_output)
        {
            std::vector<Detection> all = parse_detections(raw_output);

            // Keep only detections matching the target class
            std::vector<Detection> filtered;
            for (const auto& d : all)
                if (d.object_id == target_id)
                    filtered.push_back(d);

            std::printf("[hailo] cam%d: %zu raw detections, %zu match \"%s\"\n",
                        camera_id, all.size(), filtered.size(),
                        "target");  // target_word not in scope; label set outside
            std::fflush(stdout);

            std::lock_guard<std::mutex> lk(state.mtx);
            if (camera_id == 0) {
                state.cam0_detections = std::move(filtered);
                state.cam0_done = true;
            } else {
                state.cam1_detections = std::move(filtered);
                state.cam1_done = true;
            }
            state.cv.notify_all();
        });

    if (!hailo.initialize()) {
        std::fprintf(stderr, "[main] FATAL: Hailo initialise failed\n");
        return 1;
    }
    std::printf("[main] Hailo initialised\n");

    // -- Prepare and submit both frames
    // hailo_prepare converts BGR->RGB / normalises in-place, so work on copies.
    std::vector<uint8_t> left_data(left_raw.data,
                                   left_raw.data + left_raw.total() * left_raw.elemSize());
    std::vector<uint8_t> right_data(right_raw.data,
                                    right_raw.data + right_raw.total() * right_raw.elemSize());

    hailo_prepare(left_data);
    hailo_prepare(right_data);

    const uint64_t ts_left  = 1000ULL;   // synthetic timestamps; distinct so
    const uint64_t ts_right = 2000ULL;   // Hailo can distinguish the frames

    std::printf("[main] submitting frames to Hailo...\n");
    hailo.write_frame(left_data.data(),  0, ts_left);
    hailo.write_frame(right_data.data(), 1, ts_right);

    // -- Wait for both callbacks (with a 5-second timeout)
    {
        std::unique_lock<std::mutex> lk(state.mtx);
        const bool ok = state.cv.wait_for(lk, std::chrono::seconds(5), [&state] {
            return state.cam0_done && state.cam1_done;
        });
        if (!ok) {
            std::fprintf(stderr, "[main] FATAL: timed out waiting for Hailo callbacks\n");
            hailo.stop();
            return 1;
        }
    }
    std::printf("[main] both inference callbacks received\n");
    hailo.stop();

    // -- Find highest-confidence detection on each side
    const Detection* best_cam0 = nullptr;
    for (const auto& d : state.cam0_detections)
        if (!best_cam0 || d.confidence > best_cam0->confidence)
            best_cam0 = &d;

    const Detection* best_cam1 = nullptr;
    for (const auto& d : state.cam1_detections)
        if (!best_cam1 || d.confidence > best_cam1->confidence)
            best_cam1 = &d;

    if (!best_cam0) {
        std::printf("[main] WARNING: no \"%s\" detected in left frame\n",
                    target_word.c_str());
    } else {
        std::printf("[main] cam0 best: conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    best_cam0->confidence,
                    best_cam0->box.x_min, best_cam0->box.y_min,
                    best_cam0->box.x_max, best_cam0->box.y_max);
    }
    if (!best_cam1) {
        std::printf("[main] WARNING: no \"%s\" detected in right frame\n",
                    target_word.c_str());
    } else {
        std::printf("[main] cam1 best: conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    best_cam1->confidence,
                    best_cam1->box.x_min, best_cam1->box.y_min,
                    best_cam1->box.x_max, best_cam1->box.y_max);
    }

    // -- 1. Annotated left/right PNGs (unrectified)
    {
        cv::Mat l = left_raw.clone();
        cv::Mat r = right_raw.clone();

        if (best_cam0) {
            char label[64];
            std::snprintf(label, sizeof(label), "%s %.2f",
                          target_word.c_str(), best_cam0->confidence);
            draw_norm_bbox(l, best_cam0->box, cv::Scalar(0, 255, 0), label);
        } else {
            cv::putText(l, "no detection", {8, 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        if (best_cam1) {
            char label[64];
            std::snprintf(label, sizeof(label), "%s %.2f",
                          target_word.c_str(), best_cam1->confidence);
            draw_norm_bbox(r, best_cam1->box, cv::Scalar(0, 255, 0), label);
        } else {
            cv::putText(r, "no detection", {8, 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        const std::string pl = output_dir + "/left_annotated.png";
        const std::string pr = output_dir + "/right_annotated.png";
        cv::imwrite(pl, l)
            ? std::printf("[save] %s\n", pl.c_str())
            : std::fprintf(stderr, "[save] FAILED: %s\n", pl.c_str());
        cv::imwrite(pr, r)
            ? std::printf("[save] %s\n", pr.c_str())
            : std::fprintf(stderr, "[save] FAILED: %s\n", pr.c_str());
    }

    // -- Stereo depth (cam0 bbox drives the query)
    // If cam0 has no detection, fall back to full-frame box so the disparity
    // map is still generated — depth label will say "n/a".
    const BoundingBox query_box = best_cam0
        ? best_cam0->box
        : BoundingBox{0.0f, 0.0f, 1.0f, 1.0f};

    std::optional<float> depth_m = depth_est.compute(left_raw, right_raw, query_box);

    if (depth_m)
        std::printf("[depth] Z = %.3f m\n", *depth_m);
    else
        std::printf("[depth] could not compute "
                    "(textureless / out of range / bbox outside rectified image)\n");

    // -- 2. Rectified side-by-side with epipolar lines and bbox
    {
        const cv::Mat& lrect = depth_est.last_left_rect();
        const cv::Mat& rrect = depth_est.last_right_rect();

        if (lrect.empty() || rrect.empty()) {
            std::printf("[save] WARNING: rectified frames unavailable\n");
        } else {
            cv::Mat l = lrect.clone();
            cv::Mat r = rrect.clone();

            // Epipolar lines
            for (int y = 64; y < l.rows; y += 64) {
                cv::line(l, {0, y}, {l.cols - 1, y}, cv::Scalar(0, 128, 255), 1);
                cv::line(r, {0, y}, {r.cols - 1, y}, cv::Scalar(0, 128, 255), 1);
            }

            // Project cam0 bbox into left rectified image
            cv::Rect rb_l = rectify_bbox(query_box,
                depth_est.K1(), depth_est.D1(),
                depth_est.R1(), depth_est.P1(), l.size());
            if (rb_l.area() > 0)
                draw_pixel_bbox(l, rb_l.x, rb_l.y,
                                rb_l.x + rb_l.width  - 1,
                                rb_l.y + rb_l.height - 1,
                                cv::Scalar(0, 255, 0),
                                best_cam0 ? target_word : "");

            // Project cam1 bbox into right rectified image (if available)
            if (best_cam1) {
                cv::Rect rb_r = rectify_bbox(best_cam1->box,
                    depth_est.K2(), depth_est.D2(),
                    depth_est.R2(), depth_est.P2(), r.size());
                if (rb_r.area() > 0)
                    draw_pixel_bbox(r, rb_r.x, rb_r.y,
                                    rb_r.x + rb_r.width  - 1,
                                    rb_r.y + rb_r.height - 1,
                                    cv::Scalar(0, 255, 0), target_word);
            } else {
                cv::putText(r, "no detection", {8, 24},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }

            const std::string path = output_dir + "/rectified.png";
            cv::imwrite(path, side_by_side(l, r))
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    // -- 3. Disparity map
    {
        const cv::Mat& disp = depth_est.last_disparity();
        if (disp.empty()) {
            std::printf("[save] WARNING: disparity map unavailable\n");
        } else {
            cv::Mat coloured = colourise_disparity(disp);
            annotate_map(coloured, query_box, depth_est, depth_m);

            const std::string path = output_dir + "/disparity.png";
            cv::imwrite(path, coloured)
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    // -- 4. Metric depth map
    {
        const cv::Mat& disp = depth_est.last_disparity();
        if (disp.empty()) {
            std::printf("[save] WARNING: disparity unavailable; skipping depth map\n");
        } else {
            const double fx_px = depth_est.P1().at<double>(0, 0);
            cv::Mat coloured = colourise_depth(disp, fx_px, depth_est.baseline_m());
            annotate_map(coloured, query_box, depth_est, depth_m);

            const std::string path = output_dir + "/depth.png";
            cv::imwrite(path, coloured)
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    std::printf("\n[main] done. outputs in: %s/\n", output_dir.c_str());
    return depth_m ? 0 : 2;
}
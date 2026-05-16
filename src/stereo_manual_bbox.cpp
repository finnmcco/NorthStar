/*
    stereo_manual_bbox.cpp  --  offline stereo diagnostic with manual bboxes

    No inference. Pixel-coordinate bounding boxes are passed on the command
    line for both cameras. The disparity map gets a metric depth colourbar
    on its right-hand side.

    Run as:
      ./stereo_manual_bbox <calib.yaml> <left.png> <right.png> <out_dir> \
          <L_x1> <L_x2> <L_y1> <L_y2> \
          <R_x1> <R_x2> <R_y1> <R_y2>

    Example:
      ./stereo_manual_bbox calib.yaml l.png r.png out/ \
          300 320 400 420   285 305 400 420

    Outputs (in <out_dir>):
      unrectified.png     left|right with epipolar lines + manual bboxes
      rectified.png       left|right rectified, with epipolar lines + bboxes
      disparity.png       disparity map with bbox, depth label, depth key
*/

#include "stereo_distance.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
struct PixelBox { int x1, x2, y1, y2; };

static BoundingBox to_norm(const PixelBox& pb, int W, int H)
{
    BoundingBox b;
    b.x_min = static_cast<float>(pb.x1) / W;
    b.x_max = static_cast<float>(pb.x2) / W;
    b.y_min = static_cast<float>(pb.y1) / H;
    b.y_max = static_cast<float>(pb.y2) / H;
    return b;
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

// Draws a bright green bounding box with the label rendered OUTSIDE the box
// (preferred placement: just above; falls back to below if there's no room).
static void draw_bbox_with_outside_label(cv::Mat& img,
                                         int x0, int y0, int x1, int y1,
                                         const std::string& label)
{
    const cv::Scalar GREEN(0, 255, 0);   // BGR

    x0 = std::clamp(x0, 0, img.cols - 1);
    y0 = std::clamp(y0, 0, img.rows - 1);
    x1 = std::clamp(x1, 0, img.cols - 1);
    y1 = std::clamp(y1, 0, img.rows - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, GREEN, 2);

    if (label.empty()) return;

    constexpr double FONT_SCALE = 0.6;
    constexpr int    THICKNESS  = 2;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                  FONT_SCALE, THICKNESS, &baseline);

    // Prefer placement just above the box; fall back below if not enough room.
    cv::Point org;
    if (y0 - 6 - ts.height >= 0) {
        org = { x0, y0 - 6 };
    } else if (y1 + 6 + ts.height < img.rows) {
        org = { x0, y1 + 6 + ts.height };
    } else {
        org = { x0, std::min(img.rows - 4, y0 + ts.height + 4) };
    }
    // Clamp x so the label isn't clipped on the right edge.
    org.x = std::clamp(org.x, 2, std::max(2, img.cols - ts.width - 2));

    cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, GREEN, THICKNESS, cv::LINE_AA);
}

static cv::Mat side_by_side(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out; cv::hconcat(a, b, out); return out;
}

static void draw_epipolar_lines(cv::Mat& img, int spacing = 64)
{
    for (int y = spacing; y < img.rows; y += spacing)
        cv::line(img, {0, y}, {img.cols - 1, y},
                 cv::Scalar(255, 0, 0), 1);
}

// Rectify a normalised bbox into pixel coords in the rectified image.
static cv::Rect rectify_bbox(const BoundingBox& box,
                             const cv::Mat& K, const cv::Mat& D,
                             const cv::Mat& R, const cv::Mat& P,
                             const cv::Size& sz)
{
    const float W = static_cast<float>(sz.width);
    const float H = static_cast<float>(sz.height);
    std::vector<cv::Point2f> raw = {
        {box.x_min*W, box.y_min*H}, {box.x_max*W, box.y_min*H},
        {box.x_max*W, box.y_max*H}, {box.x_min*W, box.y_max*H},
    };
    std::vector<cv::Point2f> rect;
    cv::undistortPoints(raw, rect, K, D, R, P);
    float mn_x=1e9f, mn_y=1e9f, mx_x=-1e9f, mx_y=-1e9f;
    for (auto& p : rect) {
        mn_x = std::min(mn_x, p.x); mn_y = std::min(mn_y, p.y);
        mx_x = std::max(mx_x, p.x); mx_y = std::max(mx_y, p.y);
    }
    if (mx_x < 0 || mx_y < 0 ||
        mn_x > sz.width-1 || mn_y > sz.height-1) return {};
    int rx0 = std::max(0, (int)std::floor(mn_x));
    int ry0 = std::max(0, (int)std::floor(mn_y));
    int rx1 = std::min(sz.width-1, (int)std::ceil(mx_x));
    int ry1 = std::min(sz.height-1, (int)std::ceil(mx_y));
    if (rx1 <= rx0 || ry1 <= ry0) return {};
    return cv::Rect(cv::Point(rx0, ry0), cv::Point(rx1+1, ry1+1));
}

// ---------------------------------------------------------------------------
// Render a metric DEPTH map.
//   - Close objects  -> warm/bright colours (red/yellow)
//   - Far objects    -> cool/dark colours   (blue)
//   - Invalid pixels -> black
//
// Depths above max_depth_m are treated as invalid. This avoids a handful of
// spurious far-distance pixels dominating the colour scale.
// ---------------------------------------------------------------------------
static cv::Mat render_depth_map(const cv::Mat& disparity_px,
                                double fx_px, double base_m,
                                double max_depth_m,
                                double& z_min_out, double& z_max_out)
{
    cv::Mat depth_m(disparity_px.size(), CV_32F, 0.0f);
    const float fb = static_cast<float>(fx_px * base_m);
    const float zmax_f = static_cast<float>(max_depth_m);
    for (int r = 0; r < disparity_px.rows; ++r) {
        const float* d_row = disparity_px.ptr<float>(r);
        float*       z_row = depth_m.ptr<float>(r);
        for (int c = 0; c < disparity_px.cols; ++c) {
            const float d = d_row[c];
            if (d <= 0.0f) { z_row[c] = 0.0f; continue; }
            const float z = fb / d;
            z_row[c] = (z > zmax_f) ? 0.0f : z;
        }
    }

    cv::Mat valid = depth_m > 0;
    double zmin = 0, zmax = 0;
    cv::minMaxLoc(depth_m, &zmin, &zmax, nullptr, nullptr, valid);
    if (zmax <= zmin) zmax = zmin + 1.0;

    // Map depth -> 8-bit index, INVERTED so close (zmin) -> 255 (warm),
    // far (zmax) -> 0 (cool). TURBO then renders close as red and far as blue.
    cv::Mat n8(depth_m.size(), CV_8U, cv::Scalar(0));
    const double scale = 255.0 / (zmax - zmin);
    for (int r = 0; r < depth_m.rows; ++r) {
        const float* z_row = depth_m.ptr<float>(r);
        uchar*       o_row = n8.ptr<uchar>(r);
        const uchar* v_row = valid.ptr<uchar>(r);
        for (int c = 0; c < depth_m.cols; ++c) {
            if (!v_row[c]) { o_row[c] = 0; continue; }
            const double t = (z_row[c] - zmin) * scale;   // 0..255 (low=close)
            const double inv = 255.0 - t;                  // 0..255 (high=close)
            o_row[c] = static_cast<uchar>(
                std::clamp(static_cast<int>(std::round(inv)), 0, 255));
        }
    }

    cv::Mat coloured;
    cv::applyColorMap(n8, coloured, cv::COLORMAP_BONE);
    coloured.setTo(cv::Scalar(0, 0, 0), ~valid);

    z_min_out = zmin;
    z_max_out = zmax;
    return coloured;
}

// ---------------------------------------------------------------------------
// Build a vertical depth colour key matching the depth map's scale.
// Top of bar  = z_min (close, warm/red)
// Bottom of bar = z_max (far, cool/blue)
// ---------------------------------------------------------------------------
static cv::Mat build_depth_key(int map_h,
                               double z_min, double z_max,
                               int colormap = cv::COLORMAP_BONE)
{
    constexpr int BAR_WIDTH    = 32;
    constexpr int LABEL_WIDTH  = 90;
    constexpr int MARGIN_LEFT  = 8;
    constexpr int MARGIN_RIGHT = 8;
    const int total_w = MARGIN_LEFT + BAR_WIDTH + 6 + LABEL_WIDTH + MARGIN_RIGHT;

    cv::Mat key(map_h, total_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Gradient bar: top row = 255 (warm/close), bottom row = 0 (cool/far),
    // matching the inverted mapping used in render_depth_map().
    cv::Mat gradient(map_h, BAR_WIDTH, CV_8U);
    for (int r = 0; r < map_h; ++r) {
        const uchar v = static_cast<uchar>(
            std::round(255.0 * (1.0 - static_cast<double>(r) /
                                std::max(1, map_h - 1))));
        gradient.row(r).setTo(v);
    }
    cv::Mat gradient_col;
    cv::applyColorMap(gradient, gradient_col, colormap);
    gradient_col.copyTo(key(cv::Rect(MARGIN_LEFT, 0, BAR_WIDTH, map_h)));

    // Bar outline
    cv::rectangle(key,
                  {MARGIN_LEFT - 1, 0},
                  {MARGIN_LEFT + BAR_WIDTH, map_h - 1},
                  cv::Scalar(200, 200, 200), 1);

    // Tick labels — top label is z_min (close), bottom is z_max (far)
    constexpr int N_TICKS = 5;
    const int label_x = MARGIN_LEFT + BAR_WIDTH + 6;
    for (int i = 0; i < N_TICKS; ++i) {
        const double frac = static_cast<double>(i) / (N_TICKS - 1);
        const int y = static_cast<int>(std::round(frac * (map_h - 1)));
        const double z = z_min + frac * (z_max - z_min);

        cv::line(key,
                 {MARGIN_LEFT + BAR_WIDTH, y},
                 {MARGIN_LEFT + BAR_WIDTH + 4, y},
                 cv::Scalar(220, 220, 220), 1);

        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f m", z);
        const int text_y = std::clamp(y + 5, 12, map_h - 4);
        cv::putText(key, buf, {label_x, text_y},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }

    cv::putText(key, "depth", {MARGIN_LEFT - 2, 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    return key;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static int usage_err(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s <calib.yaml> <left.png> <right.png> <out_dir> \\\n"
        "         <L_x1> <L_x2> <L_y1> <L_y2> \\\n"
        "         <R_x1> <R_x2> <R_y1> <R_y2> \\\n"
        "         [max_depth_m]   (default 7.0) \\\n"
        "         [run_tag]       (e.g. 60 -> _60 suffix on filenames)\n",
        prog);
    return 1;
}

int main(int argc, char* argv[])
{
    if (argc < 13 || argc > 15) return usage_err(argv[0]);

    const std::string calib_path = argv[1];
    const std::string left_path  = argv[2];
    const std::string right_path = argv[3];
    const std::string out_dir    = argv[4];

    PixelBox box_L { std::atoi(argv[5]),  std::atoi(argv[6]),
                     std::atoi(argv[7]),  std::atoi(argv[8])  };
    PixelBox box_R { std::atoi(argv[9]),  std::atoi(argv[10]),
                     std::atoi(argv[11]), std::atoi(argv[12]) };

    const double max_depth_m = (argc >= 14) ? std::atof(argv[13]) : 7.0;
    if (max_depth_m <= 0.0) {
        std::fprintf(stderr, "[main] ERROR: max_depth_m must be > 0\n");
        return 1;
    }
    std::printf("[main] max depth for colour scale: %.2f m\n", max_depth_m);

    const std::string run_tag = (argc == 15) ? std::string("_") + argv[14] : "";
    if (!run_tag.empty())
        std::printf("[main] filename suffix: %s\n", run_tag.c_str());

    // -- Load images
    cv::Mat left_raw  = cv::imread(left_path,  cv::IMREAD_COLOR);
    cv::Mat right_raw = cv::imread(right_path, cv::IMREAD_COLOR);
    if (left_raw.empty()) {
        std::fprintf(stderr, "[main] ERROR: cannot load %s\n", left_path.c_str());
        return 1;
    }
    if (right_raw.empty()) {
        std::fprintf(stderr, "[main] ERROR: cannot load %s\n", right_path.c_str());
        return 1;
    }
    std::printf("[main] left  : %s (%dx%d)\n", left_path.c_str(),
                left_raw.cols, left_raw.rows);
    std::printf("[main] right : %s (%dx%d)\n", right_path.c_str(),
                right_raw.cols, right_raw.rows);

    // Swap R/B channels: source PNGs were stored RGB-ordered, so OpenCV
    // (which expects BGR) renders red as blue and vice versa. Do this BEFORE
    // any drawing so the overlay colours below are interpreted correctly.
    cv::cvtColor(left_raw,  left_raw,  cv::COLOR_BGR2RGB);
    cv::cvtColor(right_raw, right_raw, cv::COLOR_BGR2RGB);
    std::printf("[main] L bbox px: x[%d..%d] y[%d..%d]\n",
                box_L.x1, box_L.x2, box_L.y1, box_L.y2);
    std::printf("[main] R bbox px: x[%d..%d] y[%d..%d]\n",
                box_R.x1, box_R.x2, box_R.y1, box_R.y2);

    // -- Output dir
    {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            std::fprintf(stderr, "[main] FATAL: cannot create %s -- %s\n",
                         out_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    // -- Calibration
    StereoDepthEstimator depth_est(calib_path);
    const double fx_px  = depth_est.P1().at<double>(0, 0);
    const double base_m = depth_est.baseline_m();
    std::printf("[main] calibration: baseline=%.4f m, rectified fx=%.2f px\n",
                base_m, fx_px);

    // -- Normalised boxes (the stereo estimator and rectify_bbox both take these)
    const BoundingBox box_L_norm = to_norm(box_L, left_raw.cols,  left_raw.rows);
    const BoundingBox box_R_norm = to_norm(box_R, right_raw.cols, right_raw.rows);

    // -- Run depth estimation using the LEFT bbox as the query region.
    // The right bbox is used only for visualisation here (the estimator
    // computes a disparity map internally and queries it at the rectified
    // left bbox).
    std::optional<float> depth_m =
        depth_est.compute(left_raw, right_raw, box_L_norm);
    if (depth_m)
        std::printf("[depth] Z = %.3f m\n", *depth_m);
    else
        std::printf("[depth] could not compute "
                    "(textureless / out of range / bbox outside rectified image)\n");

    // -----------------------------------------------------------------------
    // 1. Unrectified side-by-side
    // -----------------------------------------------------------------------
    {
        const cv::Scalar GREEN(0, 255, 0);
        cv::Mat l = left_raw.clone();
        cv::Mat r = right_raw.clone();
        draw_pixel_bbox(l, box_L.x1, box_L.y1, box_L.x2, box_L.y2,
                        GREEN, "left");
        draw_pixel_bbox(r, box_R.x1, box_R.y1, box_R.x2, box_R.y2,
                        GREEN, "right");
        draw_epipolar_lines(l);
        draw_epipolar_lines(r);
        const std::string path = out_dir + "/unrectified" + run_tag + ".png";
        cv::imwrite(path, side_by_side(l, r))
            ? std::printf("[save] %s\n", path.c_str())
            : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
    }

    // -----------------------------------------------------------------------
    // 2. Rectified side-by-side
    // -----------------------------------------------------------------------
    {
        const cv::Mat& lrect = depth_est.last_left_rect();
        const cv::Mat& rrect = depth_est.last_right_rect();
        if (lrect.empty() || rrect.empty()) {
            std::printf("[save] WARNING: rectified frames unavailable\n");
        } else {
            cv::Mat l = lrect.clone();
            cv::Mat r = rrect.clone();

            const cv::Scalar GREEN(0, 255, 0);

            cv::Rect rb_l = rectify_bbox(box_L_norm,
                depth_est.K1(), depth_est.D1(),
                depth_est.R1(), depth_est.P1(), l.size());
            if (rb_l.area() > 0)
                draw_pixel_bbox(l, rb_l.x, rb_l.y,
                                rb_l.x + rb_l.width  - 1,
                                rb_l.y + rb_l.height - 1,
                                GREEN, "left rect");

            cv::Rect rb_r = rectify_bbox(box_R_norm,
                depth_est.K2(), depth_est.D2(),
                depth_est.R2(), depth_est.P2(), r.size());
            if (rb_r.area() > 0)
                draw_pixel_bbox(r, rb_r.x, rb_r.y,
                                rb_r.x + rb_r.width  - 1,
                                rb_r.y + rb_r.height - 1,
                                GREEN, "right rect");

            draw_epipolar_lines(l);
            draw_epipolar_lines(r);

            const std::string path = out_dir + "/rectified" + run_tag + ".png";
            cv::imwrite(path, side_by_side(l, r))
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    // -----------------------------------------------------------------------
    // 3. Depth map (close = warm/red, far = cool/blue) + bbox + key
    // -----------------------------------------------------------------------
    {
        const cv::Mat& disp = depth_est.last_disparity();
        if (disp.empty()) {
            std::printf("[save] WARNING: disparity unavailable\n");
        } else {
            double z_min = 0, z_max = 0;
            cv::Mat coloured = render_depth_map(disp, fx_px, base_m,
                                                max_depth_m,
                                                z_min, z_max);

            // Overlay rectified bbox + depth label OUTSIDE the box, in red
            cv::Rect rb_l = rectify_bbox(box_L_norm,
                depth_est.K1(), depth_est.D1(),
                depth_est.R1(), depth_est.P1(), coloured.size());

            char label[64];
            if (depth_m) std::snprintf(label, sizeof(label), "Z = %.2f m", *depth_m);
            else         std::snprintf(label, sizeof(label), "Z = n/a");

            if (rb_l.area() > 0) {
                draw_bbox_with_outside_label(coloured,
                    rb_l.x, rb_l.y,
                    rb_l.x + rb_l.width  - 1,
                    rb_l.y + rb_l.height - 1,
                    label);
            } else {
                cv::putText(coloured, label, {8, 24},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            std::printf("[depth] colour scale: %.2f m (close/warm) .. %.2f m (far/cool)\n",
                        z_min, z_max);

            cv::Mat key = build_depth_key(coloured.rows, z_min, z_max);
            cv::Mat composite;
            cv::hconcat(coloured, key, composite);

            const std::string path = out_dir + "/depth" + run_tag + ".png";
            cv::imwrite(path, composite)
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    std::printf("\n[main] done. outputs in: %s/\n", out_dir.c_str());
    return depth_m ? 0 : 2;
}
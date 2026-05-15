/*
    stereo_debug_static.cpp  --  offline stereo depth diagnostic

    Takes a left and right PNG image as arguments and produces the same
    three diagnostic outputs as stereo_debug_main, without any hardware
    dependencies (no camera, button, Hailo, or IR).

    Run as:
      ./stereo_debug_static <calib.yaml> <left.png> <right.png> [output_dir]

    Outputs into output_dir (default: stereo_debug_static/) :
      pair_0001_unrectified.png
      pair_0001_rectified.png
      pair_0001_disparity.png

    No object detection is performed. The entire image is used as the
    bounding box for the disparity / depth query, so the reported depth
    is the median depth over the full scene. If you want a sub-region,
    edit QUERY_BOX below.
*/

#include "stereo_distance.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Query bounding box in normalised [0,1] coordinates.
// By default the whole image is used. Edit to restrict to a sub-region.
// ---------------------------------------------------------------------------
static constexpr BoundingBox QUERY_BOX { 0.0f, 0.0f, 1.0f, 1.0f };

// ---------------------------------------------------------------------------
// Helpers copied verbatim from stereo_debug_main.cpp
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

// Converts a per-pixel disparity map (float, pixels) to a metric depth map
// (float, metres) using Z = f*B/d, then colourise it.
// fx_px  : rectified focal length (P1[0,0])
// base_m : stereo baseline in metres
static cv::Mat colourise_depth(const cv::Mat& disparity_px,
                               double fx_px, double base_m)
{
    cv::Mat depth_m(disparity_px.size(), CV_32F, 0.0f);
    const float fb = static_cast<float>(fx_px * base_m);

    for (int r = 0; r < disparity_px.rows; ++r) {
        const float* d_row  = disparity_px.ptr<float>(r);
        float*       z_row  = depth_m.ptr<float>(r);
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
            "Usage: %s <calib.yaml> <left.png> <right.png> [output_dir]\n",
            argv[0]);
        return 1;
    }

    const std::string calib_path  = argv[1];
    const std::string left_path   = argv[2];
    const std::string right_path  = argv[3];
    const std::string output_dir  = (argc == 5) ? argv[4] : "stereo_debug_static";

    // -- Load images
    cv::Mat left  = cv::imread(left_path,  cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
    if (left.empty()) {
        std::fprintf(stderr, "[main] ERROR: could not load left image: %s\n",
                     left_path.c_str());
        return 1;
    }
    if (right.empty()) {
        std::fprintf(stderr, "[main] ERROR: could not load right image: %s\n",
                     right_path.c_str());
        return 1;
    }
    std::printf("[main] left  : %s (%dx%d)\n",
                left_path.c_str(),  left.cols,  left.rows);
    std::printf("[main] right : %s (%dx%d)\n",
                right_path.c_str(), right.cols, right.rows);

    // -- Load calibration and initialise depth estimator
    StereoDepthEstimator depth(calib_path);
    std::printf("[main] stereo calibration loaded: baseline=%.3f m\n",
                depth.baseline_m());

    // -- Create output directory
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

    // -- Run depth estimation using the full image as the query box
    const std::string prefix = output_dir + "/pair_0001";

    std::optional<float> depth_m = depth.compute(left, right, QUERY_BOX);

    if (depth_m)
        std::printf("[depth] Z = %.3f m\n", *depth_m);
    else
        std::printf("[depth] could not compute depth "
                    "(textureless / out of range / bbox outside rectified image)\n");

    // -- 1. Unrectified side-by-side
    {
        cv::Mat l = left.clone();
        cv::Mat r = right.clone();
        draw_norm_bbox(l, QUERY_BOX, cv::Scalar(0, 255, 0), "left raw");
        draw_norm_bbox(r, QUERY_BOX, cv::Scalar(0, 255, 0), "right raw");

        // Epipolar lines on raw images — will look skewed if cameras aren't
        // co-planar; contrast with the rectified version to judge alignment.
        for (int y = 64; y < l.rows; y += 64) {
            cv::line(l, {0, y}, {l.cols - 1, y}, cv::Scalar(0, 128, 255), 1);
            cv::line(r, {0, y}, {r.cols - 1, y}, cv::Scalar(0, 128, 255), 1);
        }

        const std::string path = prefix + "_unrectified.png";
        cv::imwrite(path, side_by_side(l, r))
            ? std::printf("[save] %s\n", path.c_str())
            : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
    }

    // -- 2. Rectified side-by-side
    {
        const cv::Mat& lrect = depth.last_left_rect();
        const cv::Mat& rrect = depth.last_right_rect();

        if (lrect.empty() || rrect.empty()) {
            std::printf("[save] WARNING: rectified frames not available\n");
        } else {
            cv::Mat l = lrect.clone();
            cv::Mat r = rrect.clone();

            cv::Rect rb_l = rectify_bbox(QUERY_BOX,
                depth.K1(), depth.D1(), depth.R1(), depth.P1(), l.size());
            if (rb_l.area() > 0)
                draw_pixel_bbox(l, rb_l.x, rb_l.y,
                                rb_l.x + rb_l.width - 1,
                                rb_l.y + rb_l.height - 1,
                                cv::Scalar(0, 255, 0), "left rect");

            cv::Rect rb_r = rectify_bbox(QUERY_BOX,
                depth.K2(), depth.D2(), depth.R2(), depth.P2(), r.size());
            if (rb_r.area() > 0)
                draw_pixel_bbox(r, rb_r.x, rb_r.y,
                                rb_r.x + rb_r.width - 1,
                                rb_r.y + rb_r.height - 1,
                                cv::Scalar(0, 255, 0), "right rect");

            // Overlay horizontal epipolar lines every 64 px to verify alignment
            for (int y = 64; y < l.rows; y += 64) {
                cv::line(l, {0, y}, {l.cols - 1, y}, cv::Scalar(0, 128, 255), 1);
                cv::line(r, {0, y}, {r.cols - 1, y}, cv::Scalar(0, 128, 255), 1);
            }

            const std::string path = prefix + "_rectified.png";
            cv::imwrite(path, side_by_side(l, r))
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    // -- 3. Disparity map
    {
        const cv::Mat& disp = depth.last_disparity();
        if (disp.empty()) {
            std::printf("[save] WARNING: disparity map not available\n");
        } else {
            cv::Mat coloured = colourise_disparity(disp);

            cv::Rect rb_l = rectify_bbox(QUERY_BOX,
                depth.K1(), depth.D1(), depth.R1(), depth.P1(), coloured.size());

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

            const std::string path = prefix + "_disparity.png";
            cv::imwrite(path, coloured)
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    // -- 4. Depth map (metric, Z = f*B/d)
    {
        const cv::Mat& disp = depth.last_disparity();
        if (disp.empty()) {
            std::printf("[save] WARNING: disparity not available; skipping depth map\n");
        } else {
            // P1[0,0] is the rectified focal length; read it from the estimator.
            const double fx_px = depth.P1().at<double>(0, 0);
            cv::Mat coloured = colourise_depth(disp, fx_px, depth.baseline_m());

            cv::Rect rb_l = rectify_bbox(QUERY_BOX,
                depth.K1(), depth.D1(), depth.R1(), depth.P1(), coloured.size());

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

            const std::string path = prefix + "_depth.png";
            cv::imwrite(path, coloured)
                ? std::printf("[save] %s\n", path.c_str())
                : std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    }

    return depth_m ? 0 : 2;   // exit 2 = depth could not be resolved
}
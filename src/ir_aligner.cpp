/*
    ir_aligner.cpp  —  Runtime IR spatial alignment (homography-based)

    See ir_aligner.hpp for design notes.
*/

#include "ir_aligner.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

IRAligner::IRAligner(const std::string& yaml_path)
{
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("IRAligner: could not open YAML: " + yaml_path);
    }

    fs["H"]         >> H_;
    fs["ir_width"]  >> ir_w_;
    fs["ir_height"] >> ir_h_;
    fs.release();

    if (H_.empty() || H_.rows != 3 || H_.cols != 3) {
        throw std::runtime_error(
            "IRAligner: 'H' must be a 3x3 matrix in " + yaml_path);
    }
    if (ir_w_ <= 0 || ir_h_ <= 0) {
        throw std::runtime_error(
            "IRAligner: invalid ir_width/ir_height in " + yaml_path);
    }

    // Ensure double-precision arithmetic throughout.
    H_.convertTo(H_, CV_64F);
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection
// ─────────────────────────────────────────────────────────────────────────────

IRAligner::PixelRect IRAligner::project_bbox(const BoundingBox& bbox_cam0) const
{
    PixelRect out{0, 0, 0, 0, false};

    if (H_.empty() || ir_w_ <= 0 || ir_h_ <= 0) {
        return out;
    }

    // Clamp normalised camera bbox to [0, 1].
    const double x_min = std::clamp(static_cast<double>(bbox_cam0.x_min), 0.0, 1.0);
    const double y_min = std::clamp(static_cast<double>(bbox_cam0.y_min), 0.0, 1.0);
    const double x_max = std::clamp(static_cast<double>(bbox_cam0.x_max), 0.0, 1.0);
    const double y_max = std::clamp(static_cast<double>(bbox_cam0.y_max), 0.0, 1.0);

    if (x_max <= x_min || y_max <= y_min) {
        return out;   // degenerate bbox
    }

    // Cache H elements for readability.
    const double h00 = H_.at<double>(0, 0), h01 = H_.at<double>(0, 1), h02 = H_.at<double>(0, 2);
    const double h10 = H_.at<double>(1, 0), h11 = H_.at<double>(1, 1), h12 = H_.at<double>(1, 2);
    const double h20 = H_.at<double>(2, 0), h21 = H_.at<double>(2, 1), h22 = H_.at<double>(2, 2);

    // Apply the homography: [x'; y'; w'] = H * [x; y; 1]
    // IR coordinate = [x'/w' , y'/w']
    //
    // We project all four corners because the mapping is projective —
    // a rectangle in camera space maps to a (generally non-rectangular)
    // quadrilateral in IR space. We then take the AABB of that quadrilateral.
    auto project = [&](double x, double y) -> cv::Point2d {
        const double wp = h20 * x + h21 * y + h22;
        if (std::abs(wp) < 1e-10) {
            // Point at infinity — should not occur for a valid calibration.
            return {0.0, 0.0};
        }
        return {
            (h00 * x + h01 * y + h02) / wp,
            (h10 * x + h11 * y + h12) / wp
        };
    };

    const std::array<cv::Point2d, 4> corners = {
        project(x_min, y_min),
        project(x_max, y_min),
        project(x_max, y_max),
        project(x_min, y_max)
    };

    // Axis-aligned bounding box of the projected quadrilateral.
    double min_x =  std::numeric_limits<double>::infinity();
    double min_y =  std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& p : corners) {
        if (p.x < min_x) min_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.x > max_x) max_x = p.x;
        if (p.y > max_y) max_y = p.y;
    }

    // Reject if the entire projected region is outside the IR sensor.
    if (max_x < 0.0 || max_y < 0.0 ||
        min_x > static_cast<double>(ir_w_ - 1) ||
        min_y > static_cast<double>(ir_h_ - 1)) {
        return out;
    }

    // Convert to inclusive integer pixel bounds, clamped to valid IR range.
    const int x0 = std::clamp(static_cast<int>(std::floor(min_x)), 0, ir_w_ - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(min_y)), 0, ir_h_ - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(max_x)),  0, ir_w_ - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(max_y)),  0, ir_h_ - 1);

    if (x1 < x0 || y1 < y0) {
        return out;   // clamping collapsed the region
    }

    out.x0    = x0;
    out.y0    = y0;
    out.x1    = x1;
    out.y1    = y1;
    out.valid = true;
    return out;
}
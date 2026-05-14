#include "ir_aligner.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

IRAligner::IRAligner(const std::string& yaml_path)
{
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open IR alignment YAML: " + yaml_path);
    }

    fs["M"] >> M_;
    fs["ir_width"] >> ir_w_;
    fs["ir_height"] >> ir_h_;

    if (M_.empty() || M_.rows != 2 || M_.cols != 3) {
        throw std::runtime_error("Invalid affine matrix M in: " + yaml_path);
    }

    M_.convertTo(M_, CV_64F);
}

IRAligner::PixelRect IRAligner::project_bbox(const BoundingBox& bbox_cam0) const
{
    PixelRect out{0, 0, 0, 0, false};

    if (M_.empty() || M_.rows != 2 || M_.cols != 3 || ir_w_ <= 0 || ir_h_ <= 0) {
        return out;
    }

    // Clamp normalized camera bbox to valid range.
    const double x_min = std::clamp(static_cast<double>(bbox_cam0.x_min), 0.0, 1.0);
    const double y_min = std::clamp(static_cast<double>(bbox_cam0.y_min), 0.0, 1.0);
    const double x_max = std::clamp(static_cast<double>(bbox_cam0.x_max), 0.0, 1.0);
    const double y_max = std::clamp(static_cast<double>(bbox_cam0.y_max), 0.0, 1.0);

    if (x_max <= x_min || y_max <= y_min) {
        return out;
    }

    const double m00 = M_.at<double>(0, 0);
    const double m01 = M_.at<double>(0, 1);
    const double m02 = M_.at<double>(0, 2);
    const double m10 = M_.at<double>(1, 0);
    const double m11 = M_.at<double>(1, 1);
    const double m12 = M_.at<double>(1, 2);

    auto project = [&](double x, double y) -> cv::Point2d {
        return {
            m00 * x + m01 * y + m02,
            m10 * x + m11 * y + m12
        };
    };

    const std::array<cv::Point2d, 4> pts = {
        project(x_min, y_min),
        project(x_max, y_min),
        project(x_max, y_max),
        project(x_min, y_max)
    };

    double min_x =  std::numeric_limits<double>::infinity();
    double min_y =  std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& p : pts) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    // Entire projected bbox is outside the IR image.
    if (max_x < 0.0 || max_y < 0.0 ||
        min_x > static_cast<double>(ir_w_ - 1) ||
        min_y > static_cast<double>(ir_h_ - 1)) {
        return out;
    }

    // Convert continuous projected bounds to inclusive integer pixel bounds.
    int x0 = static_cast<int>(std::floor(min_x));
    int y0 = static_cast<int>(std::floor(min_y));
    int x1 = static_cast<int>(std::ceil(max_x));
    int y1 = static_cast<int>(std::ceil(max_y));

    x0 = std::clamp(x0, 0, ir_w_ - 1);
    y0 = std::clamp(y0, 0, ir_h_ - 1);
    x1 = std::clamp(x1, 0, ir_w_ - 1);
    y1 = std::clamp(y1, 0, ir_h_ - 1);

    if (x1 < x0 || y1 < y0) {
        return out;
    }

    out.x0 = x0;
    out.y0 = y0;
    out.x1 = x1;
    out.y1 = y1;
    out.valid = true;
    return out;
}
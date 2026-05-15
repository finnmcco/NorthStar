#include "stereo_distance.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <optional>


namespace {

// SGBM parameter defaults, copied from disparity_calculator.py so behaviour
// matches between Python prototype and C++ runtime.
constexpr int   SGBM_MIN_DISPARITY     = 0;
constexpr int   SGBM_NUM_DISPARITIES   = 128;   // must be divisible by 16
constexpr int   SGBM_BLOCK_SIZE        = 5;     // must be odd, >= 3
constexpr int   SGBM_UNIQUENESS_RATIO  = 15;
constexpr int   SGBM_SPECKLE_WIN_SIZE  = 100;
constexpr int   SGBM_SPECKLE_RANGE     = 2;
constexpr int   SGBM_DISP12_MAX_DIFF   = 1;
constexpr int   SGBM_PRE_FILTER_CAP    = 63;

// Helper: read a required matrix from FileStorage, throw on failure.
cv::Mat read_required_mat(cv::FileStorage& fs, const std::string& key) {
    cv::FileNode node = fs[key];
    if (node.empty()) {
        throw std::runtime_error("Calibration YAML missing required key: " + key);
    }
    cv::Mat mat;
    node >> mat;
    if (mat.empty()) {
        throw std::runtime_error("Calibration YAML key is not a matrix: " + key);
    }
    return mat;
}

}  // anonymous namespace

StereoDepthEstimator::StereoDepthEstimator(const std::string& calibration_yaml_path) {
    cv::FileStorage fs(calibration_yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open calibration file: " + calibration_yaml_path);
    }

    // Image size
    int width = 0, height = 0;
    fs["image_width"]  >> width;
    fs["image_height"] >> height;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Calibration YAML has invalid image dimensions");
    }
    image_size_ = cv::Size(width, height);

    // Intrinsics and rectification matrices
    K1_       = read_required_mat(fs, "K1");
    D1_       = read_required_mat(fs, "D1");
    K2_       = read_required_mat(fs, "K2");
    D2_       = read_required_mat(fs, "D2");
    R1_       = read_required_mat(fs, "R1");
    P1_       = read_required_mat(fs, "P1");
    R2_       = read_required_mat(fs, "R2");
    P2_       = read_required_mat(fs, "P2");
    Q_        = read_required_mat(fs, "Q");

    // Optional: stored baseline for diagnostics
    cv::FileNode baseline_node = fs["baseline_m"];
    baseline_m_ = baseline_node.empty() ? 0.0f : static_cast<float>(baseline_node.real());

    fs.release();

    // Pre-compute rectification maps. CV_16SC2 is faster at runtime than
    // CV_32FC1 with no meaningful accuracy loss for our resolution.
    cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, image_size_,
                                CV_16SC2, map1_left_, map2_left_);
    cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, image_size_,
                                CV_16SC2, map1_right_, map2_right_);

    // P2[0,3] = -fx * baseline_in_metres. We use this directly for depth:
    // Z = -P2[0,3] / disparity = fx_baseline / disparity.
    fx_baseline_ = -static_cast<float>(P2_.at<double>(0, 3));
    if (fx_baseline_ <= 0.0f) {
        throw std::runtime_error("Calibration P2 matrix has invalid fx*baseline product");
    }

    // SGBM matching parameters: P1 and P2 control smoothness penalty.
    // Standard recipe: 8*channels*block_size^2 and 32*channels*block_size^2.
    const int channels = 1;  // we feed grayscale
    const int sgbm_p1 = 8  * channels * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE;
    const int sgbm_p2 = 32 * channels * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE;

    sgbm_ = cv::StereoSGBM::create(
        SGBM_MIN_DISPARITY,
        SGBM_NUM_DISPARITIES,
        SGBM_BLOCK_SIZE,
        sgbm_p1,
        sgbm_p2,
        SGBM_DISP12_MAX_DIFF,
        SGBM_PRE_FILTER_CAP,
        SGBM_UNIQUENESS_RATIO,
        SGBM_SPECKLE_WIN_SIZE,
        SGBM_SPECKLE_RANGE,
        cv::StereoSGBM::MODE_SGBM_3WAY  // faster than full HH mode
    );
}

std::optional<float> StereoDepthEstimator::compute(
    const cv::Mat& left_raw,
    const cv::Mat& right_raw,
    const BoundingBox& bbox_cam0)
{
    // --- Validate inputs --------------------------------------------------
    if (left_raw.empty() || right_raw.empty()) {
        return std::nullopt;
    }
    if (left_raw.size() != image_size_ || right_raw.size() != image_size_) {
        // Caller's responsibility to ensure raw frames match calibration
        return std::nullopt;
    }
    if (bbox_cam0.x_max <= bbox_cam0.x_min ||
        bbox_cam0.y_max <= bbox_cam0.y_min) {
        return std::nullopt;
    }

    // --- Rectify both images ---------------------------------------------
    cv::remap(left_raw,  left_rect_,  map1_left_,  map2_left_,
              cv::INTER_LINEAR);
    cv::remap(right_raw, right_rect_, map1_right_, map2_right_,
              cv::INTER_LINEAR);

    // --- Convert to grayscale for SGBM -----------------------------------
    cv::cvtColor(left_rect_,  left_gray_,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_rect_, right_gray_, cv::COLOR_BGR2GRAY);

    // --- Run SGBM ---------------------------------------------------------
    cv::Mat disparity_raw;
    sgbm_->compute(left_gray_, right_gray_, disparity_raw);

    // SGBM returns CV_16S, scaled by 16 for sub-pixel precision.
    // Divide by 16 to get float disparity in pixels.
    disparity_raw.convertTo(last_disparity_, CV_32F, 1.0 / 16.0);

    // --- Transform bbox from raw to rectified image space ----------------
    // The bbox came from inference on the unrectified image. Rectification
    // shifts pixels, so we need to transform the bbox corners to find the
    // corresponding region in the rectified disparity map.
    const float img_w = static_cast<float>(image_size_.width);
    const float img_h = static_cast<float>(image_size_.height);

    std::vector<cv::Point2f> corners_raw = {
        { bbox_cam0.x_min * img_w, bbox_cam0.y_min * img_h },
        { bbox_cam0.x_max * img_w, bbox_cam0.y_max * img_h }
    };
    std::vector<cv::Point2f> corners_rect;
    cv::undistortPoints(corners_raw, corners_rect, K1_, D1_, R1_, P1_);

    // Clamp to image bounds (corners can fall outside after rectification)
    int x0 = std::max(0, static_cast<int>(std::floor(corners_rect[0].x)));
    int y0 = std::max(0, static_cast<int>(std::floor(corners_rect[0].y)));
    int x1 = std::min(image_size_.width  - 1,
                      static_cast<int>(std::ceil(corners_rect[1].x)));
    int y1 = std::min(image_size_.height - 1,
                      static_cast<int>(std::ceil(corners_rect[1].y)));

    if (x1 <= x0 || y1 <= y0) {
        return std::nullopt;  // bbox projected entirely outside image
    }

    // --- Collect valid disparity values inside the rectified bbox ---------
    std::vector<float> samples;
    samples.reserve((x1 - x0 + 1) * (y1 - y0 + 1));

    for (int y = y0; y <= y1; ++y) {
        const float* row = last_disparity_.ptr<float>(y);
        for (int x = x0; x <= x1; ++x) {
            float d = row[x];
            if (d >= MIN_TRUSTWORTHY_DISPARITY_PX) {
                samples.push_back(d);
            }
        }
    }

    if (static_cast<int>(samples.size()) < MIN_VALID_SAMPLES) {
        return std::nullopt;  // not enough texture / too many holes
    }

    // --- Take robust median of disparity, convert to depth ---------------
    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    const float median_disparity = samples[mid];

    // Z = (fx * baseline) / disparity
    const float depth_m = fx_baseline_ / median_disparity;

    // Sanity bounds: anything below 5cm or above 10m is almost certainly wrong.
    if (depth_m < 0.05f || depth_m > 10.0f) {
        return std::nullopt;
    }

    return depth_m;
}

#pragma once

#include <optional>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "sensor-fusion-structs.hpp"  
#include "inference_packet.hpp"

/*
 * StereoDepthEstimator
 *
 * Loads stereo calibration from a YAML file produced by stereo_calibrate.py,
 * pre-computes rectification maps and SGBM instance at construction time,
 * and exposes a single on-demand method that takes a raw stereo frame pair
 * and a bounding box, and returns the depth at that bbox.
 *
 * Usage:
 *
 *     StereoDepthEstimator depth("stereo_calib_640.yaml");
 *     // ... later, when the filter triggers a query ...
 *     auto Z = depth.compute(left_raw, right_raw, bbox_cam0);
 *     if (Z) {
 *         std::cout << "Object at " << *Z << " m\n";
 *     }
 *
 * Not thread-safe. Intended to be called from a single fusion thread, on
 * demand, not concurrently. Internal SGBM state is reused between calls.
 */
class StereoDepthEstimator {
public:
    // Loads calibration from YAML, builds rectification maps and SGBM instance.
    // Throws std::runtime_error if the YAML can't be read or is malformed.
    explicit StereoDepthEstimator(const std::string& calibration_yaml_path);

    // Disable copy/move: SGBM pointer + cv::Mat members aren't worth the
    // complexity of supporting these, and we only ever want one instance.
    StereoDepthEstimator(const StereoDepthEstimator&) = delete;
    StereoDepthEstimator& operator=(const StereoDepthEstimator&) = delete;

    // Computes depth (in metres) at the given bbox, given a raw stereo pair.
    //
    // Internally:
    //   1. Rectifies both images using pre-computed maps
    //   2. Runs SGBM on the rectified pair
    //   3. Transforms the bbox from raw-image to rectified-image coordinates
    //   4. Samples the disparity map inside that region
    //   5. Returns the median depth, or nullopt if there aren't enough
    //      valid disparity samples to be confident.
    //
    // Returns nullopt if:
    //   - The bbox is empty or invalid
    //   - Too few valid disparity values inside the bbox (textureless region)
    //   - The median disparity is suspiciously small (object too far)
    std::optional<float> compute(const cv::Mat& left_raw,
                                 const cv::Mat& right_raw,
                                 const BoundingBox& bbox_cam0);

    // Diagnostic accessor: the most recent disparity map computed.
    // Useful for debugging / saving to disk. Empty before first compute() call.
    const cv::Mat& last_disparity() const { return last_disparity_; }

    // Diagnostic accessor: baseline recovered from calibration (metres).
    float baseline_m() const { return baseline_m_; }

private:
    // Rectification maps for both cameras (CV_16SC2 for speed)
    cv::Mat map1_left_, map2_left_;
    cv::Mat map1_right_, map2_right_;

    // Intrinsics needed for transforming bbox coordinates from raw to rectified
    cv::Mat K1_, D1_, R1_, P1_;

    // SGBM stays alive across calls so we don't pay reconstruction cost
    cv::Ptr<cv::StereoSGBM> sgbm_;

    // The Q matrix is unused now (we compute Z directly from fx*B/d), but
    // we keep it loaded in case future work wants 3D points via reprojectImageTo3D.
    cv::Mat Q_;

    // Image dimensions from calibration (e.g. 640x640)
    cv::Size image_size_;

    // Cached scalar: fx * baseline in pixel*metres, derived from P2.
    // Z = fx_baseline_ / disparity.
    float fx_baseline_;

    // Cached for diagnostics
    float baseline_m_;

    // Reusable scratch buffers - avoid per-call allocation
    cv::Mat left_rect_, right_rect_;
    cv::Mat left_gray_, right_gray_;
    cv::Mat last_disparity_;       // float32, in pixels

    // Minimum number of valid disparity samples inside a bbox to trust the median.
    // With a typical 100x100 px bbox at 640x640, you'd have ~10000 candidate pixels;
    // a sensible threshold is somewhere between 50 (just enough for a stable median)
    // and a percentage of the bbox area.
    static constexpr int MIN_VALID_SAMPLES = 50;

    // Disparity values below this are considered unreliable. At ~1px disparity
    // depth is so sensitive to noise that the estimate is meaningless.
    static constexpr float MIN_TRUSTWORTHY_DISPARITY_PX = 1.5f;
};
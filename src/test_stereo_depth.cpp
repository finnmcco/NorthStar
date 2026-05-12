// test_stereo_depth.cpp
//
// Standalone test for StereoDepthEstimator.
// Loads a calibration YAML and a stereo image pair, runs the estimator on a
// specified bounding box, and prints median disparity + computed depth.
//
// Usage:
//     ./test_stereo_depth <calib.yaml> <left.png> <right.png> \
//         [x_min y_min x_max y_max]
//
// All bbox coords are normalised [0, 1]. Default is a centred 20% box.
// Optional: pass --measured <metres> to print the error vs ground truth.

#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/imgcodecs.hpp>

#include "stereo_distance.hpp"
#include "sensor-fusion-structs.hpp"



namespace {

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <calib.yaml> <left.png> <right.png>\n"
        << "       [x_min y_min x_max y_max] [--measured <metres>]\n"
        << "\n"
        << "  calib.yaml      Stereo calibration written by stereo_calibrate.py\n"
        << "  left.png        Left/cam0 image, 640x640 BGR\n"
        << "  right.png       Right/cam1 image, 640x640 BGR\n"
        << "  x_min ... y_max Bbox in normalised [0,1] (default: 0.4..0.6)\n"
        << "  --measured      Optional ground-truth depth in metres, for error %\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string calib_path  = argv[1];
    const std::string left_path   = argv[2];
    const std::string right_path  = argv[3];

    // Default bbox: centred 20% of the frame
    BoundingBox bbox{ 0.4f, 0.4f, 0.6f, 0.6f };
    std::optional<float> measured_depth_m;

    // Parse optional bbox and --measured flag
    int i = 4;
    if (argc >= 8 &&
        std::string(argv[4]) != "--measured")
    {
        bbox.x_min = std::stof(argv[4]);
        bbox.y_min = std::stof(argv[5]);
        bbox.x_max = std::stof(argv[6]);
        bbox.y_max = std::stof(argv[7]);
        i = 8;
    }
    if (i < argc - 1 && std::string(argv[i]) == "--measured") {
        measured_depth_m = std::stof(argv[i + 1]);
    }

    // --- Load images ------------------------------------------------------
    cv::Mat left  = cv::imread(left_path);
    cv::Mat right = cv::imread(right_path);
    if (left.empty()) {
        std::cerr << "Failed to read left image: " << left_path << "\n";
        return 1;
    }
    if (right.empty()) {
        std::cerr << "Failed to read right image: " << right_path << "\n";
        return 1;
    }
    if (left.size() != right.size()) {
        std::cerr << "Image sizes don't match: "
                  << left.cols << "x" << left.rows << " vs "
                  << right.cols << "x" << right.rows << "\n";
        return 1;
    }

    // --- Construct estimator ---------------------------------------------
    std::cout << "Loading calibration: " << calib_path << "\n";
    std::unique_ptr<StereoDepthEstimator> estimator;
    try {
        estimator = std::make_unique<StereoDepthEstimator>(calib_path);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to construct estimator: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "  baseline: " << std::fixed << std::setprecision(1)
              << estimator->baseline_m() * 1000.0f << " mm\n";

    // --- Run compute ------------------------------------------------------
    std::cout << "\nBounding box (normalised): " << std::setprecision(3)
          << "[" << bbox.x_min << ", " << bbox.y_min << "] to "
          << "[" << bbox.x_max << ", " << bbox.y_max << "]\n";

    auto depth_m = estimator->compute(left, right, bbox);

    // --- Print disparity stats inside the bbox (for diagnostics) ---------
    // We re-derive the median disparity from the cached last_disparity_ map
    // so the user can see what SGBM produced, not just the final depth.
    const cv::Mat& disparity = estimator->last_disparity();
    if (!disparity.empty()) {
        // Note: this samples in raw-image bbox space, NOT rectified. So the
        // numbers will be slightly different from what compute() used
        // internally (which transforms to rectified bbox space first).
        // For a real diagnostic, we'd want compute() to expose the median;
        // for a quick eyeball this is close enough.
        int x0 = static_cast<int>(bbox.x_min * disparity.cols);
        int y0 = static_cast<int>(bbox.y_min * disparity.rows);
        int x1 = static_cast<int>(bbox.x_max * disparity.cols);
        int y1 = static_cast<int>(bbox.y_max * disparity.rows);

        std::vector<float> samples;
        for (int y = y0; y < y1; ++y) {
            const float* row = disparity.ptr<float>(y);
            for (int x = x0; x < x1; ++x) {
                if (row[x] > 1.0f) samples.push_back(row[x]);
            }
        }

        std::cout << "\nDisparity inside bbox (raw coords):\n";
        std::cout << "  valid samples: " << samples.size() << " / "
                  << (x1 - x0) * (y1 - y0) << "\n";
        if (!samples.empty()) {
            std::nth_element(samples.begin(),
                             samples.begin() + samples.size() / 2,
                             samples.end());
            float median = samples[samples.size() / 2];
            float minv = *std::min_element(samples.begin(), samples.end());
            float maxv = *std::max_element(samples.begin(), samples.end());
            std::cout << "  median: " << std::fixed << std::setprecision(2)
                      << median << " px\n";
            std::cout << "  min/max: " << minv << " / " << maxv << " px\n";
        }
    }

    // --- Print final depth -----------------------------------------------
    std::cout << "\nComputed depth: ";
    if (depth_m) {
        std::cout << std::fixed << std::setprecision(3) << *depth_m << " m\n";

        if (measured_depth_m) {
            float error_m = *depth_m - *measured_depth_m;
            float error_pct = 100.0f * error_m / *measured_depth_m;
            std::cout << "Ground truth:   " << *measured_depth_m << " m\n";
            std::cout << "Error:          " << std::showpos << error_m
                      << " m (" << error_pct << "%)" << std::noshowpos << "\n";
        }
    } else {
        std::cout << "(no valid depth)\n";
    }

    return 0;
}
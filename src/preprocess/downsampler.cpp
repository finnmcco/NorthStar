// Downsampler.cpp
#include "downsampler.hpp"

#include <opencv2/core.hpp>     // cv::Mat, cv::Rect
#include <opencv2/imgproc.hpp>  // cv::resize

/*
    Implementation notes:

    We wrap the raw pointers (inData/outData) using cv::Mat "headers".
    This does NOT copy the image data; it just creates a view.

    Steps:
    1) Validate arguments
    2) Create cv::Mat for input with correct stride
    3) Compute a centered square crop region
    4) Create a cv::Mat ROI view into the input (no copy)
    5) Create cv::Mat for output (points at outData)
    6) Resize ROI -> output using INTER_AREA (good for downsampling)
*/

bool DownsamplerRGB::process(const uint8_t* inData,
                             int inWidth,
                             int inHeight,
                             int inStrideBytes,
                             uint8_t* outData,
                             int outStrideBytes) const
{
    // ----------------------------
    // 1) Basic argument validation
    // ----------------------------
    if (!inData || !outData)
        return false;

    if (inWidth <= 0 || inHeight <= 0)
        return false;

    // Input stride must be large enough to hold one row of RGB pixels
    const int minInStride = inWidth * kChannels;
    if (inStrideBytes < minInStride)
        return false;

    // Output stride must be large enough for 640 RGB pixels per row
    const int minOutStride = kOutWidth * kChannels;
    if (outStrideBytes < minOutStride)
        return false;

    // ---------------------------------------------
    // 2) Wrap input buffer as an OpenCV cv::Mat view
    // ---------------------------------------------
    // CV_8UC3 = 8-bit unsigned, 3 channels (RGB)
    // Step/stride is specified in bytes (inStrideBytes).
    cv::Mat input(inHeight, inWidth, CV_8UC3, const_cast<uint8_t*>(inData), inStrideBytes);

    // ----------------------------------------------------
    // 3) Compute a centered square crop (remove wide edges)
    // ----------------------------------------------------
    // side is the largest square that fits in the image
    const int side = (inWidth < inHeight) ? inWidth : inHeight;

    // Center the square crop
    const int x0 = (inWidth  - side) / 2;
    const int y0 = (inHeight - side) / 2;

    // Ensure ROI is inside bounds (should be by construction)
    cv::Rect roi(x0, y0, side, side);

    // -----------------------------------------
    // 4) Create ROI view (still no data copying)
    // -----------------------------------------
    cv::Mat cropped = input(roi);

    // ----------------------------------------------
    // 5) Wrap output buffer as an OpenCV cv::Mat view
    // ----------------------------------------------
    cv::Mat output(kOutHeight, kOutWidth, CV_8UC3, outData, outStrideBytes);

    // ----------------------------------------
    // 6) Resize with INTER_AREA (downsampling)
    // ----------------------------------------
    // INTER_AREA is typically best for shrinking images; it behaves like
    // pixel area relation / averaging when scaling down.
    cv::resize(cropped, output, output.size(), 0.0, 0.0, cv::INTER_LINEAR);

    return true;
}
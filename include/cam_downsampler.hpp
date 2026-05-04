// Downsampler.hpp
#pragma once

#include <cstddef>   // std::size_t
#include <cstdint>   // uint8_t

/*
    Downsampler (RGB)

    Purpose:
    --------
    - Takes an input RGB image 
    - Crops it to a centered square region (removes "wide" left/right areas)
    - Downsamples that square to a fixed output size (default 640x640)
    - Writes the result into an output buffer you provide (from BufferPool)

    Why a separate module:
    ----------------------
    - Keeps CameraCapture "thin" and timing-focused
    - Lets you test crop + downsample without libcamera
    - Lets you swap implementations later (custom averaging, SIMD/NEON, libyuv, etc.)

    Input format:
    -------------
    - Interleaved RGB, 8-bit per channel (RGBRGBRGB...)
    - Stride is in BYTES per row (may be >= width * 3)

    Output format:
    --------------
    - Interleaved RGB, 8-bit per channel
    - Typically contiguous: outStrideBytes = outWidth * 3

    Note on downsampling method:
    ----------------------------
    - The implementation uses OpenCV cv::resize with INTER_AREA, which is
      the usual choice for "area-like" downsampling (behaves like averaging
      when scaling down).
*/

class CamDownsampler
{
public:
    // Output dimensions are fixed for your CV model
    static constexpr int kOutWidth  = 640;
    static constexpr int kOutHeight = 640;
    static constexpr int kChannels  = 3;

    CamDownsampler() = default;

    // Disable copying (not strictly necessary, but keeps usage simple/explicit)
    CamDownsampler(const CamDownsampler&) = delete;
    CamDownsampler& operator=(const CamDownsampler&) = delete;

    /*
        Process one frame.

        Parameters:
        - inData: pointer to the first byte of the input image
        - inWidth: input width in pixels
        - inHeight: input height in pixels
        - inStrideBytes: bytes per row of input (>= inWidth * 3)
        - outData: pointer to output buffer (must be large enough)
        - outStrideBytes: bytes per row of output (>= 640 * 3). If you always
                          store output as tightly packed RGB, pass 640*3.

        Returns:
        - true on success
        - false if input arguments are invalid

        Ownership / lifetime:
        - This function does NOT allocate memory.
        - It writes into outData (usually acquired from BufferPool).
        - Caller is responsible for ensuring outData remains valid until the
          downstream consumer is finished with it.
    */
    bool process(const uint8_t* inData,
                 int inWidth,
                 int inHeight,
                 int inStrideBytes,
                 uint8_t* outData,
                 int outStrideBytes) const;
};
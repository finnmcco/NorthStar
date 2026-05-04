#pragma once

#include "MLX90640.hpp"
#include "ThermalAnalyser.hpp"
#include <array>
#include <cstddef>
#include <optional>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  ThermalUtils
//
//  Processing utilities for MLX90640 temperature frames.
//
//  Sections:
//    0. ChessInterpolator  — suppress checkerboard artefact from subpage interleave
//    1. FrameAverager      — rolling average over N frames (noise reduction)
//    2. BackgroundModel    — per-pixel background subtraction / delta frames
//    3. BlobDetector       — connected-component warm-object detection
//    4. CoordTransform     — raw sensor ↔ display pixel coordinate conversion
// ─────────────────────────────────────────────────────────────────────────────

namespace MLX90640 {

// ── 0. ChessInterpolator ─────────────────────────────────────────────────────
//
// Suppresses the checkerboard artefact that appears in chess-mode captures
// when a moving subject creates a thermal mismatch between the two subpages.
//
// The MLX90640 in chess mode updates only half the pixels per subpage —
// the two halves form an interlocked checkerboard. When a warm object moves
// between the two captures, pixels from the stale subpage appear anomalously
// cold (or hot), producing the visible grid pattern.
//
// Fix: after compositing both subpages, replace each pixel with the weighted
// average of itself and its 4-connected neighbours. Neighbours always belong
// to the *opposite* subpage, so this blends the two captures together and
// eliminates the discontinuity.
//
// Two modes are provided:
//   - interpolate()      : replaces stale-subpage pixels with neighbour mean.
//                          Sharpest edges, lowest latency. Use when the
//                          object is slow-moving relative to the frame rate.
//   - interpolateBlend() : blends ALL pixels toward their neighbours with a
//                          tunable weight (default 0.5). Softer but more
//                          robust for fast-moving objects or high refresh rates.

class ChessInterpolator {
public:
    // Replace pixels belonging to `staleSubpage` with the mean of their
    // valid 4-connected neighbours (which belong to the opposite subpage).
    // staleSubpage: 0 or 1 — the subpage that was captured first (older data).
    // Writes result into out[PIXEL_COUNT]. Safe to alias temps == out.
    static void interpolate(const float* temps, float* out, int staleSubpage);

    // Blend every pixel toward the mean of its 4-connected neighbours.
    // weight: 0.0 = no change, 1.0 = full neighbour mean, 0.5 = half/half.
    // Does not require knowledge of subpage — safe to call every frame.
    static void interpolateBlend(const float* temps, float* out, float weight = 0.5f);

private:
    // Returns the chess subpage index (0 or 1) for pixel at (row, col).
    static int chessSubpage(int row, int col) {
        int ilPattern    = (row >> 1) - ((row >> 2) << 1);
        return ilPattern ^ (col & 1);
    }
};

// ── 1. FrameAverager ─────────────────────────────────────────────────────────
//
// Maintains a fixed-size ring buffer of complete frames and exposes the
// per-pixel rolling mean. At 16 Hz, capacity=4 gives ~250 ms smoothing
// while still feeling real-time.

class FrameAverager {
public:
    explicit FrameAverager(std::size_t capacity = 4);

    // Push a new frame into the ring buffer.
    void push(const float* temps);

    // Returns true once the buffer has been filled at least once —
    // i.e. the average is over a full window.
    bool ready() const;

    // Number of frames currently accumulated (≤ capacity).
    std::size_t count() const;

    // Write the per-pixel rolling mean into out[PIXEL_COUNT].
    void getAverage(float* out) const;

private:
    std::size_t              capacity_;
    std::size_t              head_  = 0;
    std::size_t              count_ = 0;
    // ring buffer: capacity_ × PIXEL_COUNT
    std::vector<float>       buf_;
};

// ── 2. BackgroundModel ───────────────────────────────────────────────────────
//
// Learns a per-pixel background temperature by exponential moving average,
// then produces a delta frame (foreground - background).
//
// alpha controls adaptation speed: small alpha = slow adaptation (stable
// background), large alpha = fast adaptation (follows slow scene changes).
// Typical value: 0.05 (background settles in ~20 frames).

class BackgroundModel {
public:
    explicit BackgroundModel(float alpha = 0.05f);

    // True once at least one frame has been ingested.
    bool initialised() const;

    // Update the background model with a new frame.
    // On the first call the background is seeded directly from the frame.
    void update(const float* temps);

    // Write the per-pixel delta (temps - background) into delta[PIXEL_COUNT].
    // Requires initialised() == true; throws otherwise.
    void getDelta(const float* temps, float* delta) const;

    // Direct access to the background estimate.
    void getBackground(float* out) const;

    // Reset — clears the background so the next update() re-seeds it.
    void reset();

    float alpha() const { return alpha_; }
    void  setAlpha(float a) { alpha_ = a; }

private:
    float              alpha_;
    bool               init_ = false;
    std::array<float, PIXEL_COUNT> bg_{};
};

// ── 3. BlobDetector ──────────────────────────────────────────────────────────
//
// Finds connected regions of pixels above a temperature threshold.
// Connectivity: 4-connected (up/down/left/right).
//
// Useful for:
//   - Occupancy detection (person-sized blobs at ~30–35 °C)
//   - Hot-component localisation
//   - Feeding bounding boxes to ThermalAnalyser

struct Blob {
    BBox  bbox;           // tight bounding box in raw sensor coords
    int   pixelCount;     // number of pixels in the blob
    float peakTemp;       // maximum temperature in the blob
    float meanTemp;       // mean temperature in the blob
    int   centroidX;      // centroid column (rounded)
    int   centroidY;      // centroid row    (rounded)
};

class BlobDetector {
public:
    // threshold : minimum temperature (°C) for a pixel to be considered "hot"
    // minPixels : blobs smaller than this are discarded (noise rejection)
    explicit BlobDetector(float threshold = 28.0f, int minPixels = 4);

    // Run detection on a frame. Returns blobs sorted by pixelCount descending.
    std::vector<Blob> detect(const float* temps) const;

    float threshold()  const { return threshold_; }
    int   minPixels()  const { return minPixels_; }
    void  setThreshold(float t) { threshold_ = t; }
    void  setMinPixels(int   n) { minPixels_ = n; }

private:
    float threshold_;
    int   minPixels_;
};

// ── 4. CoordTransform ────────────────────────────────────────────────────────
//
// Stateless helpers for converting between raw sensor coordinates (32×24)
// and scaled display coordinates (W*scale × H*scale).
//
// The horizontal flip applied in Main.cpp is accounted for — pass
// flipped=true (the default) to get display coords that match what's on screen.

struct DisplayCoord { int x, y; };
struct SensorCoord  { int x, y; };

class CoordTransform {
public:
    // scale : the SCALE constant from Main.cpp (default 20)
    // flipped : whether the display has been horizontally flipped
    explicit CoordTransform(int scale = SENSOR_W * 20 / SENSOR_W,
                            bool flipped = true);

    // Raw sensor pixel → display pixel (top-left of that sensor pixel's cell)
    DisplayCoord toDisplay(SensorCoord s) const;

    // Display pixel → raw sensor pixel
    SensorCoord  toSensor(DisplayCoord d) const;

    // Convert a BBox from sensor coords to a display-space cv::Rect-compatible
    // {x, y, width, height}.
    struct DisplayRect { int x, y, w, h; };
    DisplayRect  toDisplayRect(const BBox& b) const;

    // Convert a display-space bounding box back to sensor coords.
    BBox         toSensorBBox(int dx0, int dy0, int dx1, int dy1) const;

    int  scale()   const { return scale_;   }
    bool flipped() const { return flipped_; }

private:
    int  scale_;
    bool flipped_;
};

} // namespace MLX90640
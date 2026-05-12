#include "ThermalUtils.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <cmath>

namespace MLX90640 {

// ─────────────────────────────────────────────────────────────────────────────
//  ChessInterpolator
// ─────────────────────────────────────────────────────────────────────────────

void ChessInterpolator::interpolate(const float* temps, float* out, int staleSubpage) {
    // Copy first so we can alias temps == out safely
    if (out != temps)
        std::memcpy(out, temps, PIXEL_COUNT * sizeof(float));

    for (int row = 0; row < SENSOR_H; ++row) {
        for (int col = 0; col < SENSOR_W; ++col) {
            if (chessSubpage(row, col) != staleSubpage) continue;

            // Accumulate valid 4-connected neighbours (opposite subpage)
            float sum   = 0.0f;
            int   count = 0;

            auto add = [&](int r, int c) {
                if (r < 0 || r >= SENSOR_H || c < 0 || c >= SENSOR_W) return;
                sum += temps[r * SENSOR_W + c];
                ++count;
            };

            add(row - 1, col);
            add(row + 1, col);
            add(row,     col - 1);
            add(row,     col + 1);

            if (count > 0)
                out[row * SENSOR_W + col] = sum / static_cast<float>(count);
        }
    }
}

void ChessInterpolator::interpolateBlend(const float* temps, float* out, float weight) {
    weight = std::clamp(weight, 0.0f, 1.0f);
    const float selfW = 1.0f - weight;

    for (int row = 0; row < SENSOR_H; ++row) {
        for (int col = 0; col < SENSOR_W; ++col) {
            float sum   = 0.0f;
            int   count = 0;

            auto add = [&](int r, int c) {
                if (r < 0 || r >= SENSOR_H || c < 0 || c >= SENSOR_W) return;
                sum += temps[r * SENSOR_W + c];
                ++count;
            };

            add(row - 1, col);
            add(row + 1, col);
            add(row,     col - 1);
            add(row,     col + 1);

            float neighbourMean = (count > 0) ? sum / static_cast<float>(count) : temps[row * SENSOR_W + col];
            out[row * SENSOR_W + col] = selfW * temps[row * SENSOR_W + col] + weight * neighbourMean;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FrameAverager
// ─────────────────────────────────────────────────────────────────────────────

FrameAverager::FrameAverager(std::size_t capacity)
    : capacity_(capacity)
    , buf_(capacity * PIXEL_COUNT, 0.0f)
{}

void FrameAverager::push(const float* temps) {
    float* slot = buf_.data() + head_ * PIXEL_COUNT;
    std::memcpy(slot, temps, PIXEL_COUNT * sizeof(float));
    head_ = (head_ + 1) % capacity_;
    if (count_ < capacity_) ++count_;
}

bool FrameAverager::ready() const {
    return count_ == capacity_;
}

std::size_t FrameAverager::count() const {
    return count_;
}

void FrameAverager::getAverage(float* out) const {
    if (count_ == 0) {
        std::fill(out, out + PIXEL_COUNT, 0.0f);
        return;
    }

    std::fill(out, out + PIXEL_COUNT, 0.0f);

    for (std::size_t f = 0; f < count_; ++f) {
        const float* frame = buf_.data() + f * PIXEL_COUNT;
        for (int p = 0; p < PIXEL_COUNT; ++p)
            out[p] += frame[p];
    }

    float inv = 1.0f / static_cast<float>(count_);
    for (int p = 0; p < PIXEL_COUNT; ++p)
        out[p] *= inv;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BackgroundModel
// ─────────────────────────────────────────────────────────────────────────────

BackgroundModel::BackgroundModel(float alpha)
    : alpha_(alpha)
{}

bool BackgroundModel::initialised() const {
    return init_;
}

void BackgroundModel::update(const float* temps) {
    if (!init_) {
        std::memcpy(bg_.data(), temps, PIXEL_COUNT * sizeof(float));
        init_ = true;
        return;
    }

    // Exponential moving average: bg = (1-alpha)*bg + alpha*temps
    for (int p = 0; p < PIXEL_COUNT; ++p)
        bg_[p] = bg_[p] + alpha_ * (temps[p] - bg_[p]);
}

void BackgroundModel::getDelta(const float* temps, float* delta) const {
    if (!init_)
        throw std::runtime_error("BackgroundModel: not yet initialised — call update() first");

    for (int p = 0; p < PIXEL_COUNT; ++p)
        delta[p] = temps[p] - bg_[p];
}

void BackgroundModel::getBackground(float* out) const {
    std::memcpy(out, bg_.data(), PIXEL_COUNT * sizeof(float));
}

void BackgroundModel::reset() {
    init_ = false;
    bg_.fill(0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  BlobDetector
// ─────────────────────────────────────────────────────────────────────────────

BlobDetector::BlobDetector(float threshold, int minPixels)
    : threshold_(threshold)
    , minPixels_(minPixels)
{}

std::vector<Blob> BlobDetector::detect(const float* temps) const {
    // Label map: -1 = unvisited, -2 = below threshold, ≥0 = blob id
    std::array<int, PIXEL_COUNT> labels;
    labels.fill(-1);

    // Mark cold pixels
    for (int p = 0; p < PIXEL_COUNT; ++p)
        if (temps[p] < threshold_) labels[p] = -2;

    // 4-connected flood fill
    static constexpr int dx[] = { 1, -1,  0,  0 };
    static constexpr int dy[] = { 0,  0,  1, -1 };

    std::vector<Blob> blobs;
    int nextId = 0;

    for (int seed = 0; seed < PIXEL_COUNT; ++seed) {
        if (labels[seed] != -1) continue;   // already visited or cold

        // BFS from this seed
        const int blobId = nextId++;
        std::queue<int> queue;
        queue.push(seed);
        labels[seed] = blobId;

        // Accumulate blob stats as we go
        int   count    = 0;
        float peakTemp = temps[seed];
        float sumTemp  = 0.0f;
        int   sumX     = 0, sumY = 0;
        int   minX = SENSOR_W, maxX = 0;
        int   minY = SENSOR_H, maxY = 0;

        while (!queue.empty()) {
            int idx = queue.front(); queue.pop();
            int row = idx / SENSOR_W;
            int col = idx % SENSOR_W;

            ++count;
            float t = temps[idx];
            if (t > peakTemp) peakTemp = t;
            sumTemp += t;
            sumX    += col;
            sumY    += row;
            if (col < minX) minX = col;
            if (col > maxX) maxX = col;
            if (row < minY) minY = row;
            if (row > maxY) maxY = row;

            // Expand to 4-neighbours
            for (int d = 0; d < 4; ++d) {
                int nc = col + dx[d];
                int nr = row + dy[d];
                if (nc < 0 || nc >= SENSOR_W || nr < 0 || nr >= SENSOR_H) continue;
                int nIdx = nr * SENSOR_W + nc;
                if (labels[nIdx] != -1) continue;
                labels[nIdx] = blobId;
                queue.push(nIdx);
            }
        }

        if (count < minPixels_) continue;   // too small — noise

        Blob b;
        b.bbox       = { minX, minY, maxX, maxY };
        b.pixelCount = count;
        b.peakTemp   = peakTemp;
        b.meanTemp   = sumTemp / static_cast<float>(count);
        b.centroidX  = static_cast<int>(std::round(
                            static_cast<float>(sumX) / static_cast<float>(count)));
        b.centroidY  = static_cast<int>(std::round(
                            static_cast<float>(sumY) / static_cast<float>(count)));
        blobs.push_back(b);
    }

    // Sort largest-first — most significant objects come first
    std::sort(blobs.begin(), blobs.end(),
              [](const Blob& a, const Blob& b) {
                  return a.pixelCount > b.pixelCount;
              });

    return blobs;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CoordTransform
// ─────────────────────────────────────────────────────────────────────────────

CoordTransform::CoordTransform(int scale, bool flipped)
    : scale_(scale)
    , flipped_(flipped)
{}

DisplayCoord CoordTransform::toDisplay(SensorCoord s) const {
    int x = flipped_ ? (SENSOR_W - 1 - s.x) * scale_ : s.x * scale_;
    int y = s.y * scale_;
    return { x, y };
}

SensorCoord CoordTransform::toSensor(DisplayCoord d) const {
    int col = d.x / scale_;
    if (flipped_) col = SENSOR_W - 1 - col;
    int row = d.y / scale_;
    col = std::clamp(col, 0, SENSOR_W - 1);
    row = std::clamp(row, 0, SENSOR_H - 1);
    return { col, row };
}

CoordTransform::DisplayRect CoordTransform::toDisplayRect(const BBox& b) const {
    // After flip, x0 (left column) maps to the right side of the display,
    // so we need to recompute the left edge from x1 when flipped.
    int dispX, dispW;
    dispW = (b.x1 - b.x0 + 1) * scale_;

    if (flipped_)
        dispX = (SENSOR_W - 1 - b.x1) * scale_;
    else
        dispX = b.x0 * scale_;

    int dispY = b.y0 * scale_;
    int dispH = (b.y1 - b.y0 + 1) * scale_;
    return { dispX, dispY, dispW, dispH };
}

BBox CoordTransform::toSensorBBox(int dx0, int dy0, int dx1, int dy1) const {
    SensorCoord tl = toSensor({ dx0, dy0 });
    SensorCoord br = toSensor({ dx1, dy1 });

    // Ensure x0 ≤ x1 after potential flip inversion
    BBox b;
    b.x0 = std::min(tl.x, br.x);
    b.x1 = std::max(tl.x, br.x);
    b.y0 = std::min(tl.y, br.y);
    b.y1 = std::max(tl.y, br.y);
    return b;
}

} // namespace MLX90640
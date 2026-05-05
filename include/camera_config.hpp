#pragma once
#include <cstdint>

// ── Sensor resolution ─────────────────────────────────────────────────────────
static constexpr int CAMERA_WIDTH  = 1920;
static constexpr int CAMERA_HEIGHT = 1080;
static constexpr int CAMERA_FPS    = 30;

// ── Pipeline ──────────────────────────────────────────────────────────────────
static constexpr int FRAME_QUEUE_DEPTH = 4;

// ── Stereo matching ───────────────────────────────────────────────────────────
// Two results are paired if their timestamps are within this window.
// Currently set to half a frame at CAMERA_FPS — needs tuning once dual-camera
// capture is validated and real timestamp deltas can be measured.
static constexpr uint64_t STEREO_MATCH_TOLERANCE_NS =
    1'000'000'000ULL / (CAMERA_FPS * 2);   // 16 666 666 ns at 30 fps
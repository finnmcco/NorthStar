#pragma once
#include <cstdint>


static constexpr int N_CLASSES    = 80;   // COCO class count
static constexpr int REG_MAX      = 16;   // DFL bin count — must match compiled model
static constexpr int INPUT_WIDTH  = 640;  // model input width  (pixels)
static constexpr int INPUT_HEIGHT = 640;  // model input height (pixels)


static constexpr float CONF_THRESHOLD = 0.25f; // minimum confidence to keep a detection
static constexpr float NMS_IOU_THRESH = 0.45f; // IoU above which weaker box is suppressed

static constexpr int INFERENCE_TIMEOUT_S = 5;  // seconds to wait for a callback in tests
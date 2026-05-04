#pragma once
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

/*
    detection_utils.hpp
    ════════════════════
    Utilities shared by every binary that consumes Hailo NMS output.

    Detection
    ─────────
    Normalised bounding box in [0,1] × [0,1] image space (x-axis = width).
    Multiply by the image dimensions to get pixel coordinates.

    parse_nms_output(raw)
    ─────────────────────
    Parses the raw byte buffer produced by Hailo's NMS post-processor.

    HailoRT NMS output format (HAILO_FORMAT_ORDER_HAILO_NMS):
        For each of the N_CLASSES classes:
            float32  bbox_count                  ← number of valid boxes
            float32  y_min, x_min, y_max, x_max, score  × MAX_BBOXES

    MAX_BBOXES is inferred from the buffer size so this function is independent
    of the compile-time model parameters.

    draw_detections(img, dets)
    ──────────────────────────
    Draws bounding boxes and class labels onto a BGR OpenCV Mat in-place.

    COCO_CLASSES
    ─────────────
    80-entry label table indexed by class_id.

    CONF_THRESHOLD
    ──────────────
    Detections below this confidence are discarded by parse_nms_output().
    Change here and recompile — no cmake re-run needed.
*/

// ─────────────────────────────────────────────────────────────────────────────
//  Tunable
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float CONF_THRESHOLD = 0.45f;

// ─────────────────────────────────────────────────────────────────────────────
//  Detection result
// ─────────────────────────────────────────────────────────────────────────────
struct Detection {
    uint8_t class_id;
    float   score;
    float   x_min, y_min, x_max, y_max; // normalised [0,1]
};

// ─────────────────────────────────────────────────────────────────────────────
//  API
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& raw);

void draw_detections(cv::Mat& bgr_img, const std::vector<Detection>& dets);

// ─────────────────────────────────────────────────────────────────────────────
//  COCO label table
// ─────────────────────────────────────────────────────────────────────────────
extern const char* const COCO_CLASSES[80];

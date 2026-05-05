#pragma once
#include <cstdint>
#include <vector>
#include <string>

#include <opencv2/core.hpp>

// ─────────────────────────────────────────────────────────────────────────────
//  Tunable thresholds
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float CONF_THRESHOLD = 0.25f;  // minimum confidence
static constexpr float NMS_IOU_THRESH = 0.45f;  // NMS overlap threshold

// ─────────────────────────────────────────────────────────────────────────────
//  Detection result  (normalised coordinates, [0,1] × [0,1])
// ─────────────────────────────────────────────────────────────────────────────
struct Detection {
    uint8_t class_id;
    float   score;
    float   x_min, y_min, x_max, y_max;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Unified entry point — auto-dispatches by output stream count
//
//  outputs.size() == 1  →  NMS built-in model (e.g. yolov8s_h8.hef)
//                          calls parse_nms_output(outputs[0])
//
//  outputs.size() == 6  →  Raw-tensor model   (e.g. yolov8n.hef)
//                          runs DFL decode + sigmoid + CPU NMS
//                          on the six NHWC float32 tensors
//
//  The correct model is selected by setting NORTHSTAR_MODEL_FILENAME in
//  config.hpp — no other code changes are needed.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> parse_detections(
    const std::vector<std::vector<uint8_t>>& outputs);

// Low-level parsers (exposed so tests can call them directly)
std::vector<Detection> parse_nms_output (const std::vector<uint8_t>& raw);
std::vector<Detection> parse_yolov8_raw (const std::vector<std::vector<uint8_t>>& outputs);

// Draw boxes + labels onto a BGR OpenCV Mat.
void draw_detections(cv::Mat& bgr_img, const std::vector<Detection>& dets);

// 80 COCO class labels.
extern const char* const COCO_CLASSES[80];
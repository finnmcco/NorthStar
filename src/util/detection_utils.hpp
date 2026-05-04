#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// ─────────────────────────────────────────────
//  Detection struct
// ─────────────────────────────────────────────
struct Detection {
    int   class_id;
    float score;
    float y_min, x_min, y_max, x_max;  // normalized 0.0–1.0
};

// ─────────────────────────────────────────────
//  COCO class labels (80 classes)
// ─────────────────────────────────────────────
extern const std::vector<std::string> COCO_CLASSES;

// ─────────────────────────────────────────────
//  NMS output parser
//
//  Parses the raw Hailo NMS output buffer into a
//  vector of Detection structs, filtering by
//  CONF_THRESHOLD and discarding invalid boxes.
// ─────────────────────────────────────────────
std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& buffer);

// ─────────────────────────────────────────────
//  Draw detections onto a BGR cv::Mat in-place.
//
//  Draws bounding boxes and class/confidence
//  labels for each detection.
// ─────────────────────────────────────────────
void draw_detections(cv::Mat& img, const std::vector<Detection>& dets);
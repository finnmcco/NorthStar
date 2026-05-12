#pragma once
#include <vector>
#include <opencv2/core.hpp>

#include "inference_packet.hpp"   // Detection, BoundingBox, InferencePacket
#include "inference_config.hpp"   // CONF_THRESHOLD, NMS_IOU_THRESH, N_CLASSES, REG_MAX


std::vector<Detection> parse_detections(
    const std::vector<std::vector<uint8_t>>& outputs);

// Low-level parsers exposed for direct use in tests.
std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& raw);
std::vector<Detection> parse_yolov8_raw(const std::vector<std::vector<uint8_t>>& outputs);

void draw_detections(cv::Mat& bgr_img, const std::vector<Detection>& dets);

extern const char* const COCO_CLASSES[80];
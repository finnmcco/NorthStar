#include "detection_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
//  COCO labels
// ─────────────────────────────────────────────────────────────────────────────
const char* const COCO_CLASSES[80] = {
    "person",        "bicycle",       "car",           "motorcycle",
    "airplane",      "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "couch",         "potted plant",  "bed",
    "dining table",  "toilet",        "tv",            "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"
};

// ─────────────────────────────────────────────────────────────────────────────
//  parse_nms_output  —  for models with Hailo NMS built-in (e.g. yolov8s_h8.hef)
//
//  HailoRT NMS buffer layout (HAILO_FORMAT_ORDER_HAILO_NMS):
//    for each class c in [0, 80):
//      float32  bbox_count
//      float32  y_min, x_min, y_max, x_max, score   × MAX_BBOXES
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& raw)
{
    std::vector<Detection> result;

    constexpr std::size_t N_CLASSES      = 80;
    constexpr std::size_t FLOATS_PER_BOX = 5;

    if (raw.size() % sizeof(float) != 0) return result;

    const std::size_t total_floats     = raw.size() / sizeof(float);
    const std::size_t floats_per_class = total_floats / N_CLASSES;
    if (floats_per_class < 1 + FLOATS_PER_BOX) return result;

    const std::size_t max_bboxes = (floats_per_class - 1) / FLOATS_PER_BOX;
    const auto* f = reinterpret_cast<const float*>(raw.data());

    for (std::size_t cls = 0; cls < N_CLASSES; ++cls) {
        const float* p     = f + cls * floats_per_class;
        const int    count = std::min(static_cast<int>(*p),
                                      static_cast<int>(max_bboxes));
        for (int b = 0; b < count; ++b) {
            const float* bbox  = p + 1 + b * FLOATS_PER_BOX;
            const float  score = bbox[4];
            if (score < CONF_THRESHOLD) continue;
            Detection d;
            d.class_id = static_cast<uint8_t>(cls);
            d.score    = score;
            d.y_min    = bbox[0]; d.x_min = bbox[1];
            d.y_max    = bbox[2]; d.x_max = bbox[3];
            result.push_back(d);
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  YOLOv8 raw-tensor post-processing
//  (for models without built-in NMS, e.g. yolov8n.hef)
//
//  Step 1 — DFL decode:  64 channels → 4 distances (l, t, r, b)
//  Step 2 — Sigmoid:     80 channels → 80 class probabilities
//  Step 3 — CPU NMS
// ─────────────────────────────────────────────────────────────────────────────

// Softmax + weighted sum of bins [0..REG_MAX-1]
static float dfl_decode(const float* vals, int reg_max = 16)
{
    float max_v = *std::max_element(vals, vals + reg_max);
    float sum   = 0.f;
    float result = 0.f;
    // Two-pass for numerical stability
    std::vector<float> s(reg_max);
    for (int i = 0; i < reg_max; ++i) { s[i] = std::exp(vals[i] - max_v); sum += s[i]; }
    for (int i = 0; i < reg_max; ++i) result += (s[i] / sum) * i;
    return result;
}

static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

static float iou(const Detection& a, const Detection& b)
{
    float ix1 = std::max(a.x_min, b.x_min), iy1 = std::max(a.y_min, b.y_min);
    float ix2 = std::min(a.x_max, b.x_max), iy2 = std::min(a.y_max, b.y_max);
    float iw  = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    float area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> nms_suppress(std::vector<Detection> dets)
{
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.score > b.score; });

    std::vector<bool>      suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].class_id == dets[j].class_id)
                if (iou(dets[i], dets[j]) > NMS_IOU_THRESH)
                    suppressed[j] = true;
        }
    }
    return result;
}

// Decode one (H×W) scale and append candidates to `out`.
// box_buf: float32 NHWC (H, W, 64)   — DFL box distributions
// cls_buf: float32 NHWC (H, W, 80)   — raw class logits
static void decode_scale(const float* box_buf, const float* cls_buf,
                          int H, int W, int stride,
                          std::vector<Detection>& out)
{
    constexpr int REG_MAX   = 16;
    constexpr int N_CLASSES = 80;
    const float   img_size  = static_cast<float>(H * stride); // e.g. 640

    for (int iy = 0; iy < H; ++iy) {
        for (int ix = 0; ix < W; ++ix) {
            // ── Class scores ──────────────────────────────────────────────────
            const float* cls = cls_buf + (iy * W + ix) * N_CLASSES;
            int   best_cls   = 0;
            float best_score = 0.f;
            for (int c = 0; c < N_CLASSES; ++c) {
                float s = sigmoid(cls[c]);
                if (s > best_score) { best_score = s; best_cls = c; }
            }
            if (best_score < CONF_THRESHOLD) continue;

            // ── Box decode (DFL) ──────────────────────────────────────────────
            const float* box = box_buf + (iy * W + ix) * (4 * REG_MAX);
            // Each group of REG_MAX channels is one directional distribution.
            // Multiply decoded distance by stride to get pixel distance.
            float l = dfl_decode(box + 0 * REG_MAX) * stride;
            float t = dfl_decode(box + 1 * REG_MAX) * stride;
            float r = dfl_decode(box + 2 * REG_MAX) * stride;
            float b = dfl_decode(box + 3 * REG_MAX) * stride;

            float cx = (ix + 0.5f) * stride;
            float cy = (iy + 0.5f) * stride;

            Detection d;
            d.class_id = static_cast<uint8_t>(best_cls);
            d.score    = best_score;
            d.x_min    = std::max(0.f, std::min(1.f, (cx - l) / img_size));
            d.y_min    = std::max(0.f, std::min(1.f, (cy - t) / img_size));
            d.x_max    = std::max(0.f, std::min(1.f, (cx + r) / img_size));
            d.y_max    = std::max(0.f, std::min(1.f, (cy + b) / img_size));
            out.push_back(d);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  parse_yolov8_raw
//
//  Identifies the 6 output tensors by their byte size (float32 NHWC):
//    1 638 400 bytes → (80,80,64)  stride-8  box
//    2 048 000 bytes → (80,80,80)  stride-8  class
//      409 600 bytes → (40,40,64)  stride-16 box
//      512 000 bytes → (40,40,80)  stride-16 class
//      102 400 bytes → (20,20,64)  stride-32 box
//      128 000 bytes → (20,20,80)  stride-32 class
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> parse_yolov8_raw(
    const std::vector<std::vector<uint8_t>>& outputs)
{
    const std::vector<uint8_t>* box8  = nullptr;
    const std::vector<uint8_t>* cls8  = nullptr;
    const std::vector<uint8_t>* box16 = nullptr;
    const std::vector<uint8_t>* cls16 = nullptr;
    const std::vector<uint8_t>* box32 = nullptr;
    const std::vector<uint8_t>* cls32 = nullptr;

    for (const auto& buf : outputs) {
        switch (buf.size()) {
            case 80u*80u*64u*4u: box8  = &buf; break;
            case 80u*80u*80u*4u: cls8  = &buf; break;
            case 40u*40u*64u*4u: box16 = &buf; break;
            case 40u*40u*80u*4u: cls16 = &buf; break;
            case 20u*20u*64u*4u: box32 = &buf; break;
            case 20u*20u*80u*4u: cls32 = &buf; break;
            default:
                std::cerr << "[detection_utils] unexpected output size: "
                          << buf.size() << " bytes\n";
        }
    }

    if (!box8 || !cls8 || !box16 || !cls16 || !box32 || !cls32) {
        std::cerr << "[detection_utils] could not identify all 6 YOLOv8 tensors\n";
        return {};
    }

    std::vector<Detection> candidates;
    candidates.reserve(512);

    decode_scale(reinterpret_cast<const float*>(box8->data()),
                 reinterpret_cast<const float*>(cls8->data()),
                 80, 80, 8, candidates);

    decode_scale(reinterpret_cast<const float*>(box16->data()),
                 reinterpret_cast<const float*>(cls16->data()),
                 40, 40, 16, candidates);

    decode_scale(reinterpret_cast<const float*>(box32->data()),
                 reinterpret_cast<const float*>(cls32->data()),
                 20, 20, 32, candidates);

    return nms_suppress(std::move(candidates));
}

// ─────────────────────────────────────────────────────────────────────────────
//  parse_detections  —  unified entry point, auto-dispatches by stream count
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> parse_detections(
    const std::vector<std::vector<uint8_t>>& outputs)
{
    if (outputs.size() == 1)
        return parse_nms_output(outputs[0]);   // built-in NMS model
    else
        return parse_yolov8_raw(outputs);       // raw-tensor model
}

// ─────────────────────────────────────────────────────────────────────────────
//  draw_detections
// ─────────────────────────────────────────────────────────────────────────────
void draw_detections(cv::Mat& img, const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        const int x1 = static_cast<int>(d.x_min * img.cols);
        const int y1 = static_cast<int>(d.y_min * img.rows);
        const int x2 = static_cast<int>(d.x_max * img.cols);
        const int y2 = static_cast<int>(d.y_max * img.rows);

        const cv::Scalar colour(0, 255, 0);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colour, 2);

        const std::string label =
            std::string(COCO_CLASSES[d.class_id]) + " " +
            std::to_string(static_cast<int>(d.score * 100)) + "%";

        int baseline = 0;
        auto ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::rectangle(img,
                      cv::Point(x1, std::max(y1 - ts.height - 4, 0)),
                      cv::Point(x1 + ts.width, std::max(y1, ts.height + 4)),
                      colour, cv::FILLED);
        cv::putText(img, label,
                    cv::Point(x1, std::max(y1 - 3, ts.height)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}
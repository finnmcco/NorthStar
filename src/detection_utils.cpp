#include "detection_utils.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
//  COCO label table
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
//  parse_nms_output
// ─────────────────────────────────────────────────────────────────────────────
/*
    HailoRT NMS buffer layout for HAILO_FORMAT_ORDER_HAILO_NMS
    ─────────────────────────────────────────────────────────────
    Repeated for each of the N_CLASSES (80 for COCO YOLOv8):

        float32  bbox_count          ← number of valid boxes (0 .. MAX_BBOXES)
        float32  y_min  ┐
        float32  x_min  │  repeated MAX_BBOXES times
        float32  y_max  │  (only the first bbox_count entries are valid)
        float32  x_max  │
        float32  score  ┘

    MAX_BBOXES is not a compile-time constant here; it is inferred from the
    buffer size so the parser works with any model variant.
*/
std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& raw)
{
    std::vector<Detection> result;

    constexpr std::size_t N_CLASSES     = 80;
    constexpr std::size_t FLOATS_PER_BOX = 5; // y_min x_min y_max x_max score

    if (raw.size() % sizeof(float) != 0) {
        std::cerr << "[detection_utils] buffer not float-aligned: " << raw.size() << " bytes\n";
        return result;
    }

    const std::size_t total_floats     = raw.size() / sizeof(float);
    const std::size_t floats_per_class = total_floats / N_CLASSES;

    // floats_per_class = 1 (count) + MAX_BBOXES * 5
    if (floats_per_class < 1 + FLOATS_PER_BOX) {
        std::cerr << "[detection_utils] buffer too small\n";
        return result;
    }

    const std::size_t max_bboxes = (floats_per_class - 1) / FLOATS_PER_BOX;
    const auto* f = reinterpret_cast<const float*>(raw.data());

    for (std::size_t cls = 0; cls < N_CLASSES; ++cls)
    {
        const float* p     = f + cls * floats_per_class;
        const int    count = static_cast<int>(*p);
        if (count <= 0) continue;

        const int n = std::min(count, static_cast<int>(max_bboxes));

        for (int b = 0; b < n; ++b)
        {
            const float* bbox  = p + 1 + b * FLOATS_PER_BOX;
            const float  y_min = bbox[0];
            const float  x_min = bbox[1];
            const float  y_max = bbox[2];
            const float  x_max = bbox[3];
            const float  score = bbox[4];

            if (score < CONF_THRESHOLD) continue;

            Detection d;
            d.class_id = static_cast<uint8_t>(cls);
            d.score    = score;
            d.x_min    = x_min;
            d.y_min    = y_min;
            d.x_max    = x_max;
            d.y_max    = y_max;
            result.push_back(d);
        }
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  draw_detections
// ─────────────────────────────────────────────────────────────────────────────
void draw_detections(cv::Mat& img, const std::vector<Detection>& dets)
{
    for (const auto& d : dets)
    {
        const int x1 = static_cast<int>(d.x_min * img.cols);
        const int y1 = static_cast<int>(d.y_min * img.rows);
        const int x2 = static_cast<int>(d.x_max * img.cols);
        const int y2 = static_cast<int>(d.y_max * img.rows);

        const cv::Scalar colour(0, 255, 0); // green

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colour, 2);

        const std::string label =
            std::string(COCO_CLASSES[d.class_id]) + " " +
            std::to_string(static_cast<int>(d.score * 100)) + "%";

        // Background rectangle for readability
        int baseline = 0;
        const auto text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);

        const cv::Point tl(x1, std::max(y1 - text_size.height - 4, 0));
        const cv::Point br(x1 + text_size.width, std::max(y1, text_size.height + 4));
        cv::rectangle(img, tl, br, colour, cv::FILLED);

        cv::putText(img, label,
                    cv::Point(x1, std::max(y1 - 3, text_size.height)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

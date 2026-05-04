#include "detection_utils.hpp"

#include <algorithm>  // std::clamp
#include <cstring>    // std::memcpy

#include <hailo/hailort.h>

#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
static constexpr int   NUM_CLASSES              = 80;
static constexpr int   MAX_DETECTIONS_PER_CLASS = 100;
static constexpr float CONF_THRESHOLD           = 0.05f;

// ─────────────────────────────────────────────
//  COCO class labels
// ─────────────────────────────────────────────
const std::vector<std::string> COCO_CLASSES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
};

// ─────────────────────────────────────────────
//  NMS output parser
// ─────────────────────────────────────────────
std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& buffer)
{
    std::vector<Detection> detections;

    constexpr int BBOX_SIZE = sizeof(hailo_bbox_float32_t);

    const uint8_t* ptr = buffer.data();
    const uint8_t* end = ptr + buffer.size();

    for (int cls = 0; cls < NUM_CLASSES; ++cls)
    {
        if (ptr + sizeof(uint32_t) > end) break;

        // Hailo NMS buffer stores count as float32, not uint32
        float count_f = 0.f;
        std::memcpy(&count_f, ptr, sizeof(float));
        ptr += sizeof(float);

        int count = static_cast<int>(count_f);
        if (count < 0 || count > MAX_DETECTIONS_PER_CLASS)
            count = 0;

        for (int i = 0; i < MAX_DETECTIONS_PER_CLASS; ++i)
        {
            if (ptr + BBOX_SIZE > end) break;

            hailo_bbox_float32_t bbox{};
            std::memcpy(&bbox, ptr, BBOX_SIZE);
            ptr += BBOX_SIZE;

            if (i >= (int)count)             continue;
            if (bbox.score < CONF_THRESHOLD) continue;
            if (bbox.score >= 1.0f)          continue;
            if (bbox.x_min > 1.0f || bbox.y_min > 1.0f ||
                bbox.x_max > 1.0f || bbox.y_max > 1.0f) continue;

            detections.push_back({
                cls,
                bbox.score,
                bbox.y_min, bbox.x_min,
                bbox.y_max, bbox.x_max
            });
        }
    }

    return detections;
}

// ─────────────────────────────────────────────
//  Draw detections onto a BGR cv::Mat
// ─────────────────────────────────────────────
void draw_detections(cv::Mat& img, const std::vector<Detection>& dets)
{
    const int    W          = img.cols;
    const int    H          = img.rows;
    const int    font       = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.5;
    const int    thickness  = 2;

    for (const auto& d : dets)
    {
        int x1 = std::clamp((int)(d.x_min * W), 0, W - 1);
        int y1 = std::clamp((int)(d.y_min * H), 0, H - 1);
        int x2 = std::clamp((int)(d.x_max * W), 0, W - 1);
        int y2 = std::clamp((int)(d.y_max * H), 0, H - 1);

        cv::rectangle(img, {x1, y1}, {x2, y2},
                      cv::Scalar(0, 255, 0), thickness);

        char label[64];
        std::snprintf(label, sizeof(label), "%s %.0f%%",
                      COCO_CLASSES[d.class_id].c_str(),
                      d.score * 100.f);

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, font, font_scale,
                                      thickness, &baseline);
        int ly = std::max(y1, ts.height + 4);
        cv::rectangle(img,
                      {x1, ly - ts.height - 4},
                      {x1 + ts.width, ly},
                      cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(img, label, {x1, ly - 2},
                    font, font_scale,
                    cv::Scalar(0, 0, 0), thickness - 1);
    }
}
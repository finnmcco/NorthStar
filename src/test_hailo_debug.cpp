#include "inference/hailo8_inference.hpp"

#include <hailo/hailort.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

static const std::vector<std::string> COCO_CLASSES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static constexpr int   NUM_CLASSES              = 80;
static constexpr int   MAX_DETECTIONS_PER_CLASS = 100;
static constexpr float CONF_THRESHOLD           = 0.3f;

struct Detection {
    int   class_id;
    float score;
    float y_min, x_min, y_max, x_max;  // normalized 0.0-1.0
};

std::vector<Detection> parse_nms_output(const std::vector<uint8_t>& buffer)
{
    std::vector<Detection> detections;

    constexpr int BBOX_SIZE  = sizeof(hailo_bbox_float32_t);
    constexpr int CLASS_SIZE = sizeof(uint32_t) + MAX_DETECTIONS_PER_CLASS * BBOX_SIZE;

    std::cout << "Output buffer size: " << buffer.size() << " bytes\n";
    std::cout << "Expected size:      " << (NUM_CLASSES * CLASS_SIZE) << " bytes\n";

    const uint8_t* ptr = buffer.data();
    const uint8_t* end = ptr + buffer.size();

    for (int cls = 0; cls < NUM_CLASSES; ++cls)
    {
        if (ptr + sizeof(uint32_t) > end) break;

        uint32_t count = 0;
        std::memcpy(&count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        if (count > MAX_DETECTIONS_PER_CLASS) count = MAX_DETECTIONS_PER_CLASS;

        for (int i = 0; i < MAX_DETECTIONS_PER_CLASS; ++i)
        {
            if (ptr + BBOX_SIZE > end) break;

            hailo_bbox_float32_t bbox{};
            std::memcpy(&bbox, ptr, BBOX_SIZE);
            ptr += BBOX_SIZE;


            if (i >= (int)count)             continue;
            if (bbox.score < CONF_THRESHOLD) continue;
            if (bbox.score > 1.0f)           continue;  // filter garbage
            if (bbox.x_min > 1.0f || bbox.y_min > 1.0f || 
                bbox.x_max > 1.0f || bbox.y_max > 1.0f) continue;  // filter out-of-bounds

            detections.push_back({
                cls,
                bbox.score,
                bbox.y_min, bbox.x_min,
                bbox.y_max, bbox.x_max
            });
        }
    }

    std::cout << "Bytes consumed: " << (ptr - buffer.data())
              << " / " << buffer.size() << "\n";

    

    return detections;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections)
{
    const int    img_w      = img.cols;
    const int    img_h      = img.rows;
    const int    font_face  = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.5;
    const int    thickness  = 2;

    for (const auto& d : detections)
    {
        int x1 = (int)(d.x_min * img_w);
        int y1 = (int)(d.y_min * img_h);
        int x2 = (int)(d.x_max * img_w);
        int y2 = (int)(d.y_max * img_h);

        // Clamp to image bounds
        x1 = std::max(0, std::min(x1, img_w - 1));
        y1 = std::max(0, std::min(y1, img_h - 1));
        x2 = std::max(0, std::min(x2, img_w - 1));
        y2 = std::max(0, std::min(y2, img_h - 1));

        // Draw bounding box
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), thickness);

        // Build label
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.0f%%",
                      COCO_CLASSES[d.class_id].c_str(),
                      d.score * 100.0f);

        // Background rectangle for label
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, font_face, font_scale, thickness, &baseline);

        int label_y = std::max(y1, text_size.height + 4);
        cv::rectangle(img,
                      cv::Point(x1, label_y - text_size.height - 4),
                      cv::Point(x1 + text_size.width, label_y),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        // Draw label text
        cv::putText(img, label,
                    cv::Point(x1, label_y - 2),
                    font_face, font_scale,
                    cv::Scalar(0, 0, 0),  // black text
                    thickness - 1);

        std::cout << "[" << COCO_CLASSES[d.class_id] << "] "
                  << "conf=" << d.score
                  << "  box=(" << x1 << "," << y1
                  << ")-(" << x2 << "," << y2 << ")\n";
    }
}

int main()
{
    std::string hef_path = "/usr/share/hailo-models/yolov8s_h8.hef";

    Hailo8Inference hailo(hef_path);
    if (!hailo.initialize()) {
        std::cerr << "Failed to initialize Hailo\n";
        return 1;
    }

    cv::Mat img = cv::imread("frame7.png");
    if (img.empty()) {
        std::cerr << "Failed to load frame0.png\n";
        return 1;
    }

    cv::resize(img, img, cv::Size(640, 640));

    // Inference needs RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    if (!hailo.run(rgb.data)) {
        std::cerr << "Inference failed\n";
        return 1;
    }

    const auto& output = hailo.get_output();
    auto detections = parse_nms_output(output);

    std::cout << "\n=== Detections (conf >= " << CONF_THRESHOLD << ") ===\n";

    if (detections.empty()) {
        std::cout << "No detections.\n";
    }

    // Draw on the original BGR image (not the RGB copy)
    draw_detections(img, detections);

    // Save result
    cv::imwrite("result.png", img);
    std::cout << "\nSaved result.png\n";

    return 0;
}
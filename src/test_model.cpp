#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#include <hailo/hailort.h>
#include <opencv2/opencv.hpp>

#include "inference/hailo8_inference.hpp"

// ─────────────────────────────────────────────
//  COCO labels
// ─────────────────────────────────────────────
static const std::vector<std::string> COCO_CLASSES = {
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

static constexpr int   NUM_CLASSES              = 80;
static constexpr int   MAX_DETECTIONS_PER_CLASS = 100;
static constexpr float CONF_THRESHOLD           = 0.05f;

// ─────────────────────────────────────────────
//  Raw NMS dump — shows exactly what Hailo returns
//  before any parsing/filtering
// ─────────────────────────────────────────────
void raw_nms_dump(const std::vector<uint8_t>& buffer)
{
    const uint8_t* ptr = buffer.data();
    const uint8_t* end = ptr + buffer.size();

    std::cout << "\n=== RAW NMS DUMP ===\n";
    bool any = false;

    for (int cls = 0; cls < NUM_CLASSES; ++cls)
    {
        if (ptr + sizeof(uint32_t) > end) break;

        uint32_t count = 0;
        std::memcpy(&count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        if (count > 0 && count <= 100) {
            any = true;
            std::cout << "Class " << cls << " [" << COCO_CLASSES[cls]
                      << "]: " << count << " detections\n";
            for (uint32_t i = 0; i < count; ++i) {
                hailo_bbox_float32_t bbox{};
                std::memcpy(&bbox, ptr, sizeof(hailo_bbox_float32_t));
                std::cout << "  [" << i << "] score=" << bbox.score
                          << " x=(" << bbox.x_min << "-" << bbox.x_max << ")"
                          << " y=(" << bbox.y_min << "-" << bbox.y_max << ")\n";
                ptr += sizeof(hailo_bbox_float32_t);
            }
        } else {
            ptr += MAX_DETECTIONS_PER_CLASS * sizeof(hailo_bbox_float32_t);
        }
    }

    if (!any)
        std::cout << "  (no detections in any class)\n";

    std::cout << "====================\n\n";
}

// ─────────────────────────────────────────────
//  Run inference on a single image file
// ─────────────────────────────────────────────
void test_image(Hailo8Inference& hailo, const std::string& path)
{
    std::cout << "\n--- Testing: " << path << " ---\n";

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cout << "  ERROR: could not load image\n";
        return;
    }

    std::cout << "  Original size: " << img.cols << "x" << img.rows << "\n";

    // Resize to 640x640
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));

    // Convert BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Save what we're sending to Hailo for visual inspection
    std::string debug_path = path + "_hailo_input.png";
    cv::imwrite(debug_path, resized);  // save as BGR (correct for imwrite)
    std::cout << "  Saved input to: " << debug_path << "\n";

    // Run inference
    if (!hailo.run(rgb.data)) {
        std::cout << "  ERROR: inference failed\n";
        return;
    }

    // Raw dump
    raw_nms_dump(hailo.get_output());
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // Allow HEF path as optional argument, default to yolov8s
    std::string hef_path = "/usr/share/hailo-models/yolov8s_h8.hef";
    if (argc > 1)
        hef_path = argv[1];

    std::cout << "Loading model: " << hef_path << "\n";

    Hailo8Inference hailo(hef_path);
    if (!hailo.initialize()) {
        std::cerr << "Failed to initialize Hailo\n";
        return 1;
    }

    // Test images — pass as arguments or use defaults
    std::vector<std::string> images;
    for (int i = 2; i < argc; ++i)
        images.push_back(argv[i]);

    // Default test images if none provided
    if (images.empty()) {
        images = {
            "/tmp/dog.jpg",
            "/tmp/bus.jpg",
            "/tmp/cat.jpg",
        };
    }

    for (const auto& img_path : images)
        test_image(hailo, img_path);

    std::cout << "\nDone.\n";
    return 0;
}

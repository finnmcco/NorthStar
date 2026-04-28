#include "inference/hailo8_inference.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

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

void parse_and_draw(const std::vector<uint8_t>& output,
                    cv::Mat& image,
                    float score_threshold = 0.5f)
{
    const float* data =
        reinterpret_cast<const float*>(output.data());

    const size_t total_floats = output.size() / sizeof(float);

    const int num_classes = 80;
    const int max_boxes_per_class = 100;
    const int floats_per_class = 1 + (max_boxes_per_class * 5);

    if (total_floats != num_classes * floats_per_class)
    {
        std::cerr << "Unexpected output size: "
                  << total_floats << " floats\n";
        return;
    }

    int img_w = image.cols;
    int img_h = image.rows;

    for (int class_id = 0; class_id < num_classes; class_id++)
    {
        int offset = class_id * floats_per_class;

        int num_detections =
            static_cast<int>(data[offset]);

        if (num_detections <= 0 || num_detections > 100)
            continue;

        for (int i = 0; i < num_detections; i++)
        {
            int base = offset + 1 + (i * 5);

            float x1    = data[base + 0];
            float y1    = data[base + 1];
            float x2    = data[base + 2];
            float y2    = data[base + 3];
            float score = data[base + 4];

            if (score < score_threshold)
                continue;

            int px1 = static_cast<int>(x1 * img_w);
            int py1 = static_cast<int>(y1 * img_h);
            int px2 = static_cast<int>(x2 * img_w);
            int py2 = static_cast<int>(y2 * img_h);

            if (px1 > px2) std::swap(px1, px2);
            if (py1 > py2) std::swap(py1, py2);

            px1 = std::clamp(px1, 0, img_w - 1);
            px2 = std::clamp(px2, 0, img_w - 1);
            py1 = std::clamp(py1, 0, img_h - 1);
            py2 = std::clamp(py2, 0, img_h - 1);

            cv::rectangle(image,
                          cv::Point(px1, py1),
                          cv::Point(px2, py2),
                          cv::Scalar(0, 255, 0),
                          2);

            std::string label =
                COCO_CLASSES[class_id] +
                " " + cv::format("%.2f", score);

            cv::putText(image,
                        label,
                        cv::Point(px1, std::max(py1 - 5, 0)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1);
        }
    }
}

int main()
{
    std::string hef_path =
        "/usr/share/hailo-models/yolov8s_h8.hef";

    Hailo8Inference hailo(hef_path);

    if (!hailo.initialize())
    {
        std::cerr << "Failed to initialize Hailo\n";
        return 1;
    }

    for (int i = 0; i <= 93; i++)
    {
        std::string input_name =
            "frame" + std::to_string(i) + ".png";

        std::string output_name =
            "results" + std::to_string(i) + ".png";

        cv::Mat img = cv::imread(input_name);

        if (img.empty())
        {
            std::cerr << "Failed to load "
                      << input_name << "\n";
            continue;
        }

        cv::resize(img, img, cv::Size(640, 640));

        // Important: do NOT convert to RGB for this HEF
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        if (!hailo.run(img.data))
        {
            std::cerr << "Inference failed for "
                      << input_name << "\n";
            continue;
        }

        const auto& output = hailo.get_output();

        parse_and_draw(output, img, 0.5f);

        cv::imwrite(output_name, img);

        std::cout << "Saved " << output_name << "\n";
    }

    std::cout << "All frames processed.\n";

    return 0;
}
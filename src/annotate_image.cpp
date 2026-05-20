/*
    annotate_image.cpp  --  run inference on an image or directory and save annotated results

    Feeds 640x640 BGR PNG(s) through the Hailo-8, draws bounding boxes
    and confidence scores for specified COCO object classes, and saves
    each result with an _annotated suffix.

    --flip flips the channel order of the SAVED annotated image only.
    Inference always receives the raw BGR input unchanged.

    Usage:
        ./annotate_image <image.png|directory> [--flip] [word1 word2 ...]

    Examples:
        ./annotate_image ../inf_front
        ./annotate_image ../inf_front --flip phone knife cup
        ./annotate_image ../inf_front/frame.png cup
*/

#include "config.hpp"
#include "colour.hpp"
#include "coco_lookup.hpp"
#include "detection_utils.hpp"
#include "hailo8_inference.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

static void process_image(const std::string&          path,
                          bool                        flip_output,
                          const std::vector<uint8_t>& target_ids,
                          Hailo8Inference&             hailo)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "  SKIP %s (cannot read)\n", path.c_str());
        return;
    }
    if (img.cols != 640 || img.rows != 640)
        cv::resize(img, img, {640, 640});

    // Inference always on raw BGR
    std::vector<uint8_t> data(img.data,
                              img.data + img.total() * img.elemSize());
    hailo_prepare(data);

    std::mutex              mtx;
    std::condition_variable cv;
    std::vector<Detection>  detections;
    bool                    ready = false;

    hailo.register_callback(
        [&](uint8_t, uint64_t, std::vector<std::vector<uint8_t>> raw)
        {
            auto dets = parse_detections(raw);
            {
                std::lock_guard<std::mutex> lk(mtx);
                detections = std::move(dets);
                ready      = true;
            }
            cv.notify_one();
        });

    hailo.write_frame(data.data(), 0, 0);
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&] { return ready; });
    }

    // Filter to target classes if specified
    if (!target_ids.empty()) {
        detections.erase(
            std::remove_if(detections.begin(), detections.end(),
                [&](const Detection& d) {
                    return std::find(target_ids.begin(), target_ids.end(),
                                     d.object_id) == target_ids.end();
                }),
            detections.end());
    }

    // Build output path
    const std::filesystem::path p(path);
    const std::string output_path =
        (p.parent_path() / (p.stem().string() + "_annotated.png")).string();

    // Draw bounding boxes on a copy of the raw image
    cv::Mat annotated = img.clone();
    draw_detections(annotated, detections);

    // Optionally flip channels of the saved output only
    if (flip_output)
        cv::cvtColor(annotated, annotated, cv::COLOR_BGR2RGB);

    cv::imwrite(output_path, annotated);

    std::printf("  %-45s  %zu detection(s)\n",
                p.filename().string().c_str(),
                detections.size());
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <image.png|directory> [--flip] [word1 word2 ...]\n"
            "  e.g. %s ../inf_front --flip phone knife cup\n",
            argv[0], argv[0]);
        return 1;
    }

    const std::string input = argv[1];
    bool flip_output = false;
    std::vector<uint8_t> target_ids;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--flip") {
            flip_output = true;
            continue;
        }
        uint8_t id = coco_id_for_word(argv[i]);
        if (id == 255)
            std::fprintf(stderr, "WARNING: \"%s\" is not a COCO class\n",
                         argv[i]);
        else {
            std::printf("  Filtering for: %s (id=%d)\n", argv[i], id);
            target_ids.push_back(id);
        }
    }

    // Collect input files
    std::vector<std::string> files;
    if (std::filesystem::is_directory(input)) {
        for (const auto& e : std::filesystem::directory_iterator(input)) {
            const auto& p = e.path();
            if (p.extension() == ".png" &&
                p.stem().string().find("_annotated") == std::string::npos)
                files.push_back(p.string());
        }
        std::sort(files.begin(), files.end());
        std::printf("\nFound %zu PNG(s) in %s\n\n",
                    files.size(), input.c_str());
    } else {
        files.push_back(input);
    }

    if (files.empty()) {
        std::printf("No PNG files found.\n");
        return 0;
    }

    Hailo8Inference hailo(DEFAULT_HEF_PATH);
    if (!hailo.initialize()) {
        std::fprintf(stderr, "FATAL: Hailo initialise failed\n");
        return 1;
    }
    std::printf("[Hailo] ready\n\n");

    for (const auto& path : files)
        process_image(path, flip_output, target_ids, hailo);

    hailo.stop();
    std::printf("\nDone. %zu image(s) annotated.\n", files.size());
    return 0;
}
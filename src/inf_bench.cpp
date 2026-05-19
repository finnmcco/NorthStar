/*
    inf_bench.cpp  --  inference latency and detection benchmark

    Scans inf_front/ and inf_back/ for PNG images, feeds each through the
    Hailo-8, records latency and detections for cell phone, knife, and cup,
    and saves per-image results to a CSV in each folder.

    Usage:
        ./inf_bench           -- images are already BGR
        ./inf_bench --flip    -- flip BGR->RGB before inference

    Output:
        inf_front/results.csv
        inf_back/results.csv

    CSV columns:
        filename, latency_ms,
        phone_found, phone_conf,
        knife_found, knife_conf,
        cup_found,   cup_conf
*/

#include "config.hpp"
#include "colour.hpp"
#include "detection_utils.hpp"
#include "hailo8_inference.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Target COCO class IDs
static constexpr uint8_t ID_CUP   = 41;
static constexpr uint8_t ID_KNIFE = 43;
static constexpr uint8_t ID_PHONE = 67;

struct FrameResult {
    std::string filename;
    double      latency_ms  = 0.0;
    bool        phone_found = false;
    float       phone_conf  = 0.f;
    bool        knife_found = false;
    float       knife_conf  = 0.f;
    bool        cup_found   = false;
    float       cup_conf    = 0.f;
};

// Find highest confidence detection for a given class
static float best_conf(const std::vector<Detection>& dets, uint8_t id) {
    float best = 0.f;
    for (const auto& d : dets)
        if (d.object_id == id && d.confidence > best)
            best = d.confidence;
    return best;
}

static std::vector<std::string> collect_pngs(const std::string& dir) {
    std::vector<std::string> files;
    if (!std::filesystem::exists(dir)) return files;
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".png")
            files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    return files;
}

static void save_csv(const std::string& path,
                     const std::vector<FrameResult>& results)
{
    std::ofstream f(path);
    f << "filename,latency_ms,"
      << "phone_found,phone_conf,"
      << "knife_found,knife_conf,"
      << "cup_found,cup_conf\n";
    for (const auto& r : results) {
        f << std::filesystem::path(r.filename).filename().string() << ","
          << r.latency_ms << ","
          << (r.phone_found ? 1 : 0) << "," << r.phone_conf << ","
          << (r.knife_found ? 1 : 0) << "," << r.knife_conf << ","
          << (r.cup_found   ? 1 : 0) << "," << r.cup_conf   << "\n";
    }
    std::printf("  Saved: %s\n", path.c_str());
}

static void print_summary(const std::string& folder,
                          const std::vector<FrameResult>& results)
{
    if (results.empty()) return;

    double total_ms = 0;
    int phone = 0, knife = 0, cup = 0;
    for (const auto& r : results) {
        total_ms += r.latency_ms;
        if (r.phone_found) ++phone;
        if (r.knife_found) ++knife;
        if (r.cup_found)   ++cup;
    }
    int n = static_cast<int>(results.size());

    std::printf("\n  %-12s  %d images\n", folder.c_str(), n);
    std::printf("  Avg latency:  %.2f ms\n", total_ms / n);
    std::printf("  Phone found:  %d/%d  (%.0f%%)\n",
                phone, n, 100.0*phone/n);
    std::printf("  Knife found:  %d/%d  (%.0f%%)\n",
                knife, n, 100.0*knife/n);
    std::printf("  Cup found:    %d/%d  (%.0f%%)\n",
                cup,   n, 100.0*cup/n);
}

// Process one folder — returns results
static std::vector<FrameResult> process_folder(
    const std::string& folder,
    Hailo8Inference&   hailo,
    bool               flip_bgr)
{
    auto files = collect_pngs(folder);
    if (files.empty()) {
        std::printf("  [%s] no PNG files found\n", folder.c_str());
        return {};
    }
    std::printf("\n[%s]  %zu images  flip=%s\n",
                folder.c_str(), files.size(), flip_bgr ? "yes" : "no");

    std::vector<FrameResult> results;
    results.reserve(files.size());

    // Synchronisation between main thread (write) and callback (read)
    std::mutex              mtx;
    std::condition_variable cv;
    std::vector<Detection>  cb_detections;
    bool                    cb_ready = false;

    hailo.register_callback(
        [&](uint8_t, uint64_t, std::vector<std::vector<uint8_t>> raw)
        {
            auto dets = parse_detections(raw);
            {
                std::lock_guard<std::mutex> lk(mtx);
                cb_detections = std::move(dets);
                cb_ready      = true;
            }
            cv.notify_one();
        });

    for (const auto& path : files) {
        std::string fname = std::filesystem::path(path).filename().string();

        // Load image
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::printf("  SKIP %s (cannot read)\n", fname.c_str());
            continue;
        }

        // Resize to 640x640 if needed
        if (img.cols != 640 || img.rows != 640)
            cv::resize(img, img, {640, 640});

        // Optional channel flip
        if (flip_bgr)
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Prepare for Hailo (hailo_prepare may be a no-op depending on config)
        std::vector<uint8_t> data(img.data, img.data + img.total() * img.elemSize());
        hailo_prepare(data);

        // Reset callback state
        {
            std::lock_guard<std::mutex> lk(mtx);
            cb_ready = false;
            cb_detections.clear();
        }

        // Time the inference
        const auto t0 = std::chrono::steady_clock::now();
        hailo.write_frame(data.data(), 0, 0);

        // Wait for result
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return cb_ready; });
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        FrameResult r;
        r.filename   = path;
        r.latency_ms = ms;
        r.phone_conf = best_conf(cb_detections, ID_PHONE);
        r.knife_conf = best_conf(cb_detections, ID_KNIFE);
        r.cup_conf   = best_conf(cb_detections, ID_CUP);
        r.phone_found = r.phone_conf > 0.f;
        r.knife_found = r.knife_conf > 0.f;
        r.cup_found   = r.cup_conf   > 0.f;

        std::printf("  %-40s  %6.2f ms  phone=%s(%.2f)  knife=%s(%.2f)  cup=%s(%.2f)\n",
                    fname.c_str(), ms,
                    r.phone_found ? "Y" : "N", r.phone_conf,
                    r.knife_found ? "Y" : "N", r.knife_conf,
                    r.cup_found   ? "Y" : "N", r.cup_conf);

        results.push_back(r);
    }

    std::filesystem::create_directories("results");
    const std::string model_name = std::filesystem::path(DEFAULT_HEF_PATH).stem().string();
    const std::string flip_tag   = flip_bgr ? "_flip" : "";
    save_csv("results/" + std::filesystem::path(folder).filename().string()
            + "_" + model_name + flip_tag + "_results.csv", results);
    
    return results;
}

int main(int argc, char* argv[])
{
    bool flip_bgr = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--flip") flip_bgr = true;

    std::printf("inf_bench  |  model: %s  |  flip: %s\n\n",
                DEFAULT_HEF_PATH, flip_bgr ? "BGR->RGB" : "none");

    Hailo8Inference hailo(DEFAULT_HEF_PATH);
    if (!hailo.initialize()) {
        std::fprintf(stderr, "FATAL: Hailo initialise failed\n");
        return 1;
    }
    std::printf("[Hailo] ready  input=%zu bytes\n", hailo.input_frame_size());

    auto front_results = process_folder("inf_front", hailo, flip_bgr);
    auto back_results  = process_folder("inf_back",  hailo, flip_bgr);

    std::printf("\n════════════════════════════════════════════\n");
    std::printf("  SUMMARY\n");
    std::printf("════════════════════════════════════════════\n");
    print_summary("inf_front", front_results);
    print_summary("inf_back",  back_results);

    hailo.stop();
    return 0;
}

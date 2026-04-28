#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <atomic>
#include <csignal>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <thread>

#include <hailo/hailort.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>

#include "capture/camera_capture.hpp"
#include "capture/frame_packet.hpp"
#include "util/queue.hpp"
#include "util/buffer_pool.hpp"
#include "inference/hailo8_inference.hpp"

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
static constexpr int   NUM_CLASSES              = 80;
static constexpr int   MAX_DETECTIONS_PER_CLASS = 100;
static constexpr float CONF_THRESHOLD           = 0.4f;
static constexpr int   WARMUP_FRAMES            = 30;
static constexpr int   REPORT_INTERVAL          = 100;

// ─────────────────────────────────────────────
//  Shutdown flag
// ─────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// ─────────────────────────────────────────────
//  InferPacket — owns its RGB buffer so the
//  camera pool buffer can be released immediately
//  after the BGR->RGB conversion
// ─────────────────────────────────────────────
struct InferPacket {
    std::vector<uint8_t> rgb;           // 640*640*3, RGB
    uint64_t             cam_ts;        // ns — when frame left camera
    uint64_t             preprocess_ns; // ns — BGR->RGB duration
};

// ─────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────
struct Detection {
    int   class_id;
    float score;
    float y_min, x_min, y_max, x_max;
};

// ─────────────────────────────────────────────
//  NMS parser
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
        uint32_t count = 0;
        std::memcpy(&count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        if (count > (uint32_t)MAX_DETECTIONS_PER_CLASS)
            count = MAX_DETECTIONS_PER_CLASS;

        for (int i = 0; i < MAX_DETECTIONS_PER_CLASS; ++i)
        {
            if (ptr + BBOX_SIZE > end) break;
            hailo_bbox_float32_t bbox{};
            std::memcpy(&bbox, ptr, BBOX_SIZE);
            ptr += BBOX_SIZE;
            if (i >= (int)count)             continue;
            if (bbox.score < CONF_THRESHOLD) continue;
            detections.push_back({ cls, bbox.score,
                bbox.y_min, bbox.x_min, bbox.y_max, bbox.x_max });
        }
    }
    return detections;
}

// ─────────────────────────────────────────────
//  Timing helpers
// ─────────────────────────────────────────────
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Ms        = std::chrono::duration<double, std::milli>;

static uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

static double to_ms(uint64_t ns) { return ns / 1.0e6; }

// ─────────────────────────────────────────────
//  Stats printer
// ─────────────────────────────────────────────
static void print_stats(
    int frame_count,
    const std::vector<double>& preprocess_ms,
    const std::vector<double>& inference_ms,
    const std::vector<double>& parse_ms,
    const std::vector<double>& e2e_ms,
    double elapsed_wall_s)
{
    auto stats = [](const std::vector<double>& v, const std::string& name) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double mn   = *std::min_element(v.begin(), v.end());
        double mx   = *std::max_element(v.begin(), v.end());
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        double p50 = s[s.size() * 50  / 100];
        double p95 = s[s.size() * 95  / 100];
        double p99 = s[s.size() * 99  / 100];
        std::cout << "  " << name << ":\n"
                  << "    mean=" << mean << "ms"
                  << "  min="   << mn   << "ms"
                  << "  max="   << mx   << "ms"
                  << "  p50="   << p50  << "ms"
                  << "  p95="   << p95  << "ms"
                  << "  p99="   << p99  << "ms\n";
    };

    std::cout << "\n══════════════════════════════════════════\n"
              << "  BENCHMARK — " << frame_count << " frames\n"
              << "  Wall time:  " << elapsed_wall_s << "s\n"
              << "  Throughput: " << (frame_count / elapsed_wall_s) << " fps\n"
              << "──────────────────────────────────────────\n";
    stats(preprocess_ms, "BGR->RGB cvtColor ");
    stats(inference_ms,  "Hailo inference   ");
    stats(parse_ms,      "NMS parse         ");
    stats(e2e_ms,        "End-to-end (cam)  ");
    std::cout << "══════════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main()
{
    std::signal(SIGINT, on_sigint);

    // ── Hailo ───────────────────────────────
    const std::string hef_path = "/usr/share/hailo-models/yolov8s_h8.hef";
    Hailo8Inference hailo(hef_path);
    if (!hailo.initialize()) {
        std::cerr << "Failed to initialize Hailo\n";
        return 1;
    }

    // ── Camera pipeline ─────────────────────
    // Extra pool buffers so camera never stalls while
    // preprocess thread is busy converting
    constexpr std::size_t kBuffers       = 8;
    constexpr std::size_t kBytesPerFrame = 640 * 640 * 3;

    queue<FramePacket>  frameQueue(4);
    queue<InferPacket*> inferQueue(4);
    BufferPool pool(kBuffers, kBytesPerFrame);

    CameraCapture capture(frameQueue, pool, 30);
    if (!capture.start()) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // ── Per-frame stats (written only by inference thread) ──
    std::vector<double> preprocess_ms;
    std::vector<double> inference_ms;
    std::vector<double> parse_ms;
    std::vector<double> e2e_ms;
    preprocess_ms.reserve(4096);
    inference_ms .reserve(4096);
    parse_ms     .reserve(4096);
    e2e_ms       .reserve(4096);

    std::atomic<int>  frame_count{0};
    TimePoint         wall_start;
    std::atomic<bool> wall_started{false};

    // ═════════════════════════════════════════
    //  INFERENCE THREAD
    //  Drains inferQueue → Hailo → parse → stats
    // ═════════════════════════════════════════
    std::thread infer_thread([&]()
    {
            // Pin inference to core 1
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        InferPacket* ipkt = nullptr;

        while (inferQueue.pop(ipkt))


        {   
            std::cout << "inferQ after pop: " << inferQueue.size() << "\n";
            if (!ipkt) continue;

            // ── Inference ───────────────────
            uint64_t t0 = now_ns();
            hailo.run(ipkt->rgb.data());
            uint64_t inf_ns = now_ns() - t0;

            // ── Parse NMS ───────────────────
            t0 = now_ns();
            auto detections = parse_nms_output(hailo.get_output());
            uint64_t parse_ns = now_ns() - t0;

            // ── End-to-end from camera arrival ──
            uint64_t e2e_ns = now_ns() - ipkt->cam_ts;

            // ── Record ──────────────────────
            preprocess_ms.push_back(to_ms(ipkt->preprocess_ns));
            inference_ms .push_back(to_ms(inf_ns));
            parse_ms     .push_back(to_ms(parse_ns));
            e2e_ms       .push_back(to_ms(e2e_ns));

            int fc = ++frame_count;

            std::cout << "Frame " << fc
                      << "  inf="  << to_ms(inf_ns)  << "ms"
                      << "  e2e="  << to_ms(e2e_ns)  << "ms"
                      << "  dets=" << detections.size()
                      << "\n";

            if (fc % REPORT_INTERVAL == 0 && wall_started.load()) {
                double elapsed =
                    Ms(Clock::now() - wall_start).count() / 1000.0;
                print_stats(fc,
                    preprocess_ms, inference_ms,
                    parse_ms, e2e_ms, elapsed);
            }

            delete ipkt;
        }
    });

    // ═════════════════════════════════════════
    //  PREPROCESS THREAD  (main thread)
    //  Drains frameQueue → BGR->RGB → inferQueue
    //  Releases pool buffer immediately after
    //  conversion so camera never stalls
    // ═════════════════════════════════════════
    std::cout << "Warming up for " << WARMUP_FRAMES << " frames...\n";

    FramePacket pkt{};
    int warmup_count = 0;

    while (g_running && frameQueue.pop(pkt))
    {
        const uint64_t cam_ts = now_ns();

        // ── Warmup: just drain frames ────────
        if (warmup_count < WARMUP_FRAMES) {
            pool.release(pkt.data);
            ++warmup_count;
            if (warmup_count == WARMUP_FRAMES) {
                std::cout << "Warmup done. Measuring...\n\n";
                wall_start   = Clock::now();
                wall_started = true;
            }
            continue;
        }

        // ── BGR -> RGB ──────────────────────
        uint64_t t0 = now_ns();
        cv::Mat bgr(640, 640, CV_8UC3, pkt.data, 640 * 3);
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        uint64_t preprocess_ns = now_ns() - t0;

        // Release pool buffer immediately — inference thread
        // works from the copied rgb vector below
        pool.release(pkt.data);

        // ── Build and queue infer packet ─────
        auto* ipkt          = new InferPacket();
        ipkt->rgb           = std::vector<uint8_t>(
                                  rgb.data, rgb.data + 640 * 640 * 3);
        ipkt->cam_ts        = cam_ts;
        ipkt->preprocess_ns = preprocess_ns;

        if (!inferQueue.push(ipkt)) {
            std::cout << "inferQ after push: " << inferQueue.size()
                    << "  frameQ: " << frameQueue.size() << "\n";
            delete ipkt;
            break;
        }
    }

    // ── Drain and shut down inference thread ──
    inferQueue.stop();
    infer_thread.join();

    // ── Final report ────────────────────────
    if (!inference_ms.empty() && wall_started.load()) {
        double elapsed =
            Ms(Clock::now() - wall_start).count() / 1000.0;
        std::cout << "\n── Final Report ──\n";
        print_stats(frame_count.load(),
            preprocess_ms, inference_ms,
            parse_ms, e2e_ms, elapsed);
    }

    capture.stop();
    frameQueue.stop();

    std::cout << "Done. Measured " << frame_count.load() << " frames.\n";
    return 0;
}
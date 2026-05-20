/*
    integrated_main_audio.cpp  —  NorthStar end-to-end perception pipeline
                                  with TTS speech output

    Forked from integrated_main_2.cpp. Identical pipeline behaviour but
    additionally speaks the assembled sentence through the I²S MAX98357A
    speaker after each qualifying detection pair (once per button press;
    future agreement function will replace the once-per-press stub).

    Subsystems:
        Button        — gates camera + IR capture (held = on, released = off)
        Sensor Ingest — CaptureController (camera + IR)
        Inference     — Hailo8Inference (always-on, processes frames as they arrive)
        Mic Intent    — Pipeline (always-on, Vosk ASR → COCO id)
        Filter        — DetectionFilter (correlates cam0/cam1 detections)
        Speech        — SpeechAggregator (Piper TTS → libpulse → I²S DAC)

    Output: FilteredInferencePair structs printed to stdout, distilled into
            a natural-language sentence, printed and spoken aloud.

    Log prefixes:
        [main]       lifecycle / shutdown
        [button]     press / release events
        [cap]        capture controller state
        [cam]        camera consumer thread
        [ir]         IR consumer thread
        [hailo]      Hailo inference callbacks
        [mic]        mic pipeline + intent
        [filter]     filtered pair output
        [speech]     TTS aggregator + playback
*/

#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "coco_lookup.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "detection_filter.hpp"
#include "detection_utils.hpp"
#include "inference_packet.hpp"
#include "pipeline.hpp"
#include "button-driver.h"
#include "gpio.h"
#include "hailo8_inference.hpp"
#include "frame_buffer.hpp"
#include "stereo_distance.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ir_frame_buffer.hpp"
#include "ir_aligner.hpp"
#include "temp_estimator.hpp"
#include "direction_estimator.hpp"
#include "output_struct.hpp"
#include "speech_aggregator.hpp"

static IRFrameBuffer g_frame_buf_ir;
static IRAligner* g_ir_aligner_ptr = nullptr;
static TempEstimator* g_temp_estimator_ptr = nullptr;
static DirectionEstimator* g_dir_estimator_ptr = nullptr;
static SpeechAggregator* g_speech_ptr = nullptr;


// ─── Runtime debug controls ─────────────────────────────────────────────────
enum DebugBits : uint32_t {
    DBG_MAIN   = 1u << 0,
    DBG_BUTTON = 1u << 1,
    DBG_CAP    = 1u << 2,
    DBG_CAM    = 1u << 3,
    DBG_IR     = 1u << 4,
    DBG_HAILO  = 1u << 5,
    DBG_MIC    = 1u << 6,
    DBG_FILTER = 1u << 7,
    DBG_DEPTH  = 1u << 8,
    DBG_TEMP   = 1u << 9,
    DBG_SPEECH = 1u << 10,
    DBG_SAVE   = 1u << 11,
    DBG_ALL    = 0xFFFFFFFFu
};

static std::atomic<uint32_t> g_debug_mask{DBG_MAIN | DBG_BUTTON | DBG_CAP |
                                          DBG_MIC | DBG_FILTER | DBG_DEPTH |
                                          DBG_TEMP | DBG_SPEECH};
static constexpr double g_speech_gain = 1.0;  // TTS gain fixed at unity for now
static constexpr const char* kDefaultVoskModelDir = "../model_inf/vosk-model-small-en-us-0.15";

// ─── Piper TTS configuration ─────────────────────────────────────────────────
// Passed to SpeechAggregator so the persistent TTSEngine subprocess (and its
// 65 MB ONNX model) is loaded exactly once at startup rather than per utterance.
// Override at runtime with --piper-bin=<path> / --piper-model=<path> if needed.
static const char* g_piper_bin   = "/home/dst5/NSJamie/NorthStar/.venv/bin/piper";
static const char* g_piper_model = "/home/dst5/NSJamie/NorthStar/model_inf/en_US-lessac-medium.onnx";
static constexpr int kTtsSampleRate = 22050;

static bool dbg_on(uint32_t bit) {
    return (g_debug_mask.load() & bit) != 0;
}

static void dbg_printf(uint32_t bit, const char* fmt, ...) {
    if (!dbg_on(bit)) return;
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
}

static uint32_t debug_bit_for_name(const std::string& name) {
    if (name == "main")   return DBG_MAIN;
    if (name == "button") return DBG_BUTTON;
    if (name == "cap")    return DBG_CAP;
    if (name == "cam")    return DBG_CAM;
    if (name == "ir")     return DBG_IR;
    if (name == "hailo")  return DBG_HAILO;
    if (name == "mic")    return DBG_MIC;
    if (name == "filter") return DBG_FILTER;
    if (name == "depth")  return DBG_DEPTH;
    if (name == "temp")   return DBG_TEMP;
    if (name == "speech") return DBG_SPEECH;
    if (name == "save")   return DBG_SAVE;
    return 0;
}

static bool parse_debug_mask(const std::string& spec, uint32_t& out) {
    if (spec == "all")  { out = DBG_ALL; return true; }
    if (spec == "none" || spec == "quiet") { out = 0; return true; }

    uint32_t mask = 0;
    size_t start = 0;
    while (start <= spec.size()) {
        size_t comma = spec.find(',', start);
        std::string token = spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            const uint32_t bit = debug_bit_for_name(token);
            if (!bit) return false;
            mask |= bit;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    out = mask;
    return true;
}

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options] <word1> [word2 ...]\n"
        "\n"
        "Options:\n"
        "  --debug=<list>        Debug categories to print. Use comma-separated names.\n"
        "                        Categories: main,button,cap,cam,ir,hailo,mic,filter,depth,temp,speech,save\n"
        "                        Special values: all, none, quiet\n"
        "                        Default: main,button,cap,mic,filter,depth,temp,speech\n"
        "  --vosk-model=<dir>    Override Vosk model directory.\n"
        "                        Default: ../model_inf/vosk-model-small-en-us-0.15\n"
        "  --piper-bin=<path>    Path to piper executable.\n"
        "                        Default: /usr/local/bin/piper\n"
        "  --piper-model=<path>  Path to Piper .onnx voice model.\n"
        "                        Default: /opt/piper/en_US-lessac-medium.onnx\n"
        "  --speech-gain=<N>     Accepted for compatibility, but ignored; gain is fixed at 1.0.\n"
        "  --help                Show this help.\n"
        "\n"
        "Example:\n"
        "  %s --debug=main,mic,speech cup person\n",
        argv0, argv0);
}

// ─── Diagnostic image save state ─────────────────────────────────────────────
static std::mutex        g_save_mutex;
static std::atomic<int>  g_pairs_saved{0};
static std::string       g_save_dir;   // set in main()

// ─── Drawing helpers ──────────────────────────────────────────────────────────
static void draw_norm_bbox(cv::Mat& img, const BoundingBox& box,
                           const cv::Scalar& colour, const std::string& label)
{
    const int W = img.cols, H = img.rows;
    const int x0 = std::clamp(static_cast<int>(box.x_min * W), 0, W - 1);
    const int y0 = std::clamp(static_cast<int>(box.y_min * H), 0, H - 1);
    const int x1 = std::clamp(static_cast<int>(box.x_max * W), 0, W - 1);
    const int y1 = std::clamp(static_cast<int>(box.y_max * H), 0, H - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, colour, 2);
    if (!label.empty())
        cv::putText(img, label, {x0 + 4, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
}

static void draw_pixel_bbox(cv::Mat& img, int x0, int y0, int x1, int y1,
                            const cv::Scalar& colour, const std::string& label)
{
    x0 = std::clamp(x0, 0, img.cols - 1);
    y0 = std::clamp(y0, 0, img.rows - 1);
    x1 = std::clamp(x1, 0, img.cols - 1);
    y1 = std::clamp(y1, 0, img.rows - 1);
    cv::rectangle(img, {x0, y0}, {x1, y1}, colour, 2);
    if (!label.empty())
        cv::putText(img, label, {x0 + 4, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
}

static cv::Mat side_by_side(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out; cv::hconcat(a, b, out); return out;
}

// Rectify a normalised bbox into pixel coords in the rectified image.
static cv::Rect rectify_bbox(const BoundingBox& box,
                             const cv::Mat& K, const cv::Mat& D,
                             const cv::Mat& R, const cv::Mat& P,
                             const cv::Size& sz)
{
    const float W = static_cast<float>(sz.width);
    const float H = static_cast<float>(sz.height);
    std::vector<cv::Point2f> raw = {
        {box.x_min*W, box.y_min*H}, {box.x_max*W, box.y_min*H},
        {box.x_max*W, box.y_max*H}, {box.x_min*W, box.y_max*H},
    };
    std::vector<cv::Point2f> rect;
    cv::undistortPoints(raw, rect, K, D, R, P);
    float mn_x= 1e9, mn_y= 1e9, mx_x=-1e9, mx_y=-1e9;
    for (auto& p : rect) {
        mn_x=std::min(mn_x,p.x); mn_y=std::min(mn_y,p.y);
        mx_x=std::max(mx_x,p.x); mx_y=std::max(mx_y,p.y);
    }
    if (mx_x<0||mx_y<0||mn_x>sz.width-1||mn_y>sz.height-1) return {};
    int rx0=std::max(0,(int)std::floor(mn_x)), ry0=std::max(0,(int)std::floor(mn_y));
    int rx1=std::min(sz.width-1,(int)std::ceil(mx_x)), ry1=std::min(sz.height-1,(int)std::ceil(mx_y));
    if (rx1<=rx0||ry1<=ry0) return {};
    return cv::Rect(cv::Point(rx0,ry0), cv::Point(rx1+1,ry1+1));
}

static cv::Mat colourise_disparity(const cv::Mat& disp)
{
    cv::Mat mask = disp > 0;
    double lo=0, hi=0;
    cv::minMaxLoc(disp, &lo, &hi, nullptr, nullptr, mask);
    if (hi<=lo) hi=lo+1.0;
    cv::Mat n8; disp.convertTo(n8, CV_8U, 255.0/(hi-lo), -255.0*lo/(hi-lo));
    cv::Mat col; cv::applyColorMap(n8, col, cv::COLORMAP_JET);
    col.setTo(cv::Scalar(0,0,0), ~mask);
    return col;
}

// Save diagnostics for one pair: unrectified side-by-side, rectified
// side-by-side (with epipolar lines), disparity map.
static void save_pair_diagnostics(int idx,
                                  const cv::Mat& left_raw,
                                  const cv::Mat& right_raw,
                                  const BoundingBox& box_cam0,
                                  const BoundingBox* box_cam1,
                                  std::optional<float> depth_m,
                                  const StereoDepthEstimator& est)
{
    char prefix[256];
    std::snprintf(prefix, sizeof(prefix), "%s/pair_%04d",
                  g_save_dir.c_str(), idx);

    const auto save = [&](const std::string& path, const cv::Mat& img) {
        if (cv::imwrite(path, img)) {
            dbg_printf(DBG_SAVE, "[save] %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "[save] FAILED: %s\n", path.c_str());
        }
    };

    // 1. Unrectified side-by-side
    {
        cv::Mat l = left_raw.clone(), r = right_raw.clone();
        draw_norm_bbox(l, box_cam0, cv::Scalar(0,255,0), "cam0");
        if (box_cam1)
            draw_norm_bbox(r, *box_cam1, cv::Scalar(0,255,0), "cam1");
        else
            cv::putText(r, "no cam1 det", {8,24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 1, cv::LINE_AA);
        for (int y=64; y<l.rows; y+=64) {
            cv::line(l,{0,y},{l.cols-1,y},cv::Scalar(0,128,255),1);
            cv::line(r,{0,y},{r.cols-1,y},cv::Scalar(0,128,255),1);
        }
        save(std::string(prefix)+"_unrectified.png", side_by_side(l,r));
    }

    // 2. Rectified side-by-side
    {
        const cv::Mat& lr = est.last_left_rect();
        const cv::Mat& rr = est.last_right_rect();
        if (!lr.empty() && !rr.empty()) {
            cv::Mat l = lr.clone(), r = rr.clone();
            for (int y=64; y<l.rows; y+=64) {
                cv::line(l,{0,y},{l.cols-1,y},cv::Scalar(0,128,255),1);
                cv::line(r,{0,y},{r.cols-1,y},cv::Scalar(0,128,255),1);
            }
            cv::Rect rb_l = rectify_bbox(box_cam0,
                est.K1(),est.D1(),est.R1(),est.P1(),l.size());
            if (rb_l.area()>0)
                draw_pixel_bbox(l,rb_l.x,rb_l.y,
                                rb_l.x+rb_l.width-1,rb_l.y+rb_l.height-1,
                                cv::Scalar(0,255,0),"cam0 rect");
            if (box_cam1) {
                cv::Rect rb_r = rectify_bbox(*box_cam1,
                    est.K2(),est.D2(),est.R2(),est.P2(),r.size());
                if (rb_r.area()>0)
                    draw_pixel_bbox(r,rb_r.x,rb_r.y,
                                    rb_r.x+rb_r.width-1,rb_r.y+rb_r.height-1,
                                    cv::Scalar(0,255,0),"cam1 rect");
            }
            save(std::string(prefix)+"_rectified.png", side_by_side(l,r));
        } else {
            dbg_printf(DBG_SAVE, "[save] WARNING: rectified frames not available for pair %d\n", idx);
        }
    }

    // 3. Disparity map
    {
        const cv::Mat& disp = est.last_disparity();
        if (!disp.empty()) {
            cv::Mat col = colourise_disparity(disp);
            cv::Rect rb = rectify_bbox(box_cam0,
                est.K1(),est.D1(),est.R1(),est.P1(),col.size());
            char dlabel[32] = "depth: n/a";
            if (depth_m) std::snprintf(dlabel, sizeof(dlabel), "depth: %.2f m", *depth_m);
            if (rb.area()>0)
                draw_pixel_bbox(col,rb.x,rb.y,
                                rb.x+rb.width-1,rb.y+rb.height-1,
                                cv::Scalar(255,255,255),dlabel);
            else
                cv::putText(col,dlabel,{8,24},
                            cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(255,255,255),1,cv::LINE_AA);
            save(std::string(prefix)+"_disparity.png", col);
        } else {
            dbg_printf(DBG_SAVE, "[save] WARNING: disparity not available for pair %d\n", idx);
        }
    }
}


// ─── Globals for signal handling ─────────────────────────────────────────────
std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;
std::atomic<bool> mic_armed{false};
static std::atomic<bool> g_button_down{false};
static std::atomic<uint64_t> g_session_generation{0};
static std::atomic<uint64_t> g_target_session_generation{0};

// ─── Counters for periodic reporting ─────────────────────────────────────────
static std::atomic<uint64_t> g_cam_frames_consumed{0};
static std::atomic<uint64_t> g_ir_frames_consumed{0};
static std::atomic<uint64_t> g_hailo_callbacks{0};
static std::atomic<uint64_t> g_pairs_emitted{0};

static FrameBuffer g_frame_buf_cam0;
static FrameBuffer g_frame_buf_cam1;
static StereoDepthEstimator* g_depth_ptr = nullptr;

// ─── Distance-validation tunables ────────────────────────────────────────────
// Pairs with depth > kMaxValidDistanceM are hard-rejected before they reach
// the agreement window. Pairs that pass that check accumulate in a sliding
// window of size kAgreementWindow; once kAgreementRequired of those agree to
// within kAgreementToleranceFrac of each other, we report the median of the
// agreeing subset.
static constexpr float kMaxValidDistanceM       = 3.5f;
static constexpr size_t kAgreementWindow        = 6;
static constexpr size_t kAgreementRequired      = 3;
static constexpr float  kAgreementToleranceFrac = 0.10f;  // 10 %

// One pace ≈ 0.75 m (close to British military pace, comfortable adult stride).
static constexpr float kMetresPerPace = 0.75f;

// Room-temperature band — outside this we describe the object qualitatively
// (cool / warm / hot) rather than naming a number.
static constexpr float kRoomTempLowC  = 18.0f;
static constexpr float kRoomTempHighC = 25.0f;
static constexpr float kHotThresholdC = 35.0f;

// Rolling window of recently accepted distances within the current session.
// Held under g_agreement_mutex. Cleared on button press alongside filter.reset()
// and speech_aggregator.reset().
//
// Single-threaded in practice (DetectionFilter serialises pair callbacks) but
// the press callback runs on the button thread, so we still need the mutex
// to make the clear-on-press race-free.
static std::mutex            g_agreement_mutex;
static std::deque<float>     g_recent_distances;

// Returns the median of an agreeing subset of >= kAgreementRequired samples
// from the rolling window where max/min <= 1 + kAgreementToleranceFrac.
// Empty optional means no such subset exists yet.
static std::optional<float> check_agreement()
{
    std::lock_guard<std::mutex> lk(g_agreement_mutex);
    if (g_recent_distances.size() < kAgreementRequired) return std::nullopt;

    // Sort a copy; slide a window of kAgreementRequired entries; the first
    // window with max/min within tolerance wins. The median of that window
    // is what we report.
    std::vector<float> sorted(g_recent_distances.begin(), g_recent_distances.end());
    std::sort(sorted.begin(), sorted.end());

    const float ratio_limit = 1.0f + kAgreementToleranceFrac;
    for (size_t i = 0; i + kAgreementRequired <= sorted.size(); ++i) {
        const float lo = sorted[i];
        const float hi = sorted[i + kAgreementRequired - 1];
        if (lo > 0 && hi / lo <= ratio_limit) {
            return sorted[i + kAgreementRequired / 2];  // median of the window
        }
    }
    return std::nullopt;
}

static void push_distance(float d_m)
{
    std::lock_guard<std::mutex> lk(g_agreement_mutex);
    g_recent_distances.push_back(d_m);
    while (g_recent_distances.size() > kAgreementWindow)
        g_recent_distances.pop_front();
}

static void clear_distances()
{
    std::lock_guard<std::mutex> lk(g_agreement_mutex);
    g_recent_distances.clear();
}

// Snapshot of the current buffer for logging without holding the lock during printf.
static std::vector<float> snapshot_distances()
{
    std::lock_guard<std::mutex> lk(g_agreement_mutex);
    return {g_recent_distances.begin(), g_recent_distances.end()};
}

// ─── Phrase builders ─────────────────────────────────────────────────────────
static std::string distance_phrase(float d_m)
{
    if (d_m < 1.0f) return "within arms reach";
    const int paces = static_cast<int>(std::lround(d_m / kMetresPerPace));
    if (paces <= 1) return "around 1 pace away";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "around %d paces away", paces);
    return buf;
}

static std::string temperature_phrase(float t_c)
{
    if (t_c >= kRoomTempLowC && t_c <= kRoomTempHighC)
        return "It looks like it's at room temperature.";
    if (t_c < kRoomTempLowC)      return "It feels cool.";
    if (t_c <= kHotThresholdC)    return "It feels warm.";
    return "It feels hot.";
}

static void on_signal(int /*sig*/)
{
    g_running = false;
    if (g_speech_ptr) g_speech_ptr->interrupt();
    if (g_controller_ptr) g_controller_ptr->stop_capture();
}


// ─── FilteredInferencePair print callback ───────────────────────────────────
static void on_filtered_pair(FilteredInferencePair pair)
{
    const uint64_t pair_session = g_session_generation.load(std::memory_order_acquire);
    const uint64_t n = ++g_pairs_emitted;
    OutputReport output_report; 
    output_report.object_id = pair.object_id;

    // Once this button/session has spoken one complete result, ignore all
    // remaining queued pairs before doing depth/temp/report work. This keeps
    // stale post-release results from burning CPU and creating delayed speech.
    if (g_speech_ptr && g_speech_ptr->already_spoken()) {
        dbg_printf(DBG_SPEECH,
                   "[speech] drop: result ignored, already spoken this session\n");
        std::fflush(stdout);
        return;
    }

    dbg_printf(DBG_FILTER, "\n[filter] PAIR #%llu  object_id=%-3d  ts=%-12llu  "
                "cam0=%zu det  cam1=%zu det\n",
                static_cast<unsigned long long>(n),
                pair.object_id,
                static_cast<unsigned long long>(pair.timestamp_avg),
                pair.cam0_detections.size(),
                pair.cam1_detections.size());

    if (pair.cam0_detections.empty() || pair.cam1_detections.empty()) {
        dbg_printf(DBG_FILTER, "[filter] pair missing detections on a side; skipping depth\n");
        std::fflush(stdout);
        return;
    }

    // Highest-confidence cam0 detection (filter already guarantees these are
    // all of the target class).
    const Detection* best_cam0 = &pair.cam0_detections.front();
    for (const auto& d : pair.cam0_detections) {
        if (d.confidence > best_cam0->confidence) best_cam0 = &d;
    }

    dbg_printf(DBG_FILTER, "[filter]   cam0  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                best_cam0->confidence,
                best_cam0->box.x_min, best_cam0->box.y_min,
                best_cam0->box.x_max, best_cam0->box.y_max);

    for (const auto& d : pair.cam1_detections)
        dbg_printf(DBG_FILTER, "[filter]   cam1  conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                    d.confidence,
                    d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max);

    // ── Retrieve frames closest to this pair's timestamp ────────────────────
    cv::Mat left  = g_frame_buf_cam0.find_closest(pair.timestamp_avg);
    cv::Mat right = g_frame_buf_cam1.find_closest(pair.timestamp_avg);

    if (left.empty() || right.empty()) {
        dbg_printf(DBG_DEPTH, "[depth] no frame in buffer for pair ts=%llu  "
                    "(cam0_empty=%d cam1_empty=%d)\n",
                    static_cast<unsigned long long>(pair.timestamp_avg),
                    left.empty(), right.empty());
        std::fflush(stdout);
        return;
    }

    // ── Compute stereo depth at the cam0 bounding box ───────────────────────
    if (!g_depth_ptr) {
        dbg_printf(DBG_DEPTH, "[depth] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    auto depth_m = g_depth_ptr->compute(left, right, best_cam0->box);

    if (depth_m) {
        if (*depth_m > kMaxValidDistanceM) {
            // Hard-reject: don't pollute the agreement window with what is
            // almost certainly a mismatched-pair / textureless artefact.
            dbg_printf(DBG_DEPTH,
                "[depth] REJECT Z = %.2f m > %.2f m (not added to agreement window)\n",
                *depth_m, kMaxValidDistanceM);
            std::fflush(stdout);
            depth_m.reset();
        } else {
            dbg_printf(DBG_DEPTH, "[depth] Z = %.2f m  (object_id=%d)\n",
                        *depth_m, pair.object_id);
            output_report.distance = *depth_m;
            push_distance(*depth_m);

            if (dbg_on(DBG_FILTER)) {
                const auto snap = snapshot_distances();
                std::printf("[filter] agreement window (%zu/%zu): [",
                            snap.size(), kAgreementWindow);
                for (size_t i = 0; i < snap.size(); ++i)
                    std::printf("%s%.2f", i ? ", " : "", snap[i]);
                std::printf("]\n");
                std::fflush(stdout);
            }
        }
    } else {
        dbg_printf(DBG_DEPTH, "[depth] could not compute (textureless / out of range / "
                    "bbox outside rectified image)\n");
    }
    std::fflush(stdout);

    // ── Save diagnostic PNGs for this pair ──────────────────────────────────
    {
        const Detection* best_cam1 = pair.cam1_detections.empty()
            ? nullptr : &pair.cam1_detections.front();
        for (const auto& d : pair.cam1_detections)
            if (d.confidence > best_cam1->confidence) best_cam1 = &d;

        std::lock_guard<std::mutex> lk(g_save_mutex);
        const int idx = g_pairs_saved.fetch_add(1) + 1;
        save_pair_diagnostics(idx, left, right,
                              best_cam0->box,
                              best_cam1 ? &best_cam1->box : nullptr,
                              depth_m,
                              *g_depth_ptr);
    }


    // now find the IR frame and align it
    auto ir = g_frame_buf_ir.find_closest(pair.timestamp_avg);

    if (!ir) {
        dbg_printf(DBG_IR, "[ir] no IR frame in buffer for pair ts=%llu\n",
                    static_cast<unsigned long long>(pair.timestamp_avg));
        std::fflush(stdout);
        return;
    }
    if (!g_ir_aligner_ptr) {
        dbg_printf(DBG_IR, "[ir] aligner not initialised\n");
        std::fflush(stdout);
        return;
    }

    IRAligner::PixelRect ir_box =
        g_ir_aligner_ptr->project_bbox(best_cam0->box);

    if (!ir_box.valid) {
        dbg_printf(DBG_IR, "[ir] projected bbox invalid / outside IR image\n");
        std::fflush(stdout);
        return;
    }


    dbg_printf(DBG_IR, "[ir] projected bbox: [%d %d %d %d]\n",
                ir_box.x0, ir_box.y0, ir_box.x1, ir_box.y1);
    std::fflush(stdout);

    if (!g_temp_estimator_ptr) {
        dbg_printf(DBG_TEMP, "[temp] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    //now estimate the temperature inside the IR projected bounding box
    auto median_temp = g_temp_estimator_ptr->estimate_temp(ir_box, *ir);
    if (median_temp) {
        dbg_printf(DBG_TEMP, "[temp] median temperature in box: %.2f C\n", *median_temp);
        output_report.temp = *median_temp;
    } else {
        dbg_printf(DBG_TEMP, "[temp] could not estimate median temperature\n");
    }

    // Direction estimation:
    if (!g_dir_estimator_ptr) {
        dbg_printf(DBG_FILTER, "[direction] estimator not initialised\n");
        std::fflush(stdout);
        return;
    }

    Direction direct = g_dir_estimator_ptr->get_direction(best_cam0->box);
    output_report.direction = g_dir_estimator_ptr->cvt_to_string(direct);

    const char* object_name = coco_word_for_id(output_report.object_id);

    if (!output_report.distance || !output_report.temp) {
        dbg_printf(DBG_FILTER, "[output] incomplete report: distance=%s temp=%s\n",
                    output_report.distance ? "yes" : "no",
                    output_report.temp ? "yes" : "no");
        std::fflush(stdout);
        return;
    }

    // ── Agreement gate ──────────────────────────────────────────────────────
    // Only speak when the rolling window contains a coherent set of distance
    // estimates. Until then, we've still done the per-pair work (depth + IR
    // + direction) — we just hold back on actually speaking.
    auto agreed_distance = check_agreement();
    if (!agreed_distance) {
        dbg_printf(DBG_FILTER,
            "[agree] not enough agreement yet (window has %zu samples; need "
            "%zu agreeing within %.0f%%)\n",
            snapshot_distances().size(),
            kAgreementRequired,
            kAgreementToleranceFrac * 100.0f);
        std::fflush(stdout);
        return;
    }
    dbg_printf(DBG_FILTER,
        "[agree] OK — reporting median agreed distance = %.2f m\n",
        *agreed_distance);
    std::fflush(stdout);

    // If a new button press happened while this pair was being processed, this
    // result belongs to the previous interaction and must not speak in the new
    // session. Also require that the mic armed a target during this same session.
    const uint64_t current_session = g_session_generation.load(std::memory_order_acquire);
    const uint64_t target_session  = g_target_session_generation.load(std::memory_order_acquire);
    if (pair_session != current_session || target_session != current_session) {
        dbg_printf(DBG_SPEECH,
            "[speech] drop: stale result/session mismatch pair=%llu current=%llu target=%llu\n",
            static_cast<unsigned long long>(pair_session),
            static_cast<unsigned long long>(current_session),
            static_cast<unsigned long long>(target_session));
        std::fflush(stdout);
        return;
    }

    // Build the natural-language sentence using the friendlier phrasing:
    //   distance: "within arms reach" / "around N paces away"
    //   temperature: qualitative band rather than a numeric reading.
    const std::string dist_phrase = distance_phrase(*agreed_distance);
    const std::string temp_phrase = temperature_phrase(*output_report.temp);

    // Overwrite the reported distance with the agreed-on median so any
    // downstream consumer of OutputReport sees the consensus value, not the
    // last raw sample.
    output_report.distance = *agreed_distance;

    char sentence_buf[256];
    std::snprintf(sentence_buf, sizeof(sentence_buf),
        "Your %s is in the %s of your vision, %s. %s",
        object_name,
        output_report.direction.c_str(),
        dist_phrase.c_str(),
        temp_phrase.c_str());

    dbg_printf(DBG_SPEECH, "[speech] sentence: %s\n", sentence_buf);
    std::fflush(stdout);

    if (g_speech_ptr) {
        g_speech_ptr->on_report(output_report, sentence_buf);
    }
}



int main(int argc, char* argv[])
{
    std::vector<std::string> positional;
    positional.reserve(argc > 1 ? static_cast<size_t>(argc - 1) : 0);
    std::string vosk_model_dir = kDefaultVoskModelDir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.rfind("--debug=", 0) == 0) {
            uint32_t mask = 0;
            const std::string spec = arg.substr(std::strlen("--debug="));
            if (!parse_debug_mask(spec, mask)) {
                std::fprintf(stderr, "[main] ERROR: invalid --debug category list: %s\n", spec.c_str());
                print_usage(argv[0]);
                return 1;
            }
            g_debug_mask.store(mask);
        } else if (arg.rfind("--vosk-model=", 0) == 0) {
            vosk_model_dir = arg.substr(std::strlen("--vosk-model="));
            if (vosk_model_dir.empty()) {
                std::fprintf(stderr, "[main] ERROR: --vosk-model requires a directory.\n");
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg.rfind("--piper-bin=", 0) == 0) {
            const std::string val = arg.substr(std::strlen("--piper-bin="));
            if (val.empty()) {
                std::fprintf(stderr, "[main] ERROR: --piper-bin requires a path.\n");
                print_usage(argv[0]);
                return 1;
            }
            // Store in a durable std::string; g_piper_bin points into it.
            static std::string s_piper_bin_store;
            s_piper_bin_store = val;
            g_piper_bin = s_piper_bin_store.c_str();
        } else if (arg.rfind("--piper-model=", 0) == 0) {
            const std::string val = arg.substr(std::strlen("--piper-model="));
            if (val.empty()) {
                std::fprintf(stderr, "[main] ERROR: --piper-model requires a path.\n");
                print_usage(argv[0]);
                return 1;
            }
            static std::string s_piper_model_store;
            s_piper_model_store = val;
            g_piper_model = s_piper_model_store.c_str();
        } else if (arg.rfind("--speech-gain=", 0) == 0) {
            // Accepted for compatibility with older run commands, but ignored.
            // Speech gain is intentionally hardcoded to 1.0 for now.
        } else if (arg == "--speech-test") {
            // Accepted for compatibility, but intentionally ignored.
            // No startup speech is emitted.
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "[main] ERROR: unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        } else {
            positional.push_back(arg);
        }
    }

    // Backward compatibility: if the first positional argument looks like a
    // Vosk model directory, treat it as the model path and use the remaining
    // positionals as object words. New usage does not require this argument.
    size_t object_arg_start = 0;
    if (!positional.empty() && std::filesystem::is_directory(positional[0]) &&
        std::filesystem::exists(std::filesystem::path(positional[0]) / "am")) {
        vosk_model_dir = positional[0];
        object_arg_start = 1;
    }

    if (positional.size() <= object_arg_start) {
        print_usage(argv[0]);
        return 1;
    }

    if (!std::filesystem::is_directory(vosk_model_dir)) {
        std::fprintf(stderr,
            "[main] ERROR: Vosk model directory not found: %s\n"
            "       Expected default location from build/: ../model_inf/vosk-model-small-en-us-0.15\n"
            "       Or pass --vosk-model=<dir>.\n",
            vosk_model_dir.c_str());
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    dbg_printf(DBG_MAIN, "[main] NorthStar integrated pipeline — initialising\n");

    dbg_printf(DBG_MAIN, "[main] setting up GPIO\n");
    gpio::setupGpio();

    // ── Filter ───────────────────────────────────────────────────────────────
    dbg_printf(DBG_MAIN, "[main] creating DetectionFilter\n");
    DetectionFilter filter(on_filtered_pair);

    // ── Mic pipeline config ──────────────────────────────────────────────────
    dbg_printf(DBG_MAIN, "[main] configuring mic pipeline (vosk model: %s)\n", vosk_model_dir.c_str());
    dbg_printf(DBG_MAIN, "[main] debug mask=0x%08x  speech_gain=%.2f\n", g_debug_mask.load(), g_speech_gain);
    Pipeline::Config mic_cfg;
    mic_cfg.model_path = vosk_model_dir;

    for (size_t i = object_arg_start; i < positional.size(); ++i) {
        const std::string word = positional[i];
        const uint8_t     id   = coco_id_for_word(word);
        if (id == 255) {
            std::fprintf(stderr,
                "[main] WARNING: \"%s\" is not a COCO class — will never match.\n",
                word.c_str());
        } else {
            dbg_printf(DBG_MAIN, "[main]   registered: \"%s\" -> COCO id %d\n",
                        word.c_str(), id);
        }
        mic_cfg.object_list.push_back(word);
        mic_cfg.object_ids.push_back(id);
    }

    mic_cfg.on_detection = [](const DetectionResult& r) {
        // Only print when armed, to avoid console spam between presses.
        if (!mic_armed.load()) return;
        dbg_printf(DBG_MIC, "[mic] heard \"%-12s\"  id=%-3d  (%s)\n",
                    r.word.c_str(), r.object_id,
                    r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    mic_cfg.on_intent = [&filter](uint8_t id) {
        if (!mic_armed.load()) {
            dbg_printf(DBG_MIC, "[mic] intent ignored (button not held): id=%d\n", id);
            std::fflush(stdout);
            return;
        }
        dbg_printf(DBG_MIC, "[mic] arming filter -> COCO id %d\n", id);
        std::fflush(stdout);
        g_target_session_generation.store(g_session_generation.load(std::memory_order_acquire), std::memory_order_release);
        filter.on_intent(id);
    };

    
    // ── Hailo inference ──────────────────────────────────────────────────────
    dbg_printf(DBG_MAIN, "[main] creating Hailo inference (hef: %s)\n", DEFAULT_HEF_PATH);
    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    /*
    hailo.register_callback(
        [&filter](uint8_t camera_id,
                  uint64_t timestamp_ns,
                  std::vector<std::vector<uint8_t>> raw_output)
        {
            const uint64_t n = ++g_hailo_callbacks;

            InferencePacket pkt;
            pkt.camera_id  = camera_id;
            pkt.timestamp  = timestamp_ns;
            pkt.detections = parse_detections(raw_output);

            // Log every callback at a low rate to avoid flooding stdout.
            if (n % 30 == 1) {
                std::printf("[hailo] callback #%llu  cam=%d  detections=%zu  "
                            "(armed=%s)\n",
                            static_cast<unsigned long long>(n),
                            camera_id,
                            pkt.detections.size(),
                            filter.is_armed() ? "yes" : "no");
                std::fflush(stdout);
            }

            filter.on_inference_packet(std::move(pkt));
        });
    */

    hailo.register_callback(
    [&filter](uint8_t camera_id,
              uint64_t timestamp_ns,
              std::vector<std::vector<uint8_t>> raw_output)
    {
        const uint64_t n = ++g_hailo_callbacks;

        InferencePacket pkt;
        pkt.camera_id  = camera_id;
        pkt.timestamp  = timestamp_ns;
        pkt.detections = parse_detections(raw_output);

        // If filter is armed, check whether any detection matches the
        // target class and log each match individually. This gives a
        // clear per-frame view of how often each camera sees the object.
        if (filter.is_armed()) {
            const uint8_t target = filter.target_object_id();
            int matches = 0;
            for (const auto& d : pkt.detections) {
                if (d.object_id == target) {
                    ++matches;
                    dbg_printf(DBG_HAILO, "[hailo] HIT cb#%llu  cam=%d  target=%d  "
                                "conf=%.2f  box=[%.2f %.2f %.2f %.2f]\n",
                                static_cast<unsigned long long>(n),
                                camera_id, target, d.confidence,
                                d.box.x_min, d.box.y_min,
                                d.box.x_max, d.box.y_max);
                }
            }
            if (matches > 0) std::fflush(stdout);
        }

        filter.on_inference_packet(std::move(pkt));
    });

    dbg_printf(DBG_MAIN, "[main] initialising Hailo...\n");
    if (!hailo.initialize()) {
        std::fprintf(stderr, "[main] FATAL: Hailo initialise failed\n");
        gpio::teardownGpio();
        return 1;
    }
    dbg_printf(DBG_MAIN, "[main] Hailo initialised: input=%zu bytes, output streams=%zu\n",
                hailo.input_frame_size(), hailo.num_output_streams());

    // ── Capture controller ──────────────────────────────────────────────────
    dbg_printf(DBG_MAIN, "[main] creating CaptureController\n");
    CaptureController controller;
    g_controller_ptr = &controller;

    dbg_printf(DBG_MAIN, "[main] loading stereo calibration\n");
    StereoDepthEstimator depth("../cam_calibration/stereo_new.yaml");  // adjust path as needed
    g_depth_ptr = &depth;
    dbg_printf(DBG_MAIN, "[main] stereo calibration loaded: baseline=%.3f m\n", depth.baseline_m());

    // ── Diagnostic image output directory (timestamped) ──────────────────────
    {
        using namespace std::chrono;
        const auto now = system_clock::to_time_t(system_clock::now());
        std::tm tm{};
        localtime_r(&now, &tm);
        char buf[64];
        std::strftime(buf, sizeof(buf), "stereo_debug/%Y%m%d_%H%M%S", &tm);
        g_save_dir = buf;
        std::error_code ec;
        std::filesystem::create_directories(g_save_dir, ec);
        if (ec)
            std::fprintf(stderr, "[main] WARNING: could not create %s -- %s\n",
                         g_save_dir.c_str(), ec.message().c_str());
        else
            dbg_printf(DBG_SAVE, "[main] diagnostic images -> %s/\n", g_save_dir.c_str());
    }

    // IR Aligner -------------------
    IRAligner ir_aligner("../src/ir_align_100.yaml");
    g_ir_aligner_ptr = &ir_aligner;

    //Temp estimator ----------
    TempEstimator temp_estimator;
    g_temp_estimator_ptr = &temp_estimator;

    //Direction estimator -----
    DirectionEstimator dir_estimator;
    g_dir_estimator_ptr = &dir_estimator;

    //Speech aggregator -------
    // TTSEngine is constructed inside SpeechAggregator.  Piper and the ONNX
    // model are loaded here once; all subsequent synthesise() calls go to the
    // persistent subprocess rather than forking a new process per utterance.
    dbg_printf(DBG_SPEECH, "[main] creating SpeechAggregator (piper: %s  model: %s)\n",
               g_piper_bin, g_piper_model);
    SpeechAggregator speech_aggregator(g_speech_gain, dbg_on(DBG_SPEECH),
                                       g_piper_bin, g_piper_model, kTtsSampleRate);
    if (!speech_aggregator.tts_ready()) {
        std::fprintf(stderr,
            "[main] FATAL: TTSEngine failed to start: %s\n"
            "       Check --piper-bin and --piper-model paths.\n"
            "       Run setup_tts.sh to install Piper if needed.\n",
            speech_aggregator.tts_error().c_str());
        gpio::teardownGpio();
        return 1;
    }
    dbg_printf(DBG_SPEECH, "[main] TTSEngine ready — Piper subprocess running\n");
    g_speech_ptr = &speech_aggregator;
    // ── Camera consumer thread ───────────────────────────────────────────────
    dbg_printf(DBG_CAM, "[main] spawning cam consumer thread\n");
    std::thread cam_thread([&]() {
        dbg_printf(DBG_CAM, "[cam] thread started\n");
        auto& cam_q = controller.get_cam_queue();
        while (auto pkt = cam_q.pop()) {
            const uint64_t n = ++g_cam_frames_consumed;
            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;

            // Store a copy of the BGR frame in the per-camera buffer BEFORE
            // hailo_prepare() (which mutates the data in place).
            cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
            if (pkt->camera_id == 0) g_frame_buf_cam0.push(ts_ns, frame.clone());
            else                     g_frame_buf_cam1.push(ts_ns, frame.clone());

            hailo_prepare(pkt->data);
            hailo.write_frame(pkt->data.data(),
                              pkt->camera_id,
                              ts_ns);

            if (n % 60 == 1) {
                dbg_printf(DBG_CAM, "[cam] consumed frame #%llu  cam=%d  qsize=%zu\n",
                            static_cast<unsigned long long>(n),
                            pkt->camera_id,
                            controller.cam_queue_size());
                std::fflush(stdout);
            }
        }
        dbg_printf(DBG_CAM, "[cam] thread exiting (queue stopped) — %llu frames total\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()));
    });

    // ── IR consumer thread ───────────────────────────────────────────────────
    dbg_printf(DBG_IR, "[main] spawning IR consumer thread\n");
    std::thread ir_thread([&]() {
        dbg_printf(DBG_IR, "[ir] thread started\n");
        auto& ir_q = controller.get_ir_queue();

        while (auto pkt = ir_q.pop()) {
            const uint64_t n = ++g_ir_frames_consumed;

            const uint64_t ts_ns = pkt->timestamp_us * 1000ULL;
            g_frame_buf_ir.push(ts_ns, pkt->temps, pkt->ambientTemp);

            if (n % 10 == 1) {
                dbg_printf(DBG_IR, "[ir] consumed frame #%llu  qsize=%zu  irbuf=%zu  ambient=%.2f C\n",
                            static_cast<unsigned long long>(n),
                            controller.ir_queue_size(),
                            g_frame_buf_ir.size(),
                            pkt->ambientTemp);
                std::fflush(stdout);
            }
        }

        dbg_printf(DBG_IR, "[ir] thread exiting (queue stopped) — %llu frames total\n",
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()));
    });

    // ── Mic pipeline (always-on) ─────────────────────────────────────────────
    dbg_printf(DBG_MIC, "[main] starting mic pipeline\n");
    Pipeline mic_pipeline(std::move(mic_cfg));
    mic_pipeline.start();

    
    // ── Button ───────────────────────────────────────────────────────────────
    dbg_printf(DBG_BUTTON, "[main] registering button callbacks\n");
    button_driver::ButtonDriver btn;

    btn.registerPressCallback([&]() {
        if (g_button_down.exchange(true)) {
            dbg_printf(DBG_BUTTON, "\n[button] PRESS ignored — already pressed\n");
            std::fflush(stdout);
            return;
        }

        dbg_printf(DBG_BUTTON, "\n[button] PRESS  — interrupt speech, capture ON, mic active, filter reset for new query\n");
        std::fflush(stdout);

        // A new press means the user is asking a new question. Stop any old
        // speech immediately, then clear the old query/filter state.
        speech_aggregator.interrupt();
        const uint64_t new_session = g_session_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        g_target_session_generation.store(0, std::memory_order_release);
        (void)new_session;

        filter.reset();
        speech_aggregator.reset();
        clear_distances();
        mic_armed.store(true);

        controller.start_capture();
        dbg_printf(DBG_CAP, "[cap] capture started\n");
        std::fflush(stdout);
    });

    btn.registerReleaseCallback([&]() {
        if (!g_button_down.exchange(false)) {
            dbg_printf(DBG_BUTTON, "\n[button] RELEASE ignored — button was not pressed\n");
            std::fflush(stdout);
            return;
        }

        dbg_printf(DBG_BUTTON, "\n[button] RELEASE  — capture OFF, mic ignored, filter kept until next press\n");
        std::fflush(stdout);
        mic_armed.store(false);
        controller.stop_capture();
        dbg_printf(DBG_CAP, "[cap] capture stopped\n");
        dbg_printf(DBG_MAIN, "[main] session totals: cam_frames=%llu  ir_frames=%llu  "
                    "hailo_cb=%llu  pairs=%llu\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                    static_cast<unsigned long long>(g_hailo_callbacks.load()),
                    static_cast<unsigned long long>(g_pairs_emitted.load()));
        std::fflush(stdout);
    });
    

    /*
    // ── Button ───────────────────────────────────────────────────────────────
    dbg_printf(DBG_BUTTON, "[main] registering button callbacks\n");
    button_driver::ButtonDriver btn;

    static constexpr uint8_t HARDCODED_OBJECT_ID = 67;  // cell phone

    btn.registerPressCallback([&]() {
        std::printf("\n[button] PRESS  — capture ON, filter armed for cell phone (id=%d)\n",
                    HARDCODED_OBJECT_ID);
        std::fflush(stdout);
        filter.reset();
        mic_armed.store(true);
        controller.start_capture();
        filter.on_intent(HARDCODED_OBJECT_ID);
        dbg_printf(DBG_CAP, "[cap] capture started\n");
        std::fflush(stdout);
    });

    btn.registerReleaseCallback([&]() {
        std::printf("\n[button] RELEASE  — capture OFF, filter kept until next press\n");
        std::fflush(stdout);
        mic_armed.store(false);
        controller.stop_capture();
        dbg_printf(DBG_CAP, "[cap] capture stopped\n");
        dbg_printf(DBG_MAIN, "[main] session totals: cam_frames=%llu  ir_frames=%llu  "
                    "hailo_cb=%llu  pairs=%llu\n",
                    static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                    static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                    static_cast<unsigned long long>(g_hailo_callbacks.load()),
                    static_cast<unsigned long long>(g_pairs_emitted.load()));
        std::fflush(stdout);
    });
    */

    std::printf("\n────────────────────────────────────────────────────\n");
    std::printf(" NorthStar ready.\n");
    std::printf("  - Hold button to capture frames.\n");
    std::printf("  - Speak a registered object name while held to arm filter.\n");
    std::printf("  - Ctrl-C to exit.\n");
    std::printf("────────────────────────────────────────────────────\n\n");
    std::fflush(stdout);

    // ── Main idle loop ───────────────────────────────────────────────────────
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    std::printf("\n[main] shutdown initiated\n");

    dbg_printf(DBG_MAIN, "[main] stopping capture\n");
    controller.stop_capture();

    dbg_printf(DBG_MAIN, "[main] stopping mic pipeline\n");
    mic_pipeline.stop();

    dbg_printf(DBG_MAIN, "[main] shutting down capture controller (will stop queues)\n");
    controller.shutdown();

    dbg_printf(DBG_MAIN, "[main] joining cam thread\n");
    cam_thread.join();

    dbg_printf(DBG_MAIN, "[main] joining ir thread\n");
    ir_thread.join();

    dbg_printf(DBG_MAIN, "[main] stopping Hailo\n");
    hailo.stop();

    dbg_printf(DBG_MAIN, "[main] tearing down GPIO\n");
    gpio::teardownGpio();

    std::printf("\n[main] FINAL: cam_frames=%llu  ir_frames=%llu  "
                "hailo_cb=%llu  pairs=%llu  mic_drops=%zu\n",
                static_cast<unsigned long long>(g_cam_frames_consumed.load()),
                static_cast<unsigned long long>(g_ir_frames_consumed.load()),
                static_cast<unsigned long long>(g_hailo_callbacks.load()),
                static_cast<unsigned long long>(g_pairs_emitted.load()),
                mic_pipeline.queue_drops());

    dbg_printf(DBG_MAIN, "[main] goodbye\n");
    return 0;
}
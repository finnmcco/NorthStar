/*
    latency_bench.cpp — NorthStar residual latency benchmarks
    ══════════════════════════════════════════════════════════
    Measures the four pipeline stages not yet covered by audio_test / inf_bench:

    1. cam_to_hailo    — libcamera capture timestamp → Hailo result callback,
                         broken into: cap→dequeue, dequeue→write_frame, write→callback.
    2. filter_arm      — DetectionFilter::on_intent() → first FilteredInferencePair,
                         in both the pre-buffered case (real-world) and the cold case.
    3. tts_synth       — TTSEngine::synthesise() for four sentence lengths.
    4. pa_latency      — PulseAudio server-reported buffer latency immediately after
                         the first pa_simple_write(), plus round-trip drain timing.

    Build target: latency_bench (see CMakeLists stanza at bottom of this file).

    Usage:
        ./latency_bench [--skip-cam] [--skip-filter] [--skip-tts] [--skip-pa]

    The cam sub-test requires real camera + Hailo hardware and a compiled .hef
    at DEFAULT_HEF_PATH. Pass --skip-cam to run only the software tests.
*/

// ─── Standard ─────────────────────────────────────────────────────────────────
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ─── Project ──────────────────────────────────────────────────────────────────
#include "config.hpp"
#include "capture_controller.hpp"
#include "camera_queue.hpp"
#include "frame_packet.hpp"
#include "hailo8_inference.hpp"
#include "detection_utils.hpp"
#include "detection_filter.hpp"
#include "inference_packet.hpp"
#include "tts.hpp"
#include "speaker.hpp"

// ─── PulseAudio ───────────────────────────────────────────────────────────────
#include <pulse/simple.h>
#include <pulse/error.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool>  g_running{true};
static CaptureController* g_controller_ptr = nullptr;

static void on_sigint(int)
{
    g_running = false;
    if (g_controller_ptr)
        g_controller_ptr->stop_capture();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Timing helpers
// ─────────────────────────────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::nanoseconds;
using Ms    = std::chrono::duration<double, std::milli>;

static uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Ns>(Clock::now().time_since_epoch()).count());
}

static double ns_to_ms(uint64_t ns)
{
    return static_cast<double>(ns) / 1.0e6;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Statistics
// ─────────────────────────────────────────────────────────────────────────────
struct Stats {
    double mean = 0, min = 0, max = 0;
    double p50 = 0, p95 = 0, p99 = 0;
    size_t n = 0;
};

static Stats compute_stats(std::vector<double> v)
{
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    Stats s;
    s.n    = v.size();
    s.min  = v.front();
    s.max  = v.back();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.p50  = v[v.size() * 50 / 100];
    s.p95  = v[v.size() * 95 / 100];
    s.p99  = v[v.size() * 99 / 100];
    return s;
}

static void print_stats_row(const char* label, const Stats& s)
{
    std::printf("  %-30s  n=%-4zu  mean=%6.2f ms  p50=%6.2f  p95=%6.2f  p99=%6.2f"
                "  min=%6.2f  max=%6.2f\n",
                label, s.n, s.mean, s.p50, s.p95, s.p99, s.min, s.max);
}

static void print_header(const char* title)
{
    std::printf("\n╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  %-52s║\n", title);
    std::printf("╚══════════════════════════════════════════════════════╝\n");
}

static void print_separator()
{
    std::printf("  ──────────────────────────────────────────────────────\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  1.  cam_to_hailo
//      libcamera hardware timestamp → Hailo result callback.
//      Sub-stages:
//        A  cap → dequeue   (camera DMA + queue enqueue + consumer pop)
//        B  dequeue → write (consumer scheduling)
//        C  write → callback (Hailo DMA + inference + read-thread scheduling)
//        total = A + B + C
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int CAM_WARMUP = 30;
static constexpr int CAM_FRAMES = 200;

struct CamSample {
    double cap_to_dq_ms  = 0;   // A
    double dq_to_write_ms = 0;  // B
    double write_to_cb_ms = 0;  // C
    double total_ms       = 0;  // A+B+C
};

static void bench_cam_to_hailo()
{
    print_header("1 / 4  cam_to_hailo");
    std::printf("  Frames: %d warmup + %d measured  |  model: %s\n\n",
                CAM_WARMUP, CAM_FRAMES, DEFAULT_HEF_PATH);

    Hailo8Inference hailo(DEFAULT_HEF_PATH);

    // Per-frame correlation: write_start_ns → {cap_ts_ns, dequeue_ns}
    struct FrameCtx {
        uint64_t cap_ns;
        uint64_t dq_ns;
    };
    std::mutex                          ctx_mu;
    std::unordered_map<uint64_t, FrameCtx> ctx_map;  // keyed on write_start_ns
    ctx_map.reserve(256);

    std::vector<CamSample> samples;
    samples.reserve(CAM_FRAMES);
    std::mutex samples_mu;

    hailo.register_callback(
        [&](uint8_t   /*camera_id*/,
            uint64_t  write_start_ns,          // echoed from write_frame()
            std::vector<std::vector<uint8_t>> raw_outputs)
        {
            const uint64_t cb_ns = now_ns();

            // Parse so we don't stall the Hailo pipeline.
            (void)parse_detections(raw_outputs);

            std::lock_guard<std::mutex> lk(ctx_mu);
            auto it = ctx_map.find(write_start_ns);
            if (it == ctx_map.end()) return;

            CamSample s;
            s.cap_to_dq_ms   = ns_to_ms(it->second.dq_ns  - it->second.cap_ns);
            s.dq_to_write_ms = ns_to_ms(write_start_ns     - it->second.dq_ns);
            s.write_to_cb_ms = ns_to_ms(cb_ns              - write_start_ns);
            s.total_ms       = ns_to_ms(cb_ns              - it->second.cap_ns);
            ctx_map.erase(it);

            std::lock_guard<std::mutex> slk(samples_mu);
            samples.push_back(s);
        });

    if (!hailo.initialize()) {
        std::fprintf(stderr, "  ERROR: Hailo init failed — is the Hailo8 plugged in?\n");
        return;
    }

    CaptureController controller;
    g_controller_ptr = &controller;
    controller.start_capture();

    auto& cam_q = controller.get_cam_queue();

    int warmup = 0;
    int measured = 0;

    while (g_running && measured < CAM_FRAMES) {
        auto pkt = cam_q.pop();
        if (!pkt) break;

        const uint64_t dq_ns  = now_ns();
        const uint64_t cap_ns = pkt->timestamp_us * 1000ULL;   // libcamera ns

        if (warmup < CAM_WARMUP) {
            ++warmup;
            if (warmup == CAM_WARMUP)
                std::printf("  Warmup done — measuring...\n");
            continue;
        }

        const uint64_t write_start = now_ns();

        {
            std::lock_guard<std::mutex> lk(ctx_mu);
            ctx_map[write_start] = {cap_ns, dq_ns};
        }

        hailo.write_frame(pkt->data.data(), pkt->camera_id, write_start);
        ++measured;
    }

    controller.stop_capture();
    g_controller_ptr = nullptr;

    // Give the Hailo read thread a moment to drain outstanding callbacks.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    hailo.stop();

    // ── Print results ─────────────────────────────────────────────────────────
    if (samples.empty()) {
        std::printf("  No samples collected.\n");
        return;
    }

    auto extract = [&](auto fn) {
        std::vector<double> v;
        v.reserve(samples.size());
        for (const auto& s : samples) v.push_back(fn(s));
        return compute_stats(v);
    };

    print_separator();
    print_stats_row("cap → dequeue  (A)",
        extract([](const CamSample& s){ return s.cap_to_dq_ms; }));
    print_stats_row("dequeue → write_frame  (B)",
        extract([](const CamSample& s){ return s.dq_to_write_ms; }));
    print_stats_row("write_frame → callback (C = Hailo)",
        extract([](const CamSample& s){ return s.write_to_cb_ms; }));
    print_separator();
    print_stats_row("TOTAL  (A+B+C)",
        extract([](const CamSample& s){ return s.total_ms; }));
}

// ─────────────────────────────────────────────────────────────────────────────
//  2.  filter_arm
//      How long from on_intent() to the first FilteredInferencePair callback?
//
//      Case A — pre-buffered (realistic):
//        Push N_PRE_PAIRS of synthetic cam0+cam1 packets containing the target
//        object into the filter while unarmed.  Then call on_intent() and
//        measure the time to the first emitted pair.
//        Expected: near-zero (backlog drains synchronously inside on_intent()).
//
//      Case B — cold arm (buffer empty, packets arrive after intent):
//        Call on_intent() on a fresh filter, then push one cam0 and one cam1
//        packet.  Measure time from first push to first pair callback.
//        Expected: sub-millisecond (mutex + try_pair CPU work only).
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int  FILTER_TRIALS  = 100;
static constexpr int  N_PRE_PAIRS    = 30;    // ≈ 1 s of frames at 30 fps
static constexpr uint8_t TARGET_ID   = 41;    // COCO "cup"
static constexpr uint64_t FRAME_NS   = 33'333'333ULL;  // ~30 fps

static InferencePacket make_packet(uint8_t cam_id, uint64_t ts_ns)
{
    InferencePacket pkt;
    pkt.camera_id  = cam_id;
    pkt.timestamp  = ts_ns;
    Detection det;
    det.object_id  = TARGET_ID;
    det.confidence = 0.9f;
    det.box        = {0.3f, 0.3f, 0.7f, 0.7f};
    pkt.detections.push_back(det);
    return pkt;
}

static void bench_filter_arm()
{
    print_header("2 / 4  filter_arm");
    std::printf("  Trials: %d  |  pre-buffer fill: %d pairs  |  target COCO id: %d\n\n",
                FILTER_TRIALS, N_PRE_PAIRS, TARGET_ID);

    // ── Case A: pre-buffered ──────────────────────────────────────────────────
    std::vector<double> a_samples;
    a_samples.reserve(FILTER_TRIALS);

    for (int trial = 0; trial < FILTER_TRIALS; ++trial) {
        std::mutex        mu;
        std::condition_variable cv;
        bool              fired = false;
        uint64_t          cb_ns = 0;

        DetectionFilter filter([&](FilteredInferencePair /*pair*/) {
            cb_ns = now_ns();
            std::unique_lock<std::mutex> lk(mu);
            fired = true;
            cv.notify_one();
        });

        // Fill the pre-buffer with alternating cam0/cam1 pairs.
        // Timestamps are staggered by one frame period so they look real.
        const uint64_t base_ts = now_ns() - static_cast<uint64_t>(N_PRE_PAIRS) * FRAME_NS;
        for (int i = 0; i < N_PRE_PAIRS; ++i) {
            const uint64_t ts = base_ts + static_cast<uint64_t>(i) * FRAME_NS;
            filter.on_inference_packet(make_packet(0, ts));
            filter.on_inference_packet(make_packet(1, ts));
        }

        // Arm and measure.
        const uint64_t arm_ns = now_ns();
        filter.on_intent(TARGET_ID);

        // on_intent() drains the backlog synchronously, so the callback fires
        // before on_intent() returns.  But we check anyway with a short timeout.
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(50), [&]{ return fired; });
        }

        if (fired)
            a_samples.push_back(ns_to_ms(cb_ns - arm_ns));
        else
            std::printf("  trial %d: no pair emitted within 50 ms (buffer may be empty)\n", trial);
    }

    // ── Case B: cold arm ──────────────────────────────────────────────────────
    std::vector<double> b_samples;
    b_samples.reserve(FILTER_TRIALS);

    for (int trial = 0; trial < FILTER_TRIALS; ++trial) {
        std::mutex              mu;
        std::condition_variable cv;
        bool                    fired = false;
        uint64_t                cb_ns = 0;

        DetectionFilter filter([&](FilteredInferencePair /*pair*/) {
            cb_ns = now_ns();
            std::unique_lock<std::mutex> lk(mu);
            fired = true;
            cv.notify_one();
        });

        filter.on_intent(TARGET_ID);  // arm with empty buffer

        const uint64_t push_ns = now_ns();
        const uint64_t ts      = push_ns;

        // Push cam0 then cam1 — try_pair() will emit on the cam1 push.
        filter.on_inference_packet(make_packet(0, ts));
        filter.on_inference_packet(make_packet(1, ts));

        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(50), [&]{ return fired; });
        }

        if (fired)
            b_samples.push_back(ns_to_ms(cb_ns - push_ns));
        else
            std::printf("  trial %d (cold): no pair emitted within 50 ms\n", trial);
    }

    // ── Print results ─────────────────────────────────────────────────────────
    print_separator();
    if (!a_samples.empty())
        print_stats_row("Case A: on_intent() → pair  [pre-buffered]",
                        compute_stats(a_samples));
    else
        std::printf("  Case A: no data\n");

    if (!b_samples.empty())
        print_stats_row("Case B: first push → pair   [cold arm]",
                        compute_stats(b_samples));
    else
        std::printf("  Case B: no data\n");

    print_separator();

    // Sanity check: confirm buffer depth was correct going into on_intent().
    {
        DetectionFilter probe([](FilteredInferencePair){});
        const uint64_t ts = now_ns();
        for (int i = 0; i < N_PRE_PAIRS; ++i) {
            probe.on_inference_packet(make_packet(0, ts));
            probe.on_inference_packet(make_packet(1, ts));
        }
        std::printf("  Buffer depth before arm: %zu packets  (MAX_PRE_BUFFER = %zu)\n",
                    probe.buffer_depth(), DetectionFilter::MAX_PRE_BUFFER);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  3.  tts_synth
//      Wall time for TTSEngine::synthesise() across four sentence lengths.
//
//      The Python pip Piper flushes its output buffer only when stdin closes,
//      so each TTSEngine instance can serve exactly one synthesise() call —
//      which is exactly how it is used in production (once per button press).
//      The bench constructs a fresh TTSEngine per run to match that reality.
//
//      Each run measures: subprocess fork + ONNX model load + synthesis + read.
//      This is the true end-to-end cost the user experiences per button press.
//
//      Override paths via environment variables:
//        PIPER_BIN    (default: venv piper)
//        PIPER_MODEL  (default: en_US-lessac-medium.onnx)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int N_TTS_RUNS = 5;  // per sentence; all runs are cold starts

struct TtsSentence {
    const char* label;
    const char* text;
};

static const TtsSentence TTS_SENTENCES[] = {
    {
        "short  (~5 words)",
        "There is a cup."
    },
    {
        "medium (~12 words)",
        "The cup is about fifty centimetres directly in front of you."
    },
    {
        "long   (~20 words)",
        "There is a cup approximately eighty centimetres away, slightly to your left, "
        "at around twenty-two degrees Celsius."
    },
    {
        "full   (~30 words)",
        "I can see a cup about ninety centimetres in front of you and slightly to the right. "
        "The surface temperature is approximately twenty-four degrees Celsius. "
        "Please reach carefully to your right."
    },
};

static void bench_tts_synth()
{
    print_header("3 / 4  tts_synth");

    const char* piper_bin   = std::getenv("PIPER_BIN");
    const char* piper_model = std::getenv("PIPER_MODEL");
    if (!piper_bin)   piper_bin   = "/home/dst5/NSJamie/NorthStar/.venv/bin/piper";
    if (!piper_model) piper_model = "/home/dst5/NSJamie/NorthStar/model_inf/en_US-lessac-medium.onnx";

    std::printf("  Piper : %s\n", piper_bin);
    std::printf("  Model : %s\n", piper_model);
    std::printf("  Runs  : %d per sentence — each is a fresh cold start (fork+load+synth)\n\n",
                N_TTS_RUNS);

    // Quick sanity check before the full run.
    {
        TTSEngine probe(piper_bin, piper_model, 22050);
        if (!probe.ready()) {
            std::printf("  SKIPPED — TTSEngine not ready: %s\n",
                        probe.error_message().c_str());
            return;
        }
    }

    for (const auto& sentence : TTS_SENTENCES) {
        std::vector<double> times;
        times.reserve(N_TTS_RUNS);

        for (int run = 0; run < N_TTS_RUNS; ++run) {
            // Fresh TTSEngine each run = one full cold-start, matching production.
            const auto t0 = Clock::now();
            TTSEngine tts(piper_bin, piper_model, 22050);
            auto pcm = tts.ready() ? tts.synthesise(sentence.text)
                                   : std::vector<int16_t>{};
            const double ms = Ms(Clock::now() - t0).count();

            if (pcm.empty()) {
                std::fprintf(stderr, "  ERROR run %d: %s\n",
                             run + 1, tts.error_message().c_str());
                continue;
            }

            const double audio_s = static_cast<double>(pcm.size())
                                   / static_cast<double>(tts.sample_rate());
            times.push_back(ms);
            std::printf("  [%s]  run %d: %.0f ms  audio: %.2f s  RTF: %.2f\n",
                        sentence.label, run + 1, ms, audio_s,
                        ms / (audio_s * 1000.0));
        }

        if (!times.empty()) {
            const auto s = compute_stats(times);
            print_stats_row(sentence.label, s);
        }
        std::printf("\n");
    }

    print_separator();
    std::printf("  RTF = cold-start wall time / audio duration.\n"
                "  This reflects real user-perceived latency per button press.\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  4.  pa_latency
//      PulseAudio server-reported playback latency.
//
//      Method A — pa_simple_get_latency():
//        Open a stream, write one chunk of silence, immediately read back
//        the latency figure.  This is the PA/PipeWire buffer depth (time
//        from "data written" to "speaker emits first sample").
//
//      Method B — drain round-trip:
//        Write N seconds of silence, time how long pa_simple_drain() takes
//        to complete.  Expected ≈ N s.  Excess = buffer latency added on top.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr uint32_t PA_RATE     = 22050;
static constexpr uint8_t  PA_CHANNELS = 1;
static constexpr int      PA_DRAIN_S  = 1;    // seconds of silence for drain test

static void bench_pa_latency()
{
    print_header("4 / 4  pa_latency");

    const pa_sample_spec spec {
        .format   = PA_SAMPLE_S16LE,
        .rate     = PA_RATE,
        .channels = PA_CHANNELS
    };

    int pa_err = 0;

    // ── Method A: get_latency immediately after first write ───────────────────
    {
        pa_simple* pa = pa_simple_new(
            nullptr,            // server (default)
            "latency_bench",    // app name
            PA_STREAM_PLAYBACK,
            nullptr,            // sink (default)
            "pa_latency_probe",
            &spec,
            nullptr,            // channel map (default)
            nullptr,            // buffering attributes (default)
            &pa_err
        );

        if (!pa) {
            std::fprintf(stderr, "  pa_simple_new failed: %s — is PulseAudio/PipeWire running?\n",
                         pa_strerror(pa_err));
            return;
        }

        // Write one chunk of silence (100 ms).
        const size_t chunk_samples = PA_RATE / 10;
        const std::vector<int16_t> silence(chunk_samples, 0);

        if (pa_simple_write(pa, silence.data(),
                            silence.size() * sizeof(int16_t), &pa_err) < 0) {
            std::fprintf(stderr, "  pa_simple_write failed: %s\n", pa_strerror(pa_err));
            pa_simple_free(pa);
            return;
        }

        // Collect latency readings immediately after the write.
        std::vector<double> readings;
        constexpr int N_READS = 20;
        readings.reserve(N_READS);

        for (int i = 0; i < N_READS; ++i) {
            const pa_usec_t lat = pa_simple_get_latency(pa, &pa_err);
            if (pa_err != 0) {
                std::fprintf(stderr, "  pa_simple_get_latency error: %s\n",
                             pa_strerror(pa_err));
                break;
            }
            readings.push_back(static_cast<double>(lat) / 1000.0);  // µs → ms
            pa_err = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        pa_simple_drain(pa, nullptr);
        pa_simple_free(pa);

        if (!readings.empty()) {
            const auto s = compute_stats(readings);
            print_separator();
            print_stats_row("Method A: get_latency  (ms)", s);
            std::printf("  (= PA/PipeWire buffer depth — time from write() to speaker output)\n");
        }
    }

    // ── Method B: drain round-trip ────────────────────────────────────────────
    {
        pa_simple* pa = pa_simple_new(
            nullptr, "latency_bench", PA_STREAM_PLAYBACK, nullptr,
            "pa_drain_probe", &spec, nullptr, nullptr, &pa_err);

        if (!pa) {
            std::fprintf(stderr, "  pa_simple_new (drain probe) failed: %s\n",
                         pa_strerror(pa_err));
            return;
        }

        // Write PA_DRAIN_S seconds of silence.
        const size_t n_samples = PA_RATE * PA_DRAIN_S;
        const std::vector<int16_t> silence(n_samples, 0);

        const auto write_start = Clock::now();
        if (pa_simple_write(pa, silence.data(),
                            silence.size() * sizeof(int16_t), &pa_err) < 0) {
            std::fprintf(stderr, "  pa_simple_write failed: %s\n", pa_strerror(pa_err));
            pa_simple_free(pa);
            return;
        }

        // drain() blocks until PA has played back everything written.
        if (pa_simple_drain(pa, &pa_err) < 0) {
            std::fprintf(stderr, "  pa_simple_drain failed: %s\n", pa_strerror(pa_err));
            pa_simple_free(pa);
            return;
        }
        const double drain_ms = Ms(Clock::now() - write_start).count();

        pa_simple_free(pa);

        const double audio_ms    = PA_DRAIN_S * 1000.0;
        const double overhead_ms = drain_ms - audio_ms;

        print_separator();
        std::printf("  Method B: drain round-trip\n");
        std::printf("    Audio content  : %.0f ms\n",  audio_ms);
        std::printf("    Total wall time: %.1f ms\n",  drain_ms);
        std::printf("    Buffer overhead: %.1f ms  (= extra latency above audio duration)\n",
                    overhead_ms);
    }

    print_separator();
    std::printf("  Tip: to reduce PA latency, set PA_STREAM_ADJUST_LATENCY in Speaker::open()\n"
                "  and request a smaller tlength (e.g. 50 ms) in pa_buffer_attr.\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
static bool has_flag(int argc, char* argv[], const char* flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

int main(int argc, char* argv[])
{
    std::signal(SIGINT, on_sigint);

    const bool skip_cam    = has_flag(argc, argv, "--skip-cam");
    const bool skip_filter = has_flag(argc, argv, "--skip-filter");
    const bool skip_tts    = has_flag(argc, argv, "--skip-tts");
    const bool skip_pa     = has_flag(argc, argv, "--skip-pa");

    std::printf("\nNorthStar latency bench\n");
    std::printf("═══════════════════════\n");
    if (skip_cam)    std::printf("  --skip-cam    active\n");
    if (skip_filter) std::printf("  --skip-filter active\n");
    if (skip_tts)    std::printf("  --skip-tts    active\n");
    if (skip_pa)     std::printf("  --skip-pa     active\n");

    if (!skip_cam)    bench_cam_to_hailo();
    if (!skip_filter) bench_filter_arm();
    if (!skip_tts)    bench_tts_synth();
    if (!skip_pa)     bench_pa_latency();

    std::printf("\nDone.\n\n");
    return 0;
}
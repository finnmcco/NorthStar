/*
    test_detection_filter.cpp
    ══════════════════════════
    Pure-logic unit tests for DetectionFilter.  No Hailo hardware, no cameras,
    no Vosk — only synthetic InferencePackets and direct callback calls.

    Each test is self-contained and prints [PASS] or [FAIL] with a reason.

    Build (from project root, adjust include path as needed):
        g++ -std=c++17 -I include \
            src/detection_filter.cpp \
            src/test_detection_filter.cpp \
            -o test_detection_filter -pthread
    Run:
        ./test_detection_filter
    Exit code 0 = all tests passed.

    ─────────────────────────────────────────────────────────────────────────────
    Test inventory
    ─────────────────────────────────────────────────────────────────────────────
      1.  Unarmed state on construction
      2.  Packets buffered before intent, buffer_depth reported correctly
      3.  on_intent arms the filter, clears buffer
      4.  Pre-buffer overflow: oldest packets dropped, newest retained
      5.  Packets with wrong object_id produce no output
      6.  Basic stereo pair: cam0 then cam1 with target
      7.  Reverse order pair: cam1 first, then cam0
      8.  Timestamp too far apart: no pair
      9.  One camera missing target: no pair
     10.  Multiple detections in frame: all target detections passed through, non-target excluded
     11.  Pre-buffer replay produces stereo pairs
     12.  Sequential pairs all emitted
     13.  Reset clears all state
     14.  Post-reset packets buffered again (not processed)
     15.  Stale pending cam0 overwritten by newer cam0
*/

#include "detection_filter.hpp"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Test harness helpers
// ─────────────────────────────────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_passed = 0;

static void report(const std::string& name, bool pass,
                   const std::string& detail = "")
{
    ++g_tests_run;
    if (pass) {
        ++g_tests_passed;
        std::printf("  [PASS]  %s\n", name.c_str());
    } else {
        std::printf("  [FAIL]  %s%s\n",
                    name.c_str(),
                    detail.empty() ? "" : ("  — " + detail).c_str());
    }
}

// ── Packet factories ─────────────────────────────────────────────────────────

static InferencePacket make_packet(uint8_t camera_id, uint64_t ts_ns,
                                   uint8_t object_id, float conf = 0.9f)
{
    InferencePacket pkt;
    pkt.camera_id  = camera_id;
    pkt.timestamp  = ts_ns;
    Detection d;
    d.object_id   = object_id;
    d.confidence  = conf;
    d.box = { 0.1f, 0.1f, 0.5f, 0.5f };
    pkt.detections = { d };
    return pkt;
}

static InferencePacket make_empty_packet(uint8_t camera_id, uint64_t ts_ns)
{
    InferencePacket pkt;
    pkt.camera_id = camera_id;
    pkt.timestamp = ts_ns;
    // No detections.
    return pkt;
}

// Timestamps for synthetic 30fps stereo frames (nanoseconds).
static constexpr uint64_t FRAME_NS   = 33'333'333ULL;   // ~30fps
static constexpr uint64_t SYNC_DELTA = 500'000ULL;       // 0.5ms — well within 2ms tolerance
static constexpr uint64_t TARGET_ID  = 15;               // COCO "cat"

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

// 1. Unarmed state on construction
static void test_01_unarmed_on_construction()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });

    report("01 unarmed on construction",
           !f.is_armed() && f.buffer_depth() == 0 && fired == 0);
}

// 2. Packets buffered before intent
static void test_02_buffering_before_intent()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });

    f.on_inference_packet(make_packet(0, 1000, TARGET_ID));
    f.on_inference_packet(make_packet(1, 1200, TARGET_ID));
    f.on_inference_packet(make_packet(0, FRAME_NS, TARGET_ID));

    bool ok = !f.is_armed()
           && f.buffer_depth() == 3
           && fired == 0;
    report("02 packets buffered before intent", ok,
           "depth=" + std::to_string(f.buffer_depth()) + " fired=" + std::to_string(fired));
}

// 3. on_intent arms and drains the buffer (packets too old to form pairs)
static void test_03_on_intent_arms_and_clears_buffer()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });

    // Push 3 packets that are NOT close enough to pair with each other.
    for (int i = 0; i < 3; ++i)
        f.on_inference_packet(make_packet(0, static_cast<uint64_t>(i) * FRAME_NS, TARGET_ID));

    f.on_intent(TARGET_ID);

    bool ok = f.is_armed()
           && f.target_object_id() == TARGET_ID
           && f.buffer_depth() == 0;   // buffer was drained
    report("03 on_intent arms and clears buffer", ok,
           "armed=" + std::to_string(f.is_armed()) +
           " depth=" + std::to_string(f.buffer_depth()));
}

// 4. Pre-buffer overflow: MAX_PRE_BUFFER is respected, oldest dropped
static void test_04_pre_buffer_overflow()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });

    const std::size_t OVER = DetectionFilter::MAX_PRE_BUFFER + 10;
    for (std::size_t i = 0; i < OVER; ++i)
        f.on_inference_packet(make_packet(0, i * 1000ULL, TARGET_ID));

    bool ok = f.buffer_depth() == DetectionFilter::MAX_PRE_BUFFER;
    report("04 pre-buffer overflow capped",
           ok, "depth=" + std::to_string(f.buffer_depth()));
}

// 5. Packets with wrong object_id produce no output
static void test_05_wrong_object_id_filtered()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });
    f.on_intent(TARGET_ID);

    constexpr uint8_t WRONG_ID = 7;
    f.on_inference_packet(make_packet(0, 1'000'000ULL, WRONG_ID));
    f.on_inference_packet(make_packet(1, 1'500'000ULL, WRONG_ID));

    report("05 wrong object_id filtered out",
           f.is_armed() && fired == 0,
           "fired=" + std::to_string(fired));
}

// 6. Basic stereo pair: cam0 arrives first, cam1 follows within tolerance
static void test_06_basic_stereo_pair()
{
    int fired = 0;
    FilteredInferencePair result{};
    DetectionFilter f([&](FilteredInferencePair p) { ++fired; result = p; });
    f.on_intent(TARGET_ID);

    const uint64_t t0 = 1'000'000ULL;
    const uint64_t t1 = t0 + SYNC_DELTA;

    f.on_inference_packet(make_packet(0, t0, TARGET_ID, 0.85f));
    f.on_inference_packet(make_packet(1, t1, TARGET_ID, 0.90f));

    bool ok = fired == 1
           && result.object_id     == TARGET_ID
           && result.timestamp_avg == (t0 + t1) / 2;
    report("06 basic stereo pair cam0→cam1", ok,
           "fired=" + std::to_string(fired));
}

// 7. Reverse order: cam1 first, cam0 follows within tolerance
static void test_07_reverse_order_pair()
{
    int fired = 0;
    FilteredInferencePair result{};
    DetectionFilter f([&](FilteredInferencePair p) { ++fired; result = p; });
    f.on_intent(TARGET_ID);

    const uint64_t t1 = 2'000'000ULL;
    const uint64_t t0 = t1 + SYNC_DELTA;

    f.on_inference_packet(make_packet(1, t1, TARGET_ID));
    f.on_inference_packet(make_packet(0, t0, TARGET_ID));

    bool ok = fired == 1
           && result.object_id == TARGET_ID;
    report("07 reverse order pair cam1→cam0", ok,
           "fired=" + std::to_string(fired));
}

// 8. Timestamps too far apart: no pair
static void test_08_timestamp_too_far()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });
    f.on_intent(TARGET_ID);

    const uint64_t t0 = 1'000'000ULL;
    // 5ms gap — well outside the 2ms tolerance
    const uint64_t t1 = t0 + 5'000'000ULL;

    f.on_inference_packet(make_packet(0, t0, TARGET_ID));
    f.on_inference_packet(make_packet(1, t1, TARGET_ID));

    report("08 timestamps too far apart — no pair",
           fired == 0, "fired=" + std::to_string(fired));
}

// 9. One camera missing the target: no pair
static void test_09_one_camera_missing_target()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });
    f.on_intent(TARGET_ID);

    const uint64_t t = 1'000'000ULL;
    f.on_inference_packet(make_packet(0, t, TARGET_ID));       // cam0: has target
    f.on_inference_packet(make_empty_packet(1, t + SYNC_DELTA)); // cam1: no detections

    report("09 one camera missing target — no pair",
           fired == 0, "fired=" + std::to_string(fired));
}

// 10. Multiple detections in frame — all target detections passed through
static void test_10_all_target_detections_passed()
{
    FilteredInferencePair result{};
    DetectionFilter f([&](FilteredInferencePair p) { result = p; });
    f.on_intent(TARGET_ID);

    // cam0: two detections of the target + one of a different class
    InferencePacket cam0;
    cam0.camera_id = 0;
    cam0.timestamp = 1'000'000ULL;
    Detection d1; d1.object_id = TARGET_ID; d1.confidence = 0.50f;
    d1.box = { 0.0f, 0.0f, 0.2f, 0.2f };
    Detection d2; d2.object_id = TARGET_ID; d2.confidence = 0.95f;
    d2.box = { 0.4f, 0.4f, 0.9f, 0.9f };
    Detection d3; d3.object_id = 7;         d3.confidence = 0.80f;  // wrong class
    d3.box = { 0.1f, 0.1f, 0.3f, 0.3f };
    cam0.detections = { d1, d2, d3 };

    f.on_inference_packet(cam0);
    f.on_inference_packet(make_packet(1, cam0.timestamp + SYNC_DELTA, TARGET_ID));

    // Both target detections should be in cam0_detections, the wrong class excluded.
    bool ok = result.cam0_detections.size() == 2
           && result.cam1_detections.size() == 1;

    report("10 all target detections passed, non-target excluded",
           ok,
           "cam0=" + std::to_string(result.cam0_detections.size()) +
           " cam1=" + std::to_string(result.cam1_detections.size()));
}

// 11. Pre-buffer replay produces stereo pairs
static void test_11_pre_buffer_replay_pairs()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });

    // Push a valid synchronous pair BEFORE intent.
    const uint64_t t0 = 1'000'000ULL;
    f.on_inference_packet(make_packet(0, t0, TARGET_ID));
    f.on_inference_packet(make_packet(1, t0 + SYNC_DELTA, TARGET_ID));

    // Nothing fired yet.
    assert(fired == 0);

    // Now arm — backlog should replay and the pair should emit.
    f.on_intent(TARGET_ID);

    report("11 pre-buffer replay produces pair",
           fired == 1, "fired=" + std::to_string(fired));
}

// 12. Sequential pairs: two pairs in a row both emit
static void test_12_sequential_pairs()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });
    f.on_intent(TARGET_ID);

    for (int i = 0; i < 2; ++i) {
        const uint64_t t = static_cast<uint64_t>(i) * FRAME_NS + 1'000'000ULL;
        f.on_inference_packet(make_packet(0, t, TARGET_ID));
        f.on_inference_packet(make_packet(1, t + SYNC_DELTA, TARGET_ID));
    }

    report("12 sequential pairs both emitted",
           fired == 2, "fired=" + std::to_string(fired));
}

// 13. reset() clears all state
static void test_13_reset_clears_state()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });
    f.on_intent(TARGET_ID);

    // Partial state: cam0 stored, cam1 not yet arrived.
    f.on_inference_packet(make_packet(0, 1'000'000ULL, TARGET_ID));

    f.reset();

    bool ok = !f.is_armed()
           && f.buffer_depth()    == 0
           && f.target_object_id() == 0;
    report("13 reset clears all state", ok,
           "armed=" + std::to_string(f.is_armed()) +
           " depth=" + std::to_string(f.buffer_depth()));
}

// 14. After reset, packets are buffered again (filter is unarmed)
static void test_14_post_reset_buffering()
{
    int fired = 0;
    DetectionFilter f([&](FilteredInferencePair) { ++fired; });
    f.on_intent(TARGET_ID);
    f.reset();

    f.on_inference_packet(make_packet(0, 1'000'000ULL, TARGET_ID));
    f.on_inference_packet(make_packet(1, 1'500'000ULL, TARGET_ID));

    bool ok = !f.is_armed()
           && f.buffer_depth() == 2
           && fired == 0;
    report("14 post-reset packets buffered (not processed)", ok,
           "depth=" + std::to_string(f.buffer_depth()) + " fired=" + std::to_string(fired));
}

// 15. Stale pending cam0 is overwritten by a newer cam0
static void test_15_stale_pending_overwritten()
{
    int fired = 0;
    FilteredInferencePair result{};
    DetectionFilter f([&](FilteredInferencePair p) { ++fired; result = p; });
    f.on_intent(TARGET_ID);

    // Frame 1: cam0 stores as pending, cam1 arrives OUTSIDE tolerance (no pair).
    const uint64_t t1_cam0 = 1'000'000ULL;
    const uint64_t t1_cam1 = t1_cam0 + 10'000'000ULL;   // 10ms — too far
    f.on_inference_packet(make_packet(0, t1_cam0, TARGET_ID));
    f.on_inference_packet(make_packet(1, t1_cam1, TARGET_ID));
    assert(fired == 0);

    // Frame 2: fresh synchronous pair should still work.
    const uint64_t t2_cam0 = 2 * FRAME_NS;
    const uint64_t t2_cam1 = t2_cam0 + SYNC_DELTA;
    f.on_inference_packet(make_packet(0, t2_cam0, TARGET_ID));
    f.on_inference_packet(make_packet(1, t2_cam1, TARGET_ID));

    report("15 stale pending overwritten, fresh pair fires",
           fired == 1, "fired=" + std::to_string(fired));
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    std::printf("══════════════════════════════════════════\n");
    std::printf("  TEST SUITE: DetectionFilter\n");
    std::printf("══════════════════════════════════════════\n\n");

    test_01_unarmed_on_construction();
    test_02_buffering_before_intent();
    test_03_on_intent_arms_and_clears_buffer();
    test_04_pre_buffer_overflow();
    test_05_wrong_object_id_filtered();
    test_06_basic_stereo_pair();
    test_07_reverse_order_pair();
    test_08_timestamp_too_far();
    test_09_one_camera_missing_target();
    test_10_all_target_detections_passed();
    test_11_pre_buffer_replay_pairs();
    test_12_sequential_pairs();
    test_13_reset_clears_state();
    test_14_post_reset_buffering();
    test_15_stale_pending_overwritten();

    std::printf("\n══════════════════════════════════════════\n");
    std::printf("  Results: %d / %d passed\n", g_tests_passed, g_tests_run);
    std::printf("══════════════════════════════════════════\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
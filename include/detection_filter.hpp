#pragma once

/*
    detection_filter.hpp  —  NorthStar stereo intent filter
    ══════════════════════════════════════════════════════════

    Sits between two upstream sources:

        IntentDetector / Pipeline  ──→  on_intent(object_id)
        Hailo read thread          ──→  on_inference_packet(pkt)

    State machine
    ─────────────
        UNARMED  ──[on_intent]──→  ARMED
            • packets arriving before intent are buffered (up to MAX_PRE_BUFFER)
            • on arm, the buffer is drained and replayed in arrival order

        ARMED
            • each InferencePacket is immediately filtered to target_object_id_
            • filtered packets are paired cam0 ↔ cam1 by timestamp proximity
            • complete pairs are emitted via FilteredPairCallback

    Pairing rules
    ─────────────
        • A cam0 and cam1 packet form a pair iff:
            1. |cam0.timestamp − cam1.timestamp| < PAIR_TOLERANCE_NS
            2. Both packets contain at least one detection of target_object_id_
        • When multiple detections of the target exist in one frame, the
          highest-confidence bounding box is selected.
        • Packets without the target are discarded; they do not consume a
          pending slot from the opposite camera.

    Thread safety
    ─────────────
        on_intent() and on_inference_packet() are fully re-entrant and may be
        called concurrently from different threads.
        on_pair_ is called with the internal mutex RELEASED to prevent
        re-entrancy deadlocks if the callback calls back into this class.
*/

#include "inference_packet.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>


// ─────────────────────────────────────────────────────────────────────────────
//  FilteredInferencePair  —  output of the filter
// ─────────────────────────────────────────────────────────────────────────────

struct FilteredInferencePair {
    uint64_t    timestamp_avg = 0;   // (cam0_ts + cam1_ts) / 2, nanoseconds
    uint8_t     object_id     = 0;   // COCO class index (0–79)
    BoundingBox bbox_cam0;           // highest-confidence hit in cam0 frame
    BoundingBox bbox_cam1;           // highest-confidence hit in cam1 frame
};

using FilteredPairCallback = std::function<void(FilteredInferencePair)>;


// ─────────────────────────────────────────────────────────────────────────────
//  DetectionFilter
// ─────────────────────────────────────────────────────────────────────────────

class DetectionFilter {
public:
    // Inference packets buffered while waiting for the first intent signal.
    static constexpr std::size_t MAX_PRE_BUFFER    = 64;

    // Maximum nanosecond difference for a cam0/cam1 pair to count as synchronous.
    static constexpr uint64_t    PAIR_TOLERANCE_NS = 2'000'000ULL;   // 2 ms

    // ── Construction ───────────────────────────────────────────────────────

    // on_pair is called (with the internal mutex released) whenever a
    // complete, filtered stereo pair is ready.  Must not be null.
    explicit DetectionFilter(FilteredPairCallback on_pair);

    DetectionFilter(const DetectionFilter&)            = delete;
    DetectionFilter& operator=(const DetectionFilter&) = delete;

    // ── Upstream callbacks ─────────────────────────────────────────────────

    // Called by the intent subsystem when a target word is recognised.
    // object_id is a COCO class index in [0, 79].
    // Arms the filter and replays any buffered packets.
    void on_intent(uint8_t object_id);

    // Called by the Hailo inference read thread for every processed frame.
    // Thread-safe with respect to on_intent().
    void on_inference_packet(InferencePacket pkt);

    // ── Inspection ─────────────────────────────────────────────────────────

    bool        is_armed()         const noexcept;
    uint8_t     target_object_id() const noexcept;
    std::size_t buffer_depth()     const noexcept;  // packets in pre-intent buffer

    // ── Lifecycle ──────────────────────────────────────────────────────────

    // Reset to the UNARMED state: clears the buffer, pending stereo slots,
    // target id, and armed flag.  Safe to call from any thread.
    void reset();

private:
    FilteredPairCallback on_pair_;

    mutable std::mutex           mutex_;
    bool                         armed_            = false;
    uint8_t                      target_object_id_ = 0;
    std::deque<InferencePacket>  pre_buffer_;
    std::optional<InferencePacket> pending_cam0_;
    std::optional<InferencePacket> pending_cam1_;

    // Returns the highest-confidence bounding box for target_object_id_ in pkt,
    // or nullopt if no matching detection exists.
    // Caller must hold mutex_.
    std::optional<BoundingBox> find_target_box(const InferencePacket& pkt) const;

    // Attempt to pair pkt with a buffered partner from the opposite camera.
    // Returns a complete pair on success; updates pending state.
    // Caller must hold mutex_.
    std::optional<FilteredInferencePair> try_pair(const InferencePacket& pkt);

    static bool timestamps_close(uint64_t a, uint64_t b) noexcept;
};

#pragma once

/*
    detection_filter.hpp  —  NorthStar stereo intent filter

    UNARMED  ──[on_intent]──→  ARMED
        • packets arriving before intent are buffered (up to MAX_PRE_BUFFER)
        • on arm, the buffer is drained and replayed in arrival order

    ARMED
        • each InferencePacket is filtered to target_object_id_
        • qualifying packets are paired cam0 ↔ cam1 by timestamp proximity
        • both cameras must contain the target for a pair to emit
        • complete pairs are emitted via FilteredPairCallback

    Thread safety:  on_intent() and on_inference_packet() may be called
    concurrently.  on_pair_ is invoked with the mutex released.
*/

#include "inference_packet.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>


struct FilteredInferencePair {
    uint64_t               timestamp_avg  = 0;   // (cam0_ts + cam1_ts) / 2, ns
    uint8_t                object_id      = 0;   // COCO class index (0–79)
    std::vector<Detection> cam0_detections;      // all target detections in cam0
    std::vector<Detection> cam1_detections;      // all target detections in cam1
};

using FilteredPairCallback = std::function<void(FilteredInferencePair)>;


class DetectionFilter {
public:
    static constexpr std::size_t MAX_PRE_BUFFER    = 64;
    static constexpr uint64_t PAIR_TOLERANCE_NS = 200'000'000ULL;   // 200 ms

    explicit DetectionFilter(FilteredPairCallback on_pair);

    DetectionFilter(const DetectionFilter&)            = delete;
    DetectionFilter& operator=(const DetectionFilter&) = delete;

    // ── Upstream callbacks ─────────────────────────────────────────────────

    void on_intent(uint8_t object_id);
    void on_inference_packet(InferencePacket pkt);

    // ── Inspection ─────────────────────────────────────────────────────────

    bool        is_armed()         const noexcept;
    uint8_t     target_object_id() const noexcept;
    std::size_t buffer_depth()     const noexcept;

    // ── Lifecycle ──────────────────────────────────────────────────────────

    void reset();

private:
    FilteredPairCallback on_pair_;

    mutable std::mutex             mutex_;
    bool                           armed_            = false;
    uint8_t                        target_object_id_ = 0;
    std::deque<InferencePacket>    pre_buffer_;
    std::optional<InferencePacket> pending_cam0_;
    std::optional<InferencePacket> pending_cam1_;

    std::optional<FilteredInferencePair> try_pair(const InferencePacket& pkt);
    static bool timestamps_close(uint64_t a, uint64_t b) noexcept;
};
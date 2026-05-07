/*
    detection_filter.cpp  —  NorthStar stereo intent filter
*/

#include "detection_filter.hpp"

#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

DetectionFilter::DetectionFilter(FilteredPairCallback on_pair)
    : on_pair_(std::move(on_pair))
{
    if (!on_pair_)
        throw std::invalid_argument("DetectionFilter: on_pair callback must not be null");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public interface
// ─────────────────────────────────────────────────────────────────────────────

bool DetectionFilter::is_armed() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return armed_;
}

uint8_t DetectionFilter::target_object_id() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return target_object_id_;
}

std::size_t DetectionFilter::buffer_depth() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return pre_buffer_.size();
}

void DetectionFilter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    armed_            = false;
    target_object_id_ = 0;
    pre_buffer_.clear();
    pending_cam0_.reset();
    pending_cam1_.reset();
}

// ─────────────────────────────────────────────────────────────────────────────
//  on_intent
//  Sets the target object id, arms the filter, then replays any buffered
//  packets in arrival order.  The replay happens *outside* the lock so that
//  on_pair_ may safely call back into the filter.
// ─────────────────────────────────────────────────────────────────────────────

void DetectionFilter::on_intent(uint8_t object_id) {
    std::vector<InferencePacket> backlog;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        target_object_id_ = object_id;
        armed_            = true;

        // Drain the pre-buffer into a local vector for replay outside the lock.
        backlog.assign(
            std::make_move_iterator(pre_buffer_.begin()),
            std::make_move_iterator(pre_buffer_.end()));
        pre_buffer_.clear();
    }

    for (auto& pkt : backlog) {
        std::optional<FilteredInferencePair> result;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            result = try_pair(pkt);
        }
        if (result) on_pair_(*result);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  on_inference_packet
//  If unarmed: buffer (drop oldest on overflow).
//  If armed: filter and attempt to pair; fire callback outside the lock.
// ─────────────────────────────────────────────────────────────────────────────

void DetectionFilter::on_inference_packet(InferencePacket pkt) {
    std::optional<FilteredInferencePair> result;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!armed_) {
            if (pre_buffer_.size() >= MAX_PRE_BUFFER)
                pre_buffer_.pop_front();    // drop oldest
            pre_buffer_.push_back(std::move(pkt));
            return;
        }

        result = try_pair(pkt);
    }

    if (result) on_pair_(*result);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Private helpers
// ─────────────────────────────────────────────────────────────────────────────

bool DetectionFilter::timestamps_close(uint64_t a, uint64_t b) noexcept {
    return (a >= b ? a - b : b - a) < PAIR_TOLERANCE_NS;
}

std::optional<BoundingBox>
DetectionFilter::find_target_box(const InferencePacket& pkt) const {
    const Detection* best = nullptr;
    for (const auto& det : pkt.detections) {
        if (det.object_id == target_object_id_) {
            if (!best || det.confidence > best->confidence)
                best = &det;
        }
    }
    return best ? std::make_optional(best->box) : std::nullopt;
}

std::optional<FilteredInferencePair>
DetectionFilter::try_pair(const InferencePacket& pkt) {
    // Filter step: skip this packet entirely if the target isn't present.
    auto box = find_target_box(pkt);
    if (!box)
        return std::nullopt;

    if (pkt.camera_id == 0) {
        // Check if a pending cam1 packet is close enough in time.
        if (pending_cam1_ && timestamps_close(pkt.timestamp, pending_cam1_->timestamp)) {
            if (auto other_box = find_target_box(*pending_cam1_)) {
                FilteredInferencePair pair;
                pair.timestamp_avg = (pkt.timestamp + pending_cam1_->timestamp) / 2;
                pair.object_id     = target_object_id_;
                pair.bbox_cam0     = *box;
                pair.bbox_cam1     = *other_box;
                pending_cam1_.reset();
                return pair;
            }
            // pending_cam1_ doesn't contain the target — can't form a pair with it.
            // Overwrite it; the stale cam1 is no longer useful.
            pending_cam1_.reset();
        }
        // Store cam0 and wait for a matching cam1.
        pending_cam0_ = pkt;
        return std::nullopt;

    } else {   // camera_id == 1
        // Check if a pending cam0 packet is close enough in time.
        if (pending_cam0_ && timestamps_close(pkt.timestamp, pending_cam0_->timestamp)) {
            if (auto other_box = find_target_box(*pending_cam0_)) {
                FilteredInferencePair pair;
                pair.timestamp_avg = (pending_cam0_->timestamp + pkt.timestamp) / 2;
                pair.object_id     = target_object_id_;
                pair.bbox_cam0     = *other_box;
                pair.bbox_cam1     = *box;
                pending_cam0_.reset();
                return pair;
            }
            // pending_cam0_ lacks the target; discard it.
            pending_cam0_.reset();
        }
        // Store cam1 and wait for a matching cam0.
        pending_cam1_ = pkt;
        return std::nullopt;
    }
}

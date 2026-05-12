/*
    detection_filter.cpp  —  NorthStar stereo intent filter
*/

#include "detection_filter.hpp"

#include <stdexcept>

DetectionFilter::DetectionFilter(FilteredPairCallback on_pair)
    : on_pair_(std::move(on_pair))
{
    if (!on_pair_)
        throw std::invalid_argument("DetectionFilter: on_pair callback must not be null");
}

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

void DetectionFilter::on_intent(uint8_t object_id) {
    std::vector<InferencePacket> backlog;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        target_object_id_ = object_id;
        armed_            = true;
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

void DetectionFilter::on_inference_packet(InferencePacket pkt) {
    std::optional<FilteredInferencePair> result;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!armed_) {
            if (pre_buffer_.size() >= MAX_PRE_BUFFER)
                pre_buffer_.pop_front();
            pre_buffer_.push_back(std::move(pkt));
            return;
        }
        result = try_pair(pkt);
    }
    if (result) on_pair_(*result);
}

bool DetectionFilter::timestamps_close(uint64_t a, uint64_t b) noexcept {
    return (a >= b ? a - b : b - a) < PAIR_TOLERANCE_NS;
}

std::optional<FilteredInferencePair>
DetectionFilter::try_pair(const InferencePacket& pkt) {

    if (pkt.camera_id == 0) {
        if (pending_cam1_ && timestamps_close(pkt.timestamp, pending_cam1_->timestamp)) {
            FilteredInferencePair pair;
            pair.timestamp_avg = (pkt.timestamp + pending_cam1_->timestamp) / 2;
            pair.object_id     = target_object_id_;

            for (const auto& det : pkt.detections)
                if (det.object_id == target_object_id_)
                    pair.cam0_detections.push_back(det);

            for (const auto& det : pending_cam1_->detections)
                if (det.object_id == target_object_id_)
                    pair.cam1_detections.push_back(det);

            pending_cam1_.reset();

            if (!pair.cam0_detections.empty() && !pair.cam1_detections.empty())
                return pair;
            return std::nullopt;
        }
        pending_cam0_ = pkt;
        return std::nullopt;

    } else {
        if (pending_cam0_ && timestamps_close(pkt.timestamp, pending_cam0_->timestamp)) {
            FilteredInferencePair pair;
            pair.timestamp_avg = (pending_cam0_->timestamp + pkt.timestamp) / 2;
            pair.object_id     = target_object_id_;

            for (const auto& det : pending_cam0_->detections)
                if (det.object_id == target_object_id_)
                    pair.cam0_detections.push_back(det);

            for (const auto& det : pkt.detections)
                if (det.object_id == target_object_id_)
                    pair.cam1_detections.push_back(det);

            pending_cam0_.reset();

            if (!pair.cam0_detections.empty() && !pair.cam1_detections.empty())
                return pair;
            return std::nullopt;
        }
        pending_cam1_ = pkt;
        return std::nullopt;
    }
}
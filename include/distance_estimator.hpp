#pragma once
#include "sensor-fusion-structs.hpp"
#include "inference_packet.hpp"

class DistanceEstimator {
public: 
    DistanceEstimator(float focal_length_px, float baseline)
        :   focal_length_px_(focal_length_px), baseline_(baseline)
    {
    }

    std::optional<float> compute(const BoundingBox& cam0, const BoundingBox& cam1){
        //get the horizontal centres of each box
        float centre0 = (cam0.x_min + cam0.x_max) / 2;
        float centre1 = (cam1.x_min + cam1.x_max) / 2;

        //denormalise from 0-1 to pixels
        float centre0px = centre0 * 640;
        float centre1px = centre1 * 640;

        //get the disparity
        float disparity = centre0px - centre1px; //assuming cam0 is the left one

        if (disparity < 1.0f) { //avoid absurdly large distances
            return std::nullopt; 
        }

        //calculate the distance
        float distance = (focal_length_px_ * baseline_) / disparity;

        return distance;
    }
private:
    float focal_length_px_;
    float baseline_;
};
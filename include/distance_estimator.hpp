#include "sensor-fusion-structs.hpp"
#include "ThermalUtils.hpp"

class DistanceEstimator {
public: 
    DistanceEstimator(float focal_length_px, uint16_t disparity, float baseline)
        :   focal_length_px_(focal_length_px), baseline_(baseline)
    {
    }

    std::optional<float> compute(const BoundingBox& cam0, const BoundingBox& cam1){
        //get the horizontal centres of each box
        float centre0 = (cam0.x0 + cam0.x1) / 2;
        float centre1 = (cam1.x0 + cam1.x1) / 2;

        //denormalise from 0-1 to pixels
        float centre0px = centre0 * 640;
        float centre1px = centre1 * 640;

        //get the disparity
        float disparity = centre1px - centre0px;

        //calculate the distance
        float distance = (focal_length_px_ * baseline_) / disparity;

        return distance;
    }
private:
    float focal_length_px_;
    float baseline_;
};
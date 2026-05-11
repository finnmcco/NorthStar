#pragma once
#include "sensor-fusion-structs.hpp"
#include "inference_packet.hpp"

class DirectionEstimator {
public:
    GridCell compute(BoundingBox box){
        GridCell location;

        //get the centres of the box
        float xcentre = (box.x_min + box.x_max) / 2;
        float ycentre = (box.y_min + box.y_max) / 2;

        if (xcentre < 0.33f){
            location.col = 0;
        }
        else if (xcentre < 0.67f){
            location.col = 1;
        }
        else {
            location.col = 2;
        }

        if (ycentre < 0.33f){
            location.row = 0;
        }
        else if (ycentre < 0.67f){
            location.row = 1;
        }
        else {
            location.row = 2;
        }

        return location;
    }
};
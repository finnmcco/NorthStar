#include "sensor-fusion-structs.hpp"

class DirectionEstimator {
public:
    GridCell GetDirection(BoundingBox box){
        GridCell location;

        //get the centres of the box
        float xcentre = (box.x0 + box.x1) / 2;
        float ycentre = (box.y0 + box.y1) / 2;

        if (xcentre < 0.33){
            location.col = 0;
        }
        else if (xcentre < 0.67){
            location.col = 1;
        }
        else {
            location.col = 2;
        }

        if (ycentre < 0.33){
            location.row = 0;
        }
        else if (ycentre < 0.67){
            location.row = 1;
        }
        else {
            location.row = 2;
        }

        return location;
    }
};
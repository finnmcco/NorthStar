#pragma once

#include "inference_packet.hpp"

#include <algorithm>
#include <string>

struct Direction {
    int row = 1;  // 0 = top,    1 = middle, 2 = bottom
    int col = 1;  // 0 = left,   1 = middle, 2 = right
};

class DirectionEstimator {
public:
    Direction get_direction(const BoundingBox& box) const
    {
        Direction dir;

        const float centre_x = std::clamp(
            0.5f * (box.x_max + box.x_min),
            0.0f,
            1.0f
        );

        const float centre_y = std::clamp(
            0.5f * (box.y_max + box.y_min),
            0.0f,
            1.0f
        );

        if (centre_x < 1.0f / 3.0f) {
            dir.col = 0;
        } else if (centre_x < 2.0f / 3.0f) {
            dir.col = 1;
        } else {
            dir.col = 2;
        }

        if (centre_y < 1.0f / 3.0f) {
            dir.row = 0;
        } else if (centre_y < 2.0f / 3.0f) {
            dir.row = 1;
        } else {
            dir.row = 2;
        }

        return dir;
    }

    std::string cvt_to_string(const Direction& dir) const
    {
        std::string dir_string;

        switch (dir.row) {
            case 0:
                dir_string += "top ";
                break;
            case 1:
                dir_string += "middle ";
                break;
            case 2:
                dir_string += "bottom ";
                break;
            default:
                dir_string += "unknown ";
                break;
        }

        switch (dir.col) {
            case 0:
                dir_string += "left";
                break;
            case 1:
                dir_string += "middle";
                break;
            case 2:
                dir_string += "right";
                break;
            default:
                dir_string += "unknown";
                break;
        }

        return dir_string;
    }
};
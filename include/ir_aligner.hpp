#pragma once

#include <string> 
#include "inference_packet.hpp"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

class IRAligner {
public:
    explicit IRAligner(const std::string& yaml_path);

    struct PixelRect {
        int x0, y0, x1, y1;   // inclusive
        bool valid;            // false if the bbox projected entirely outside IR FoV
    };

    PixelRect project_bbox(const BoundingBox& bbox_cam0) const;

private:
    cv::Mat M_;       // 2x3 affine
    int ir_w_, ir_h_;
};
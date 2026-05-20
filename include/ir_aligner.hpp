#pragma once

/*
    ir_aligner.hpp  —  Runtime IR spatial alignment

    Loads a 3x3 homography H that maps normalised cam0 coordinates [0,1]
    to MLX90640 IR pixel coordinates [0..31] x [0..23].

    The homography is computed offline by ir_align_solver.py from a set of
    manually-annotated cam0/IR correspondences captured by ir_calib_capture.

    Usage:
        IRAligner aligner("ir_alignment.yaml");
        auto rect = aligner.project_bbox(bbox);
        if (rect.valid) {
            // sample ir_packet.temps[rect.y0..y1][rect.x0..x1]
        }

    Coordinate convention:
        - Camera bboxes are normalised floats in [0, 1], origin top-left.
        - IR pixel coordinates are integers, (0,0) = top-left of the 32x24 frame,
          x increases right, y increases down — matching the MLX90640 row-major
          layout used in IRPacket::temps[row * 32 + col].

    Depth dependence:
        A homography is exact only for a planar scene or a fixed object depth.
        H was fitted at a specific calibration distance (stored as a comment in
        the YAML). For objects at significantly different depths the projection
        will be slightly offset due to parallax between cam0 and the IR sensor.
        The error is typically < 1-2 IR pixels for depth variations of ±50 cm
        around the calibration distance at typical indoor ranges.
*/

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

#include "inference_packet.hpp"   // for BoundingBox

class IRAligner {
public:
    // Loads H, ir_width, ir_height from a YAML written by ir_align_solver.py.
    // Throws std::runtime_error if the file cannot be opened or is malformed.
    explicit IRAligner(const std::string& yaml_path);

    // Disable copy: cv::Mat members are ref-counted but semantically we want
    // a single calibration object per YAML.
    IRAligner(const IRAligner&)            = delete;
    IRAligner& operator=(const IRAligner&) = delete;

    // Axis-aligned bounding rectangle in IR pixel coordinates.
    // All bounds are inclusive integers in [0, ir_w-1] x [0, ir_h-1].
    // valid == false means the projected region fell entirely outside the
    // IR sensor's field of view.
    struct PixelRect {
        int  x0, y0;    // top-left  (inclusive)
        int  x1, y1;    // bottom-right (inclusive)
        bool valid;
    };

    // Project a normalised camera bounding box into IR pixel space.
    // All four corners of the bbox are transformed; the returned rect is the
    // axis-aligned bounding box of the resulting quadrilateral.
    PixelRect project_bbox(const BoundingBox& bbox_cam0) const;

    // Accessors for diagnostics.
    int ir_width()  const { return ir_w_; }
    int ir_height() const { return ir_h_; }
    const cv::Mat& H() const { return H_; }

private:
    cv::Mat H_;         // 3x3 homography, CV_64F
    int     ir_w_ = 0;
    int     ir_h_ = 0;
};
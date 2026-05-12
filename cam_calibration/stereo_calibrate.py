#!/usr/bin/env python3
"""
stereo_calibrate.py  -  NorthStar stereo calibration for the two Pi Cam 3 modules.

This script performs FULL stereo calibration (intrinsics + extrinsics) on the
post-downsample 640x640 image stream that the NorthStar inference pipeline
actually consumes. It is designed to be run ONCE per physical rig and the
resulting YAML can be loaded directly into the C++ runtime via cv::FileStorage.

Two subcommands:

    capture     Live-capture chessboard pairs from /base/cam0 and /base/cam1
                via picamera2. Frames are produced at sensor native 1920x1080,
                then put through the IDENTICAL crop+resize as cam_downsampler.cpp:
                    1920x1080 --(centered 1080x1080 crop)--> 1080x1080
                            --(cv2.resize INTER_AREA)--> 640x640
                A pair is only saved if the chessboard is detected in BOTH
                images and the new pose is sufficiently different from the
                last accepted pose (to enforce coverage diversity).

    calibrate   Reads the saved 640x640 pairs, runs per-camera intrinsic
                calibration, then stereo calibration with CALIB_FIX_INTRINSIC,
                then stereoRectify, and writes everything to a single YAML.

The YAML written by calibrate is consumable in C++ as:

    cv::FileStorage fs("stereo.yaml", cv::FileStorage::READ);
    cv::Mat K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q;
    fs["K1"] >> K1;  fs["D1"] >> D1;
    fs["K2"] >> K2;  fs["D2"] >> D2;
    fs["R"]  >> R;   fs["T"]  >> T;
    fs["R1"] >> R1;  fs["R2"] >> R2;
    fs["P1"] >> P1;  fs["P2"] >> P2;  fs["Q"] >> Q;

Calibration target:
    A printed (or rigid-board) chessboard. Default is 9x6 INNER corners with
    25 mm squares -- override with --cols / --rows / --square. Print on A3 or
    mount on a flat board; flatness matters more than absolute size accuracy
    because square size only scales the metric T/baseline. Wrong square size
    => correct intrinsics, wrong baseline.

Author: NorthStar
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants that mirror the runtime pipeline exactly. Do not change these
# unless cam_downsampler.cpp also changes.
# ---------------------------------------------------------------------------

SENSOR_W       = 1920          # libcamera VideoRecording native width
SENSOR_H       = 1080          # libcamera VideoRecording native height
OUT_SIZE       = 640           # post-downsample square edge consumed by inference
IMG_SIZE       = (OUT_SIZE, OUT_SIZE)   # (w, h) for OpenCV calibration calls

# Pose-diversity gate: a new pair is only accepted if at least one chessboard
# corner has moved more than this many pixels (in 640x640 coords) since the
# last accepted pair. Prevents 30 near-identical frames dominating the solver.
MIN_POSE_DELTA_PX = 40.0


# ---------------------------------------------------------------------------
# Geometry: replicate cam_downsampler.cpp exactly.
# ---------------------------------------------------------------------------

def downsample_like_runtime(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply the same center-crop + resize as cam_downsampler.cpp::process().

    Input:  HxWx3 BGR uint8 (typically 1080x1920)
    Output: 640x640x3 BGR uint8
    """
    h, w = frame_bgr.shape[:2]
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    cropped = frame_bgr[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Chessboard detection wrapper
# ---------------------------------------------------------------------------

@dataclass
class BoardSpec:
    cols: int           # inner corners across
    rows: int           # inner corners down
    square_m: float     # square edge length, METERS (calibration uses SI)

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """3D coordinates of corners on the board, Z=0, scaled by square size."""
        objp = np.zeros((self.cols * self.rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp *= self.square_m
        return objp


def detect_corners(gray: np.ndarray, board: BoardSpec) -> Optional[np.ndarray]:
    """
    Returns refined Nx1x2 float32 corners or None.

    Prefers findChessboardCornersSB (more robust to blur / shading) and falls
    back to the classic detector + cornerSubPix refinement.
    """
    # cv2.findChessboardCornersSB is in OpenCV >= 4.0 and produces
    # sub-pixel-accurate results without a separate refinement step.
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, board.pattern_size, flags_sb)
    if found:
        return corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board.pattern_size, flags)
    if not found:
        return None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return corners.astype(np.float32)


# ---------------------------------------------------------------------------
# CAPTURE MODE
# ---------------------------------------------------------------------------

def cmd_capture(args) -> int:
    """
    Live capture of stereo chessboard pairs. Pi-only (needs picamera2 + cameras).
    """
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("ERROR: picamera2 not installed. Capture mode must run on the Pi.")
        return 1

    board = BoardSpec(args.cols, args.rows, args.square / 1000.0)
    out_dir = Path(args.out_dir)
    (out_dir / "left").mkdir(parents=True, exist_ok=True)
    (out_dir / "right").mkdir(parents=True, exist_ok=True)

    # In picamera2 the format string is libcamera's, where "RGB888" produces
    # buffers that are BGR in memory -- i.e. directly compatible with OpenCV.
    main_cfg = {"size": (SENSOR_W, SENSOR_H), "format": "RGB888"}

    cams = []
    for cam_id in (0, 1):
        p = Picamera2(camera_num=cam_id)
        p.configure(p.create_video_configuration(main=main_cfg))
        p.start()
        cams.append(p)
    # Brief settle so AE/AWB converge before any pose is accepted.
    time.sleep(1.0)

    pair_idx = 0
    last_corners_left: Optional[np.ndarray] = None

    print("Capture controls:")
    print("  SPACE = save current pair (only enabled when both detections are valid)")
    print("  q / ESC = quit and finish")
    print(f"Target: at least {args.min_pairs} good pairs, ideally 25-40.")
    print("Move the board so it covers ALL regions of the frame, including corners,")
    print("and tilt it 20-30 degrees in multiple directions between captures.\n")

    try:
        while True:
            # Pull a frame from each camera. Capture is synchronous here --
            # for calibration we do NOT need hardware-synced shutters; the
            # board just has to be still while you trigger.
            raw_left  = cams[0].capture_array("main")
            raw_right = cams[1].capture_array("main")

            left  = downsample_like_runtime(raw_left)
            right = downsample_like_runtime(raw_right)

            gray_l = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            corners_l = detect_corners(gray_l, board)
            corners_r = detect_corners(gray_r, board)
            both_ok   = corners_l is not None and corners_r is not None

            # Build a side-by-side preview with detection overlays.
            preview_l = left.copy()
            preview_r = right.copy()
            if corners_l is not None:
                cv2.drawChessboardCorners(preview_l, board.pattern_size, corners_l, True)
            if corners_r is not None:
                cv2.drawChessboardCorners(preview_r, board.pattern_size, corners_r, True)

            status = f"pairs: {pair_idx}   L:{'OK' if corners_l is not None else '--'}  R:{'OK' if corners_r is not None else '--'}"
            preview = np.hstack([preview_l, preview_r])
            cv2.putText(preview, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if both_ok else (0, 0, 255), 2)

            cv2.imshow("NorthStar stereo calibration capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            if key == ord(' '):
                if not both_ok:
                    print("  rejected: board not seen in both cameras")
                    continue

                # Pose-diversity check: compare to previously accepted left
                # corner set. We use mean absolute corner displacement.
                if last_corners_left is not None:
                    delta = float(np.mean(np.linalg.norm(
                        corners_l.squeeze(1) - last_corners_left.squeeze(1), axis=1
                    )))
                    if delta < MIN_POSE_DELTA_PX:
                        print(f"  rejected: pose too similar to previous "
                              f"(mean corner delta {delta:.1f} px < {MIN_POSE_DELTA_PX})")
                        continue

                fname = f"pair_{pair_idx:03d}.png"
                cv2.imwrite(str(out_dir / "left"  / fname), left)
                cv2.imwrite(str(out_dir / "right" / fname), right)
                last_corners_left = corners_l
                pair_idx += 1
                print(f"  saved {fname}  (total: {pair_idx})")

    finally:
        cv2.destroyAllWindows()
        for c in cams:
            c.stop()
            c.close()

    if pair_idx < args.min_pairs:
        print(f"\nWARNING: only {pair_idx} pairs captured, recommend >= {args.min_pairs}.")
    else:
        print(f"\nDone. {pair_idx} pairs saved to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CALIBRATE MODE
# ---------------------------------------------------------------------------

def _load_pairs(pairs_dir: Path) -> List[Tuple[Path, Path]]:
    """Match left/right filenames; return sorted list of (left, right) paths."""
    left_dir  = pairs_dir / "left"
    right_dir = pairs_dir / "right"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"Expected {left_dir} and {right_dir} to exist")

    left_files = sorted(left_dir.glob("*.png"))
    out = []
    for lf in left_files:
        rf = right_dir / lf.name
        if rf.exists():
            out.append((lf, rf))
    return out


def cmd_calibrate(args) -> int:
    board = BoardSpec(args.cols, args.rows, args.square / 1000.0)
    pairs_dir = Path(args.pairs_dir)
    pairs = _load_pairs(pairs_dir)
    if len(pairs) < 6:
        print(f"ERROR: only {len(pairs)} pairs found, need at least ~12.")
        return 1
    print(f"Loaded {len(pairs)} stereo pairs from {pairs_dir}")

    objp = board.object_points()
    obj_points: List[np.ndarray] = []
    img_pts_l:  List[np.ndarray] = []
    img_pts_r:  List[np.ndarray] = []
    used_names: List[str]        = []

    for left_path, right_path in pairs:
        L = cv2.imread(str(left_path),  cv2.IMREAD_COLOR)
        R = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if L is None or R is None:
            print(f"  skip {left_path.name}: failed to read")
            continue
        if L.shape[1] != OUT_SIZE or L.shape[0] != OUT_SIZE:
            print(f"  skip {left_path.name}: not {OUT_SIZE}x{OUT_SIZE} "
                  f"(got {L.shape[1]}x{L.shape[0]})")
            continue

        cL = detect_corners(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), board)
        cR = detect_corners(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), board)
        if cL is None or cR is None:
            print(f"  skip {left_path.name}: chessboard not found in both")
            continue

        obj_points.append(objp)
        img_pts_l.append(cL)
        img_pts_r.append(cR)
        used_names.append(left_path.name)

    print(f"Using {len(obj_points)} pairs with valid detections.\n")
    if len(obj_points) < 6:
        print("ERROR: too few valid pairs after detection. Recapture.")
        return 1

    # -- Step 1: per-camera intrinsic calibration ---------------------------
    # Standard pinhole + radial-tangential (5-param) distortion. The PiCam 3
    # lens is rectilinear enough that the rational/thin-prism models tend to
    # overfit on this baseline; if you observe residual barrel distortion at
    # the edges, switch to CALIB_RATIONAL_MODEL.
    print("Calibrating LEFT camera...")
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(
        obj_points, img_pts_l, IMG_SIZE, None, None
    )
    print(f"  RMS reprojection error: {rms_l:.4f} px")

    print("Calibrating RIGHT camera...")
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(
        obj_points, img_pts_r, IMG_SIZE, None, None
    )
    print(f"  RMS reprojection error: {rms_r:.4f} px")

    # -- Step 2: stereo calibration -----------------------------------------
    # FIX_INTRINSIC keeps the K/D we just solved and only optimizes R, T, E, F.
    # This is the recommended two-stage workflow -- jointly refining K's with
    # extrinsics often makes the solver less stable when the baseline is small
    # relative to depth, which is typical for a Pi Cam 3 stereo head.
    print("\nStereo calibration (FIX_INTRINSIC)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    term  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_pts_l, img_pts_r,
        K1, D1, K2, D2, IMG_SIZE,
        criteria=term, flags=flags,
    )
    baseline_m = float(np.linalg.norm(T))
    print(f"  Stereo RMS reprojection error: {rms_stereo:.4f} px")
    print(f"  Baseline ||T||: {baseline_m * 1000.0:.2f} mm")
    print(f"  T (m): [{T[0,0]:+.4f}, {T[1,0]:+.4f}, {T[2,0]:+.4f}]")

    # -- Step 3: epipolar sanity check --------------------------------------
    # A correct F should put matched corners onto each other's epipolar lines.
    # Mean symmetric epipolar distance ~< 0.5 px is good, < 1.0 px is acceptable.
    print("\nEpipolar geometry sanity check...")
    sym_err = _symmetric_epipolar_error(img_pts_l, img_pts_r, F)
    print(f"  Mean symmetric epipolar distance: {sym_err:.4f} px")

    # -- Step 4: rectification ----------------------------------------------
    # alpha=0 crops to valid pixels only (no black borders); alpha=1 keeps all
    # source pixels with some black corners. 0 is generally what you want for
    # detection-driven distance estimation.
    print("\nStereo rectification...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, IMG_SIZE, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    fx_rect = float(P1[0, 0])
    cx_rect = float(P1[0, 2])
    # In the rectified system, depth Z = fx * baseline / disparity.
    print(f"  Rectified fx: {fx_rect:.2f} px")
    print(f"  Rectified principal point cx: {cx_rect:.2f} px")
    print(f"  Z per unit disparity at baseline {baseline_m*1000:.1f} mm:  "
          f"Z(d=1px) = {fx_rect * baseline_m * 1000:.0f} mm,  "
          f"Z(d=10px) = {fx_rect * baseline_m * 100:.0f} mm")

    # -- Step 5: write YAML --------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        out_path,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        image_size=IMG_SIZE,
        rms_left=rms_l, rms_right=rms_r, rms_stereo=rms_stereo,
        sym_epipolar_err_px=sym_err,
        baseline_m=baseline_m,
        num_pairs_used=len(obj_points),
        board=board,
    )
    print(f"\nWrote calibration to {out_path}")

    # Optional debug: save a rectified pair from the first input so the user
    # can visually confirm horizontal alignment of features.
    if args.dump_rectified and pairs:
        _dump_rectified_preview(
            pairs[0][0], pairs[0][1],
            K1, D1, K2, D2, R1, R2, P1, P2,
            out_path.parent / "rectified_preview.png",
        )

    return 0


# ---------------------------------------------------------------------------
# Helpers: epipolar error, YAML I/O, rectified preview
# ---------------------------------------------------------------------------

def _symmetric_epipolar_error(pts_l, pts_r, F) -> float:
    """Mean of d(x_r, F x_l) and d(x_l, F^T x_r) over all corner matches."""
    total = 0.0
    count = 0
    for cl, cr in zip(pts_l, pts_r):
        l_pts = cl.reshape(-1, 2)
        r_pts = cr.reshape(-1, 2)
        lines_r = cv2.computeCorrespondEpilines(l_pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        lines_l = cv2.computeCorrespondEpilines(r_pts.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        # d = |a*x + b*y + c| since (a,b) is normalized by computeCorrespondEpilines
        d_r = np.abs(lines_r[:, 0]*r_pts[:, 0] + lines_r[:, 1]*r_pts[:, 1] + lines_r[:, 2])
        d_l = np.abs(lines_l[:, 0]*l_pts[:, 0] + lines_l[:, 1]*l_pts[:, 1] + lines_l[:, 2])
        total += float(d_r.sum() + d_l.sum())
        count += len(l_pts) * 2
    return total / max(count, 1)


def _write_yaml(path: Path, **kw) -> None:
    """Write a YAML readable by both cv::FileStorage (C++) and a human."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)

    # Image geometry
    fs.write("image_width",  int(kw["image_size"][0]))
    fs.write("image_height", int(kw["image_size"][1]))

    # Intrinsics
    fs.write("K1", kw["K1"]); fs.write("D1", kw["D1"])
    fs.write("K2", kw["K2"]); fs.write("D2", kw["D2"])

    # Extrinsics
    fs.write("R", kw["R"]);   fs.write("T", kw["T"])
    fs.write("E", kw["E"]);   fs.write("F", kw["F"])

    # Rectification
    fs.write("R1", kw["R1"]); fs.write("R2", kw["R2"])
    fs.write("P1", kw["P1"]); fs.write("P2", kw["P2"])
    fs.write("Q",  kw["Q"])
    # Valid ROIs in the rectified images (x, y, w, h)
    fs.write("valid_roi_left",  np.array(kw["roi1"], dtype=np.int32))
    fs.write("valid_roi_right", np.array(kw["roi2"], dtype=np.int32))

    # Diagnostics + provenance
    fs.write("baseline_m",          float(kw["baseline_m"]))
    fs.write("rms_left_px",         float(kw["rms_left"]))
    fs.write("rms_right_px",        float(kw["rms_right"]))
    fs.write("rms_stereo_px",       float(kw["rms_stereo"]))
    fs.write("sym_epipolar_err_px", float(kw["sym_epipolar_err_px"]))
    fs.write("num_pairs_used",      int(kw["num_pairs_used"]))

    board: BoardSpec = kw["board"]
    fs.write("board_cols",       int(board.cols))
    fs.write("board_rows",       int(board.rows))
    fs.write("board_square_m",   float(board.square_m))
    fs.write("calibrated_unix",  int(time.time()))
    fs.release()


def _dump_rectified_preview(left_path: Path, right_path: Path,
                            K1, D1, K2, D2, R1, R2, P1, P2,
                            out_png: Path) -> None:
    L = cv2.imread(str(left_path));  R = cv2.imread(str(right_path))
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, IMG_SIZE, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, IMG_SIZE, cv2.CV_32FC1)
    Lr = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)
    side = np.hstack([Lr, Rr])
    # Horizontal lines: in a correctly rectified pair, matched features sit
    # on the same row across both halves.
    for y in range(0, side.shape[0], 32):
        cv2.line(side, (0, y), (side.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.imwrite(str(out_png), side)
    print(f"  rectified preview written to {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="NorthStar stereo calibration (640x640)")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_board = argparse.ArgumentParser(add_help=False)
    common_board.add_argument("--cols",   type=int, default=9,
                              help="inner corners across (default 9)")
    common_board.add_argument("--rows",   type=int, default=6,
                              help="inner corners down (default 6)")
    common_board.add_argument("--square", type=float, default=25.0,
                              help="square edge length in mm (default 25)")

    cap = sub.add_parser("capture", parents=[common_board],
                         help="capture stereo pairs live from both Pi Cam 3 modules")
    cap.add_argument("--out-dir",    default="calib_pairs",
                     help="directory to save left/ and right/ PNGs into")
    cap.add_argument("--min-pairs",  type=int, default=20,
                     help="minimum acceptable pair count (warning threshold)")
    cap.set_defaults(func=cmd_capture)

    cal = sub.add_parser("calibrate", parents=[common_board],
                         help="run full stereo calibration on saved pairs")
    cal.add_argument("--pairs-dir",        default="calib_pairs",
                     help="directory containing left/ and right/ subfolders")
    cal.add_argument("--out",              default="stereo_calib_640.yaml",
                     help="output YAML path")
    cal.add_argument("--dump-rectified",   action="store_true",
                     help="also write rectified_preview.png next to the YAML")
    cal.set_defaults(func=cmd_calibrate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""
disparity_rectified.py — Rectify a raw 640x640 stereo pair with an OpenCV
stereo calibration YAML, then generate a disparity map using OpenCV SGBM.

Usage:
    python disparity_rectified.py left.png right.png \
        --calib stereo_calib_640.yaml \
        --output disparity.png \
        --raw-output disparity.npy \
        --depth-vis-output depth.png \
        --rectified-pair-output rectified_pair.png

The depth PNG includes a labelled colour key in metres.

Important:
    - The provided calibration was generated for the post-downsample 640x640
      stream. Inputs to this script must therefore already be the same 640x640
      crop+resize frames used during calibration/runtime.
    - The script uses the YAML's stored R1/R2/P1/P2/Q matrices directly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    image_size: Tuple[int, int]  # (width, height)
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    baseline_m: Optional[float]
    rms_stereo_px: Optional[float]
    sym_epipolar_err_px: Optional[float]


def _read_required_mat(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise ValueError(f"Calibration file is missing required matrix: {key}")
    mat = node.mat()
    if mat is None:
        raise ValueError(f"Calibration key exists but is not a matrix: {key}")
    return mat


def _read_optional_real(fs: cv2.FileStorage, key: str) -> Optional[float]:
    node = fs.getNode(key)
    if node.empty():
        return None
    return float(node.real())


def load_calibration(path: str | Path) -> StereoCalibration:
    """Load the OpenCV FileStorage YAML written by stereo_calibrate.py."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration YAML: {path}")

    try:
        width_node = fs.getNode("image_width")
        height_node = fs.getNode("image_height")
        if width_node.empty() or height_node.empty():
            raise ValueError("Calibration YAML must contain image_width and image_height")

        image_size = (int(width_node.real()), int(height_node.real()))

        return StereoCalibration(
            image_size=image_size,
            K1=_read_required_mat(fs, "K1"),
            D1=_read_required_mat(fs, "D1"),
            K2=_read_required_mat(fs, "K2"),
            D2=_read_required_mat(fs, "D2"),
            R1=_read_required_mat(fs, "R1"),
            R2=_read_required_mat(fs, "R2"),
            P1=_read_required_mat(fs, "P1"),
            P2=_read_required_mat(fs, "P2"),
            Q=_read_required_mat(fs, "Q"),
            baseline_m=_read_optional_real(fs, "baseline_m"),
            rms_stereo_px=_read_optional_real(fs, "rms_stereo_px"),
            sym_epipolar_err_px=_read_optional_real(fs, "sym_epipolar_err_px"),
        )
    finally:
        fs.release()


def build_rectification_maps(calib: StereoCalibration):
    """Create remap tables for left and right images."""
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib.K1, calib.D1, calib.R1, calib.P1, calib.image_size, cv2.CV_16SC2
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib.K2, calib.D2, calib.R2, calib.P2, calib.image_size, cv2.CV_16SC2
    )
    return map1x, map1y, map2x, map2y


def rectify_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    maps,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply calibration rectification maps to a stereo pair."""
    map1x, map1y, map2x, map2y = maps
    left_rect = cv2.remap(left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect


def compute_disparity(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
    uniqueness_ratio: int = 15,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
) -> np.ndarray:
    """Compute a disparity map from an already-rectified BGR stereo pair."""
    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError("num_disparities must be positive and divisible by 16")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    channels = 1
    p1 = 8 * channels * block_size ** 2
    p2 = 32 * channels * block_size ** 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity_raw = stereo.compute(left_gray, right_gray)
    return disparity_raw.astype(np.float32) / 16.0


def visualise_disparity(disparity: np.ndarray) -> np.ndarray:
    """Normalise disparity to a viewable 8-bit colour map for display."""
    valid_mask = disparity > 0

    if not valid_mask.any():
        print("Warning: no valid disparity values found.", file=sys.stderr)
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    d_min = float(disparity[valid_mask].min())
    d_max = float(disparity[valid_mask].max())

    disparity_norm = np.zeros_like(disparity, dtype=np.uint8)
    if d_max > d_min:
        scaled = ((disparity - d_min) / (d_max - d_min) * 255).clip(0, 255)
        disparity_norm = scaled.astype(np.uint8)

    return cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)




def _depth_display_range(
    depth_m: np.ndarray,
    min_depth_m: Optional[float] = None,
    max_depth_m: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Return the depth range used for visualisation."""
    finite = np.isfinite(depth_m) & (depth_m > 0)
    if not finite.any():
        return None, None

    values = depth_m[finite]
    d_min = float(min_depth_m) if min_depth_m is not None else float(np.percentile(values, 2))
    d_max = float(max_depth_m) if max_depth_m is not None else float(np.percentile(values, 98))

    if d_max <= d_min:
        d_min = float(values.min())
        d_max = float(values.max())

    return d_min, d_max


def visualise_depth(
    depth_m: np.ndarray,
    min_depth_m: Optional[float] = None,
    max_depth_m: Optional[float] = None,
) -> np.ndarray:
    """Normalise metric depth to a viewable 8-bit colour map for display.

    Nearer valid pixels are brighter/warmer after inversion. NaN/Inf/invalid
    pixels are rendered black. If min/max are omitted, robust percentiles are
    used so a few outliers do not dominate the visualisation.
    """
    finite = np.isfinite(depth_m) & (depth_m > 0)
    if not finite.any():
        print("Warning: no valid depth values found.", file=sys.stderr)
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    d_min, d_max = _depth_display_range(depth_m, min_depth_m, max_depth_m)
    if d_min is None or d_max is None or d_max <= d_min:
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    depth_norm = np.zeros(depth_m.shape, dtype=np.uint8)
    clipped = np.clip(depth_m, d_min, d_max)
    # Invert so near objects are high intensity, matching the disparity visualisation.
    scaled = (1.0 - (clipped - d_min) / (d_max - d_min)) * 255.0
    depth_norm[finite] = scaled[finite].clip(0, 255).astype(np.uint8)

    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def add_depth_key(
    depth_vis: np.ndarray,
    depth_m: np.ndarray,
    min_depth_m: Optional[float] = None,
    max_depth_m: Optional[float] = None,
    key_width: int = 110,
) -> np.ndarray:
    """Append a labelled colour key showing the depth scale in metres."""
    d_min, d_max = _depth_display_range(depth_m, min_depth_m, max_depth_m)
    if d_min is None or d_max is None or d_max <= d_min:
        return depth_vis

    h = depth_vis.shape[0]
    bar_w = 24
    margin = 12
    key = np.full((h, key_width, 3), 255, dtype=np.uint8)

    # Build a vertical bar using the same inverted depth-to-colour mapping as the image.
    # Top = near/min depth, bottom = far/max depth.
    gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    colour_bar = cv2.applyColorMap(np.repeat(gradient, bar_w, axis=1), cv2.COLORMAP_JET)
    x0 = margin
    y0 = 0
    key[y0:y0 + h, x0:x0 + bar_w] = colour_bar

    cv2.rectangle(key, (x0, 0), (x0 + bar_w - 1, h - 1), (0, 0, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_x = x0 + bar_w + 8
    tick_len = 6
    for frac in np.linspace(0.0, 1.0, 6):
        y = int(round(frac * (h - 1)))
        depth = d_min + frac * (d_max - d_min)
        cv2.line(key, (x0 + bar_w, y), (x0 + bar_w + tick_len, y), (0, 0, 0), 1)
        text = f"{depth:.2f} m" if depth < 10 else f"{depth:.1f} m"
        text_y = min(max(y + 5, 12), h - 5)
        cv2.putText(key, text, (label_x, text_y), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(key, "near", (label_x, 24), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(key, "far", (label_x, h - 12), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return np.hstack([depth_vis, key])


def make_rectified_pair_preview(left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
    """Return side-by-side rectified frames with horizontal epipolar guide lines."""
    preview = np.hstack([left_rect, right_rect])
    for y in range(0, preview.shape[0], 32):
        cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 255, 0), 1)
    return preview


def disparity_to_depth_m(disparity: np.ndarray, calib: StereoCalibration) -> np.ndarray:
    """
    Convert disparity to depth in metres using the rectified projection matrices.

    For this calibration layout: P2[0,3] = -fx * baseline, so
        Z = -P2[0,3] / disparity.
    """
    fx_baseline = -float(calib.P2[0, 3])
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = fx_baseline / disparity[valid]
    return depth


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rectify a calibrated stereo pair, then compute disparity."
    )
    parser.add_argument("left", help="Left/cam0 image path; must match calibration resolution")
    parser.add_argument("right", help="Right/cam1 image path; must match calibration resolution")
    parser.add_argument(
        "--calib",
        default="stereo_calib_640.yaml",
        help="OpenCV stereo calibration YAML containing K/D/R1/R2/P1/P2/Q",
    )
    parser.add_argument("--output", default="disparity.png", help="Output disparity visualisation path")
    parser.add_argument("--raw-output", default=None, help="Optional path to save raw float disparity as .npy")
    parser.add_argument("--depth-output", default=None, help="Optional path to save float depth-in-metres as .npy")
    parser.add_argument("--depth-vis-output", default=None, help="Optional path to save a visual metric depth map PNG")
    parser.add_argument("--min-depth", type=float, default=None, help="Optional nearest depth in metres for depth PNG scaling")
    parser.add_argument("--max-depth", type=float, default=None, help="Optional farthest depth in metres for depth PNG scaling")
    parser.add_argument("--points-output", default=None, help="Optional path to save cv2.reprojectImageTo3D output as .npy")
    parser.add_argument("--left-rect-output", default=None, help="Optional path to save the rectified left image")
    parser.add_argument("--right-rect-output", default=None, help="Optional path to save the rectified right image")
    parser.add_argument(
        "--rectified-pair-output",
        default=None,
        help="Optional path to save side-by-side rectified images with epipolar lines",
    )
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128, help="Must be divisible by 16")
    parser.add_argument("--block-size", type=int, default=5, help="Odd integer >= 3")
    parser.add_argument("--uniqueness-ratio", type=int, default=15)
    parser.add_argument("--speckle-window-size", type=int, default=100)
    parser.add_argument("--speckle-range", type=int, default=2)
    args = parser.parse_args()

    try:
        calib = load_calibration(args.calib)
    except Exception as exc:
        sys.exit(str(exc))

    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    if left is None:
        sys.exit(f"Could not read left image: {args.left}")
    if right is None:
        sys.exit(f"Could not read right image: {args.right}")
    if left.shape != right.shape:
        sys.exit(f"Image sizes don't match: {left.shape} vs {right.shape}")

    expected_w, expected_h = calib.image_size
    if (left.shape[1], left.shape[0]) != calib.image_size:
        sys.exit(
            "Input images must match the calibration resolution: "
            f"got {left.shape[1]}x{left.shape[0]}, expected {expected_w}x{expected_h}. "
            "Use the same crop+resize pipeline that was used during calibration."
        )

    print(f"Loaded calibration for {expected_w}x{expected_h} frames.")
    if calib.baseline_m is not None:
        print(f"Baseline: {calib.baseline_m * 1000.0:.2f} mm")
    if calib.rms_stereo_px is not None:
        print(f"Stereo RMS: {calib.rms_stereo_px:.3f} px")
    if calib.sym_epipolar_err_px is not None:
        print(f"Symmetric epipolar error: {calib.sym_epipolar_err_px:.3f} px")

    maps = build_rectification_maps(calib)
    left_rect, right_rect = rectify_pair(left, right, maps)

    if args.left_rect_output:
        cv2.imwrite(args.left_rect_output, left_rect)
        print(f"Saved rectified left image to {args.left_rect_output}")
    if args.right_rect_output:
        cv2.imwrite(args.right_rect_output, right_rect)
        print(f"Saved rectified right image to {args.right_rect_output}")
    if args.rectified_pair_output:
        preview = make_rectified_pair_preview(left_rect, right_rect)
        cv2.imwrite(args.rectified_pair_output, preview)
        print(f"Saved rectified pair preview to {args.rectified_pair_output}")

    print(f"Computing disparity for rectified {expected_w}x{expected_h} pair...")
    try:
        disparity = compute_disparity(
            left_rect,
            right_rect,
            min_disparity=args.min_disparity,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            uniqueness_ratio=args.uniqueness_ratio,
            speckle_window_size=args.speckle_window_size,
            speckle_range=args.speckle_range,
        )
    except Exception as exc:
        sys.exit(str(exc))

    valid = disparity > args.min_disparity
    valid_values = disparity[valid]
    if valid_values.size:
        print(f"Valid pixels: {valid_values.size} / {disparity.size} "
              f"({100 * valid_values.size / disparity.size:.1f}%)")
        print(f"Disparity range: {valid_values.min():.2f} to {valid_values.max():.2f} px")
        print(f"Median disparity: {np.median(valid_values):.2f} px")
        depth = disparity_to_depth_m(disparity, calib)
        depth_valid = depth[np.isfinite(depth)]
        if depth_valid.size:
            print(f"Median depth: {np.median(depth_valid):.2f} m")
    else:
        print("No valid disparities computed — check rectification, exposure, texture, and SGBM parameters.")

    vis = visualise_disparity(disparity)
    cv2.imwrite(args.output, vis)
    print(f"Saved disparity visualisation to {args.output}")

    if args.raw_output:
        np.save(args.raw_output, disparity)
        print(f"Saved raw disparity to {args.raw_output}")

    needs_depth = args.depth_output or args.depth_vis_output
    depth = disparity_to_depth_m(disparity, calib) if needs_depth else None

    if args.depth_output and depth is not None:
        np.save(args.depth_output, depth)
        print(f"Saved depth map to {args.depth_output}")

    if args.depth_vis_output and depth is not None:
        depth_vis = visualise_depth(depth, min_depth_m=args.min_depth, max_depth_m=args.max_depth)
        depth_vis = add_depth_key(depth_vis, depth, min_depth_m=args.min_depth, max_depth_m=args.max_depth)
        cv2.imwrite(args.depth_vis_output, depth_vis)
        print(f"Saved labelled depth visualisation to {args.depth_vis_output}")

    if args.points_output:
        points_3d = cv2.reprojectImageTo3D(disparity, calib.Q)
        points_3d[~valid] = np.nan
        np.save(args.points_output, points_3d)
        print(f"Saved 3D points to {args.points_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

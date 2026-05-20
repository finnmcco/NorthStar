
#!/usr/bin/env python3
"""
ir_align_solver.py  —  Compute and save the cam0→IR homography H.

Given a set of annotated (cam0_point, ir_point) correspondences, this script
fits a 3x3 homography H that maps normalised cam0 coordinates [0,1] to
MLX90640 IR pixel coordinates, and writes it to a YAML consumable by IRAligner
in the C++ runtime.

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

Workflow:
    1.  Capture synchronised pairs with ir_calib_capture (the C++ tool).
        Each pair produces:
            ir_calib/pair_NN_cam0.png   — 640x640 BGR
            ir_calib/pair_NN_ir.png     — false-colour heatmap (for viewing)
            ir_calib/pair_NN_ir.csv     — raw temperatures 24x32, row-major

    2.  Open each pair_NN_cam0.png and pair_NN_ir.png side by side.
        Place a thermally distinct point source (hot object, soldering iron
        tip, cup of tea) at a known position and identify:
            - The pixel (u, v) in the 640x640 cam0 image
            - The pixel (col, row) in the 32x24 IR image
        IR pixel (0,0) is top-left; x increases right, y increases down.
        
        Add entries to CORRESPONDENCES below, one per capture.

    3.  Run:
            python3 ir_align_solver.py [--calib-dir ir_calib] [--out ir_alignment.yaml]
            python3 ir_align_solver.py --interactive   # click-to-annotate mode

    4.  Inspect the printed residuals and the verification overlay images.
        A good fit has mean reprojection error < 1.5 IR pixels.

─────────────────────────────────────────────────────────────────────────────
Notes on the homography
─────────────────────────────────────────────────────────────────────────────

H maps:
    [x_ir; y_ir; 1] ~ H * [x_cam_norm; y_cam_norm; 1]

where:
    x_cam_norm = u_cam / 640   (normalised camera x, [0, 1])
    y_cam_norm = v_cam / 640   (normalised camera y, [0, 1])
    x_ir, y_ir                 (IR pixel, [0..31] x [0..23])

The homography has 8 degrees of freedom. You need at least 4 non-collinear
correspondences to solve it; >= 8-10 spread across the frame is recommended
for RANSAC robustness.

Depth dependence: H is exact only at the calibration depth (TARGET_DEPTH_M).
At other depths the projection has a parallax error proportional to the
sensor offset and the depth difference. Document the calibration depth in
the output YAML and capture at your most common operating range.

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

IR_WIDTH  = 32
IR_HEIGHT = 24
CAM_SIZE  = 640          # post-downsample square edge (pixels)

# Depth at which calibration was performed.
# Document this so the C++ runtime can warn if it's being used far from this.
TARGET_DEPTH_M = 0.5

# RANSAC reprojection threshold in IR pixels.
# A correspondence is an inlier if its projection error is below this.
# 1.0 is tight but appropriate for a 32x24 sensor; relax to 1.5 if RANSAC
# rejects too many of your carefully annotated points.
RANSAC_THRESH = 1.0

# ─── Correspondences ──────────────────────────────────────────────────────────
# Edit this list directly, or use --interactive mode.
#
# Format: (cam0_pixel_u, cam0_pixel_v, ir_col, ir_row)
#   cam0_pixel_u, cam0_pixel_v : integer pixel in 640x640 cam0 image
#   ir_col, ir_row             : integer pixel in 32x24 IR image (0-indexed)
#
# Aim for >= 8 points spread across all four quadrants of the frame.
# Also vary depth slightly if you can — RANSAC will reject outliers from
# the depth-dependent parallax shift.
#
# Example entries (replace with your measured values):
CORRESPONDENCES: List[Tuple[int, int, int, int]] = [
    # (cam_u, cam_v, ir_col, ir_row)
    # --- top row ---
    # (130, 110,  4,  3),
    # (320, 105, 15,  3),
    # (510,  98, 26,  3),
    # --- middle row ---
    # (128, 320,  4, 12),
    # (320, 318, 15, 12),
    # (511, 322, 26, 12),
    # --- bottom row ---
    # (130, 520,  4, 21),
    # (320, 525, 15, 21),
    # (505, 518, 26, 21),
    (454, 189, 13, 17),
    (337, 216, 11, 13),
    (337, 247, 13, 12),
    (452, 307, 17, 15),
    (335, 300, 15, 11),
    (305, 328, 15, 10),
]


# ─── Core solver ──────────────────────────────────────────────────────────────

def build_point_arrays(
    correspondences: List[Tuple[int, int, int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the raw correspondence list to:
        src_pts : Nx1x2 float32  (normalised cam0 coords)
        dst_pts : Nx1x2 float32  (IR pixel coords)
    """
    src = []
    dst = []
    for cam_u, cam_v, ir_col, ir_row in correspondences:
        src.append([cam_u / CAM_SIZE, cam_v / CAM_SIZE])
        dst.append([float(ir_col), float(ir_row)])
    return (np.array(src, dtype=np.float32).reshape(-1, 1, 2),
            np.array(dst, dtype=np.float32).reshape(-1, 1, 2))


def solve_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_thresh: float = RANSAC_THRESH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit H using RANSAC.  Returns (H, inlier_mask).
    Raises RuntimeError if fewer than 4 inliers remain.
    """
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError(
            "findHomography returned None — need >= 4 non-collinear correspondences."
        )
    n_inliers = int(mask.sum())
    print(f"RANSAC: {n_inliers}/{len(src_pts)} correspondences accepted as inliers.")
    if n_inliers < 4:
        raise RuntimeError(
            f"Only {n_inliers} inliers after RANSAC — too few to trust. "
            "Add more correspondences and/or relax RANSAC_THRESH."
        )
    return H, mask


def reprojection_errors(
    H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float, List[float]]:
    """
    Compute per-point reprojection error in IR pixel units.
    Returns (mean_all, mean_inliers, per_point_errors).
    """
    projected = cv2.perspectiveTransform(src_pts, H).reshape(-1, 2)
    actual    = dst_pts.reshape(-1, 2)
    errors    = np.linalg.norm(projected - actual, axis=1).tolist()

    inlier_flat = mask.flatten().astype(bool)
    mean_all     = float(np.mean(errors))
    mean_inliers = float(np.mean(np.array(errors)[inlier_flat]))
    return mean_all, mean_inliers, errors


def write_yaml(
    out_path: Path,
    H: np.ndarray,
    ir_w: int,
    ir_h: int,
    target_depth_m: float,
    mean_err_px: float,
    n_correspondences: int,
    n_inliers: int,
) -> None:
    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    fs.write("H",         H.astype(np.float64))
    fs.write("ir_width",  ir_w)
    fs.write("ir_height", ir_h)
    fs.write("cam_size",  CAM_SIZE)
    fs.write("calibration_depth_m",   float(target_depth_m))
    fs.write("mean_reprojection_err_px", float(mean_err_px))
    fs.write("n_correspondences",     n_correspondences)
    fs.write("n_inliers",             n_inliers)
    import time
    fs.write("calibrated_unix", int(time.time()))
    fs.release()
    print(f"Wrote {out_path}")


# ─── Verification visualisation ───────────────────────────────────────────────

def discover_pair_indices(calib_dir):
    """Scan calib_dir for pair_NN_ir.png and return sorted list of NN indices."""
    import re as _re
    indices = []
    for p in calib_dir.glob("pair_*_ir.png"):
        m = _re.match(r"pair_(\d+)_ir\.png", p.name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def make_verification_overlay(H, src_pts, dst_pts, mask, calib_dir, correspondences):
    """
    Produce a side-by-side cam0+IR verification PNG per correspondence.
    Pair numbers are discovered by scanning calib_dir for pair_NN_ir.png.
    The i-th correspondence maps to the i-th discovered pair index.
    """
    scale = 16
    projected = cv2.perspectiveTransform(src_pts, H).reshape(-1, 2)
    actual    = dst_pts.reshape(-1, 2)
    inliers   = mask.flatten().astype(bool)

    pair_indices = discover_pair_indices(calib_dir)

    if not pair_indices:
        print(f"[verify] no pair_NN_ir.png found in {calib_dir} — skipping overlays")
        return

    if len(pair_indices) != len(correspondences):
        print(f"[verify] {len(correspondences)} correspondences vs "
              f"{len(pair_indices)} pair files — matching first "
              f"{min(len(pair_indices), len(correspondences))}.")

    n = min(len(pair_indices), len(correspondences))
    saved = 0

    for i in range(n):
        pair_num = pair_indices[i]
        ir_path  = calib_dir / f"pair_{pair_num:02d}_ir.png"
        cam_path = calib_dir / f"pair_{pair_num:02d}_cam0.png"

        ir_img = cv2.imread(str(ir_path))
        if ir_img is None:
            print(f"  [verify] skip pair {pair_num:02d}: cannot read IR image")
            continue

        # Annotate IR panel
        ax = int(round(actual[i, 0])) * scale + scale // 2
        ay = int(round(actual[i, 1])) * scale + scale // 2
        colour = (0, 220, 0) if inliers[i] else (0, 0, 220)
        cv2.circle(ir_img, (ax, ay), 8, colour, 2)

        px_d = int(round(projected[i, 0] * scale + scale / 2))
        py_d = int(round(projected[i, 1] * scale + scale / 2))
        cv2.drawMarker(ir_img, (px_d, py_d), (255, 220, 0), cv2.MARKER_CROSS, 14, 2)

        err = float(np.linalg.norm(projected[i] - actual[i]))
        status = "OK" if inliers[i] else "OUTLIER"
        cv2.putText(ir_img, f"err={err:.2f}px [{status}]",
                    (6, ir_img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(ir_img, f"pair {pair_num:02d}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Annotate cam panel
        cam_u, cam_v   = correspondences[i][0], correspondences[i][1]
        ir_col, ir_row = correspondences[i][2], correspondences[i][3]

        if cam_path.exists():
            cam_img = cv2.imread(str(cam_path)).copy()
        else:
            cam_img = np.full((640, 640, 3), 60, dtype=np.uint8)
            cv2.putText(cam_img, "cam0 image not found",
                        (120, 320), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (180, 180, 180), 2)

        cv2.circle(cam_img, (cam_u, cam_v), 10, (0, 220, 0), 2)
        cv2.drawMarker(cam_img, (cam_u, cam_v),
                       (0, 220, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(cam_img,
                    f"cam({cam_u},{cam_v}) -> IR({ir_col},{ir_row})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 0), 2, cv2.LINE_AA)

        # Combine side-by-side
        target_h  = cam_img.shape[0]
        ir_scale  = target_h / ir_img.shape[0]
        ir_rsz    = cv2.resize(ir_img,
                               (int(ir_img.shape[1] * ir_scale), target_h),
                               interpolation=cv2.INTER_NEAREST)
        divider   = np.full((target_h, 4, 3), 200, dtype=np.uint8)
        combined  = np.hstack([cam_img, divider, ir_rsz])

        out = calib_dir / f"pair_{pair_num:02d}_verify.png"
        cv2.imwrite(str(out), combined)
        saved += 1

    print(f"Verification overlays: {saved} written to {calib_dir}/pair_NN_verify.png")
    print("  Green = inlier (cam point / IR actual)")
    print("  Red   = outlier IR point")
    print("  Cyan  = projected point via H")

# ─── Interactive annotation mode ─────────────────────────────────────────────

def interactive_annotate(calib_dir: Path) -> List[Tuple[int, int, int, int]]:
    """
    Semi-interactive annotation: for each pair found in calib_dir, show
    the cam0 image and the IR heatmap, and ask the user to click the
    corresponding hot-spot in each.

    OpenCV click callbacks are used for the camera image; for the IR image
    the user types the col/row because at 32x24 resolution clicking the
    upscaled image and then dividing by scale is error-prone.
    """
    ir_pngs  = sorted(calib_dir.glob("pair_*_ir.png"))
    cam_pngs = sorted(calib_dir.glob("pair_*_cam0.png"))
    pairs    = list(zip(cam_pngs, ir_pngs))

    if not pairs:
        print(f"No pairs found in {calib_dir}. Run ir_calib_capture first.")
        return []

    correspondences: List[Tuple[int, int, int, int]] = []
    click_pt: List[Tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt.clear()
            click_pt.append((x, y))

    print("\n─── Interactive annotation ───────────────────────────────────────")
    print("For each pair:")
    print("  1. Click the hot-spot in the camera window.")
    print("  2. Type the IR col and row when prompted.")
    print("  3. Press ENTER to accept, 's' to skip, 'q' to finish early.\n")

    cv2.namedWindow("cam0", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("cam0", on_click)

    for cam_path, ir_path in pairs:
        pair_name = cam_path.stem.replace("_cam0", "")
        cam_img = cv2.imread(str(cam_path))
        ir_img  = cv2.imread(str(ir_path))
        if cam_img is None or ir_img is None:
            print(f"  skip {pair_name}: failed to read images")
            continue

        click_pt.clear()
        print(f"\nPair: {pair_name}")
        cv2.imshow("cam0", cam_img)
        cv2.imshow("IR heatmap (read-only)", ir_img)

        print("  Click the hot-spot in the cam0 window, then press ENTER.")
        print("  Press 's' to skip this pair, 'q' to quit annotation.")

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return correspondences
            if key == ord('s'):
                print(f"  skipped {pair_name}")
                break
            if key == 13 or key == 10:  # ENTER
                if not click_pt:
                    print("  No click yet — click the cam0 window first.")
                    continue
                cam_u, cam_v = click_pt[0]
                print(f"  cam0 click: ({cam_u}, {cam_v})")

                # Draw the click on a copy of the cam image for feedback
                annotated = cam_img.copy()
                cv2.circle(annotated, (cam_u, cam_v), 6, (0, 255, 0), 2)
                cv2.imshow("cam0", annotated)
                cv2.waitKey(1)

                try:
                    ir_col_s = input(f"  IR column (0–{IR_WIDTH-1}): ").strip()
                    ir_row_s = input(f"  IR row    (0–{IR_HEIGHT-1}): ").strip()
                    ir_col = int(ir_col_s)
                    ir_row = int(ir_row_s)
                except (ValueError, EOFError):
                    print("  Invalid input — skipping pair.")
                    break

                if not (0 <= ir_col < IR_WIDTH and 0 <= ir_row < IR_HEIGHT):
                    print(f"  IR coords out of range — skipping pair.")
                    break

                correspondences.append((cam_u, cam_v, ir_col, ir_row))
                print(f"  Added: cam({cam_u},{cam_v}) → IR({ir_col},{ir_row})")
                break

    cv2.destroyAllWindows()
    return correspondences


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit cam0→IR homography from annotated correspondences."
    )
    parser.add_argument("--calib-dir", default="ir_calib_50cm",
                        help="Directory with pair_NN_cam0.png / pair_NN_ir.png")
    parser.add_argument("--out", default="ir_alignment.yaml",
                        help="Output YAML path (default: ir_alignment.yaml)")
    parser.add_argument("--interactive", action="store_true",
                        help="Click-to-annotate mode: prompts for each pair")
    parser.add_argument("--ransac-thresh", type=float, default=RANSAC_THRESH,
                        help=f"RANSAC reprojection threshold in IR px "
                             f"(default {RANSAC_THRESH})")
    parser.add_argument("--depth-m", type=float, default=TARGET_DEPTH_M,
                        help=f"Calibration depth in metres (default {TARGET_DEPTH_M})")
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)

    # ── Gather correspondences ────────────────────────────────────────────────
    if args.interactive:
        correspondences = interactive_annotate(calib_dir)
    else:
        correspondences = CORRESPONDENCES

    if len(correspondences) < 4:
        print(f"ERROR: need >= 4 correspondences, have {len(correspondences)}.")
        print("Either add entries to CORRESPONDENCES or use --interactive.")
        return 1

    print(f"\nSolving homography from {len(correspondences)} correspondences...")
    src_pts, dst_pts = build_point_arrays(correspondences)

    # ── Solve ─────────────────────────────────────────────────────────────────
    try:
        H, mask = solve_homography(src_pts, dst_pts,
                                   ransac_thresh=args.ransac_thresh)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # ── Residuals ─────────────────────────────────────────────────────────────
    mean_all, mean_inliers, per_pt_errs = reprojection_errors(H, src_pts, dst_pts, mask)
    inlier_flat = mask.flatten().astype(bool)

    print("\nPer-point reprojection errors (IR pixels):")
    for i, ((cam_u, cam_v, ir_col, ir_row), err) in enumerate(
        zip(correspondences, per_pt_errs)
    ):
        status = "inlier" if inlier_flat[i] else "OUTLIER"
        print(f"  [{i:2d}] cam({cam_u:3d},{cam_v:3d}) → IR({ir_col:2d},{ir_row:2d})"
              f"  err={err:.3f} px  [{status}]")

    print(f"\nMean reprojection error (all):     {mean_all:.4f} px")
    print(f"Mean reprojection error (inliers): {mean_inliers:.4f} px")

    if mean_inliers > 2.0:
        print("\nWARNING: mean inlier error > 2.0 px. Consider:")
        print("  - Re-annotating any obviously wrong correspondences")
        print("  - Adding more spread-out correspondences")
        print("  - Checking the IR csv to confirm hotspot pixel positions")
    elif mean_inliers > 1.0:
        print("\nAcceptable but not ideal — try adding more corner correspondences.")
    else:
        print("\nGood fit. Mean inlier error < 1.0 IR pixel.")

    # ── Save YAML ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    write_yaml(
        out_path,
        H,
        ir_w=IR_WIDTH,
        ir_h=IR_HEIGHT,
        target_depth_m=args.depth_m,
        mean_err_px=mean_inliers,
        n_correspondences=len(correspondences),
        n_inliers=int(mask.sum()),
    )

    # ── Verification overlays ─────────────────────────────────────────────────
    # Extract pair numbers from the correspondence-pair filenames if interactive,
    # or just use 1..N if correspondences were typed manually.
    make_verification_overlay(H, src_pts, dst_pts, mask, calib_dir, correspondences)

    return 0


if __name__ == "__main__":
    sys.exit(main())
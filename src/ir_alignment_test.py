#!/usr/bin/env python3
"""
ir_alignment_test.py  —  Evaluate the cam0→IR homography across multiple depths.

Tests the accuracy of the IR spatial alignment by:
  1. Discovering test captures (cam0 PNG + IR CSV pairs)
  2. Interactively annotating two bboxes per capture:
       - Camera bbox: the bounding box of the hot object in the cam0 image
       - IR ground-truth bbox: where the hot object actually appears in the
         IR heatmap, drawn manually by the user
  3. Projecting the camera bbox through H into IR pixel space
  4. Measuring the centre error between the projected bbox centroid and the
     manually-annotated IR ground-truth centroid
  5. Generating side-by-side comparison PNGs for report figures
  6. Printing a summary table grouped by distance
  7. Saving an error-vs-distance plot

─────────────────────────────────────────────────────────────────────────────
Why manual IR annotation rather than thresholding
─────────────────────────────────────────────────────────────────────────────

Thresholding is sensitive to ambient temperature, object temperature, and the
chosen threshold value. It also conflates projection error with whether the IR
sensor can see the object clearly at a given distance. Manual annotation of
"where the object appears in the IR frame" is the direct ground truth — it
measures exactly what the homography is trying to achieve.

─────────────────────────────────────────────────────────────────────────────
Error metric
─────────────────────────────────────────────────────────────────────────────

Centre error (IR pixels):
    Euclidean distance between:
      - the centroid of the projected bbox (output of H applied to cam bbox)
      - the centroid of the manually-drawn IR ground-truth bbox
    1 IR pixel ≈ 1.7° of FOV for the MLX90640 at 55° horizontal FOV.

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

Single calibration YAML:
    python3 ir_alignment_test.py \\
        --yaml ir_align_100.yaml \\
        --discover-dir ir_test_data \\
        --interactive

Depth-stratified (100cm H for <=1.25m, 150cm H beyond):
    python3 ir_alignment_test.py \\
        --yaml ir_align_100.yaml \\
        --yaml2 ir_align_150.yaml \\
        --switch-dist 1.25 \\
        --discover-dir ir_test_data \\
        --interactive

Directory layout for --discover-dir:
    ir_test_data/
        100cm/   pair_01_cam0.png  pair_01_ir.csv  ...  (5 pairs)
        125cm/   ...
        150cm/   ...
        175cm/   ...
        200cm/   ...
        225cm/   ...
        250cm/   ...
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

IR_WIDTH         = 32
IR_HEIGHT        = 24
CAM_SIZE         = 640
IR_DISPLAY_SCALE = 20    # pixels per IR pixel in the annotation window


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TestCapture:
    label:        str
    distance_m:   float
    cam_png:      str
    ir_csv:       str
    cam_bbox_px:  Optional[Tuple[int, int, int, int]] = None
    # cam_bbox_px  = (u_min, v_min, u_max, v_max) in 640×640 camera pixels
    ir_gt_bbox_px: Optional[Tuple[int, int, int, int]] = None
    # ir_gt_bbox_px = (x_min, y_min, x_max, y_max) in 32×24 IR pixels
    # Both are None until annotated interactively.


TEST_CAPTURES: List[TestCapture] = [
    # Leave empty and use --discover-dir, or fill in manually.
    # TestCapture("100cm_01", 1.00,
    #             "ir_test/100cm/pair_01_cam0.png",
    #             "ir_test/100cm/pair_01_ir.csv"),
]


@dataclass
class CaptureResult:
    capture:         TestCapture
    yaml_used:       str
    proj_rect:       Tuple[int, int, int, int]   # projected bbox in IR pixels (AABB)
    proj_centre:     Tuple[float, float]          # centroid of projected rect
    gt_centre:       Tuple[float, float]          # centroid of manually-drawn IR bbox
    centre_error_px: float                        # Euclidean distance, IR pixels
    ir_temps:        np.ndarray                   # 24×32 temperature array


# ─── Load calibration ────────────────────────────────────────────────────────

def load_homography(yaml_path: str) -> Tuple[np.ndarray, float]:
    fs = cv2.FileStorage(yaml_path, cv2.FileStorage_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration YAML: {yaml_path}")
    H       = fs.getNode("H").mat()
    depth_m = float(fs.getNode("calibration_depth_m").real())
    fs.release()
    if H is None or H.shape != (3, 3):
        raise RuntimeError(f"H must be a 3×3 matrix in {yaml_path}")
    return H.astype(np.float64), depth_m


def select_homography(
    distance_m: float,
    H1: np.ndarray, yaml1: str,
    H2: Optional[np.ndarray], yaml2: Optional[str],
    switch_dist_m: float,
) -> Tuple[np.ndarray, str]:
    if H2 is not None and distance_m > switch_dist_m:
        return H2, yaml2
    return H1, yaml1


# ─── IR utilities ─────────────────────────────────────────────────────────────

def load_ir_csv(csv_path: str) -> np.ndarray:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            rows.append([float(v) for v in row if v.strip()])
    arr = np.array(rows, dtype=np.float32)
    if arr.shape != (IR_HEIGHT, IR_WIDTH):
        raise ValueError(
            f"Expected ({IR_HEIGHT},{IR_WIDTH}), got {arr.shape} in {csv_path}"
        )
    return arr


def render_ir_heatmap(temps: np.ndarray, scale: int = IR_DISPLAY_SCALE) -> np.ndarray:
    lo, hi = float(temps.min()), float(temps.max())
    norm = ((temps - lo) / max(hi - lo, 0.1) * 255).clip(0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    upscaled = cv2.resize(
        coloured,
        (IR_WIDTH * scale, IR_HEIGHT * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.putText(upscaled, f"{lo:.1f}C — {hi:.1f}C",
                (6, upscaled.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return upscaled


# ─── Homography projection ────────────────────────────────────────────────────

def project_bbox(
    H: np.ndarray,
    bbox_px: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Project camera pixel bbox through H into IR pixel space (AABB)."""
    u0, v0, u1, v1 = bbox_px
    corners = np.array([
        [u0 / CAM_SIZE, v0 / CAM_SIZE],
        [u1 / CAM_SIZE, v0 / CAM_SIZE],
        [u1 / CAM_SIZE, v1 / CAM_SIZE],
        [u0 / CAM_SIZE, v1 / CAM_SIZE],
    ], dtype=np.float32).reshape(-1, 1, 2)

    proj = cv2.perspectiveTransform(corners, H.astype(np.float32)).reshape(-1, 2)
    min_x, min_y = proj.min(axis=0)
    max_x, max_y = proj.max(axis=0)

    if max_x < 0 or max_y < 0 or min_x > IR_WIDTH - 1 or min_y > IR_HEIGHT - 1:
        return None

    x0 = int(np.clip(np.floor(min_x), 0, IR_WIDTH  - 1))
    y0 = int(np.clip(np.floor(min_y), 0, IR_HEIGHT - 1))
    x1 = int(np.clip(np.ceil(max_x),  0, IR_WIDTH  - 1))
    y1 = int(np.clip(np.ceil(max_y),  0, IR_HEIGHT - 1))
    return x0, y0, x1, y1


def rect_centre(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = rect
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


# ─── Per-capture processing ───────────────────────────────────────────────────

def process_capture(
    capture: TestCapture,
    H: np.ndarray,
    yaml_used: str,
) -> Optional[CaptureResult]:
    if not Path(capture.cam_png).exists():
        print(f"  [skip] {capture.label}: cam PNG not found")
        return None

    try:
        ir_temps = load_ir_csv(capture.ir_csv)
    except Exception as e:
        print(f"  [skip] {capture.label}: IR CSV error: {e}")
        return None

    if capture.cam_bbox_px is None:
        print(f"  [skip] {capture.label}: no camera bbox — run with --interactive")
        return None

    if capture.ir_gt_bbox_px is None:
        print(f"  [skip] {capture.label}: no IR ground-truth bbox — run with --interactive")
        return None

    proj = project_bbox(H, capture.cam_bbox_px)
    if proj is None:
        print(f"  [warn] {capture.label}: projected bbox entirely outside IR frame")
        return None

    proj_c = rect_centre(proj)
    gt_c   = rect_centre(capture.ir_gt_bbox_px)
    error  = float(np.hypot(proj_c[0] - gt_c[0], proj_c[1] - gt_c[1]))

    return CaptureResult(
        capture          = capture,
        yaml_used        = yaml_used,
        proj_rect        = proj,
        proj_centre      = proj_c,
        gt_centre        = gt_c,
        centre_error_px  = error,
        ir_temps         = ir_temps,
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

def make_result_image(result: CaptureResult, scale: int = IR_DISPLAY_SCALE) -> np.ndarray:
    """
    Three-panel image:
      Left:   640×640 camera frame with the input bbox (green).
      Middle: IR heatmap with projected bbox (green) and GT bbox (red).
      Right:  IR heatmap zoomed to the region of interest.
    """
    # ── Camera panel ──────────────────────────────────────────────────────────
    cam_img = cv2.imread(result.capture.cam_png)
    if cam_img is None:
        cam_img = np.zeros((CAM_SIZE, CAM_SIZE, 3), dtype=np.uint8)
        cv2.putText(cam_img, "not found", (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
    else:
        cam_img = cam_img.copy()

    u0, v0, u1, v1 = result.capture.cam_bbox_px
    cv2.rectangle(cam_img, (u0, v0), (u1, v1), (0, 220, 0), 2)
    cv2.putText(cam_img, result.capture.label,
                (u0, max(v0 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(cam_img,
                f"d={result.capture.distance_m:.2f}m  {Path(result.yaml_used).stem}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ── IR panel ──────────────────────────────────────────────────────────────
    ir_vis = render_ir_heatmap(result.ir_temps, scale)

    # Projected bbox — green
    x0p, y0p, x1p, y1p = result.proj_rect
    cv2.rectangle(ir_vis,
                  (x0p * scale, y0p * scale),
                  ((x1p + 1) * scale - 1, (y1p + 1) * scale - 1),
                  (0, 220, 0), 2)

    # GT bbox — red
    x0g, y0g, x1g, y1g = result.capture.ir_gt_bbox_px
    cv2.rectangle(ir_vis,
                  (x0g * scale, y0g * scale),
                  ((x1g + 1) * scale - 1, (y1g + 1) * scale - 1),
                  (0, 60, 220), 2)

    # Projected centroid — green cross
    pcx, pcy = result.proj_centre
    cv2.drawMarker(ir_vis,
                   (int(pcx * scale + scale // 2), int(pcy * scale + scale // 2)),
                   (0, 220, 0), cv2.MARKER_CROSS, 16, 2)

    # GT centroid — red cross
    gcx, gcy = result.gt_centre
    cv2.drawMarker(ir_vis,
                   (int(gcx * scale + scale // 2), int(gcy * scale + scale // 2)),
                   (0, 60, 220), cv2.MARKER_CROSS, 16, 2)

    # Error annotation
    cv2.putText(ir_vis,
                f"centre err: {result.centre_error_px:.2f} IR px",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ir_vis, "green=projected  red=ground truth",
                (6, ir_vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Combine ───────────────────────────────────────────────────────────────
    h = max(cam_img.shape[0], ir_vis.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] < h:
            p = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, p])
        return img

    div = np.full((h, 3, 3), 200, dtype=np.uint8)
    return np.hstack([pad(cam_img), div, pad(ir_vis)])


# ─── Summary table ────────────────────────────────────────────────────────────

def build_dist_groups(
    results: List[CaptureResult],
) -> Dict[float, List[CaptureResult]]:
    groups: Dict[float, List[CaptureResult]] = {}
    for r in results:
        groups.setdefault(r.capture.distance_m, []).append(r)
    return groups


def print_and_save_summary(
    results: List[CaptureResult],
    out_dir: Path,
    switch_dist_m: Optional[float],
) -> Dict[float, Tuple[float, float]]:
    """Returns {distance_m: (mean_err, std_err)} for the plot."""
    lines = []
    lines.append("IR Alignment Test — Summary")
    if switch_dist_m is not None:
        lines.append(f"Calibration switch: <= {switch_dist_m:.2f} m → yaml1, "
                     f"> {switch_dist_m:.2f} m → yaml2")
    lines.append("Error metric: centre error between projected bbox centroid "
                 "and manually-drawn IR ground-truth bbox centroid (IR pixels)")
    lines.append("")
    lines.append(f"{'Label':<18} {'Dist(m)':>7}  {'YAML':<22}  {'CentreErr(px)':>13}")
    lines.append("─" * 68)

    groups = build_dist_groups(results)
    stats:  Dict[float, Tuple[float, float]] = {}

    for dist in sorted(groups.keys()):
        group = groups[dist]
        for r in group:
            lines.append(f"{r.capture.label:<18} {dist:>7.2f}  "
                         f"{Path(r.yaml_used).stem:<22}  "
                         f"{r.centre_error_px:>13.3f}")

        errs = [r.centre_error_px for r in group]
        if errs:
            mean_e = float(np.mean(errs))
            std_e  = float(np.std(errs))
            stats[dist] = (mean_e, std_e)
            note = ""
            if switch_dist_m is not None:
                note = "  [H1]" if dist <= switch_dist_m else "  [H2]"
            lines.append(
                f"  {'mean ± std':>16} {dist:>7.2f}  {'':22}  "
                f"{mean_e:>10.3f} ± {std_e:.3f}{note}"
            )
        lines.append("")

    all_errs = [r.centre_error_px for r in results]
    if all_errs:
        lines.append(f"Overall mean : {np.mean(all_errs):.3f} IR px")
        lines.append(f"Overall std  : {np.std(all_errs):.3f} IR px")

    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text)
    print(f"\nSummary written to {out_dir / 'summary.txt'}")
    return stats


# ─── Plot ─────────────────────────────────────────────────────────────────────

def save_error_plot(
    stats: Dict[float, Tuple[float, float]],
    out_dir: Path,
    switch_dist_m: Optional[float],
    yaml1_name: str,
    yaml2_name: Optional[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — skipping. "
              "Install with: pip install matplotlib")
        return

    xs     = [d * 100 for d in sorted(stats.keys())]
    means  = [stats[d][0] for d in sorted(stats.keys())]
    stds   = [stats[d][1] for d in sorted(stats.keys())]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(xs, means, yerr=stds,
                fmt='o-', capsize=5, linewidth=2, markersize=6,
                color='steelblue', ecolor='steelblue', elinewidth=1.5,
                label='Mean centre error ± 1 std dev')
    ax.fill_between(xs,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color='steelblue')

    if switch_dist_m is not None and yaml2_name is not None:
        ax.axvline(switch_dist_m * 100, color='tomato', linestyle='--',
                   linewidth=1.5, alpha=0.8,
                   label=(f"Calibration switch\n"
                          f"≤{switch_dist_m*100:.0f}cm: {Path(yaml1_name).stem}\n"
                          f">{switch_dist_m*100:.0f}cm: {Path(yaml2_name).stem}"))

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
               label='1.0 IR px reference')

    ax.set_xlabel('Object distance (cm)', fontsize=12)
    ax.set_ylabel('Centre error (IR pixels)', fontsize=12)
    ax.set_title('IR Projection Centre Error vs Object Distance\n'
                 '(ground truth = manually annotated IR bounding box)',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(min(xs) - 10, max(xs) + 10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / "error_vs_distance.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


# ─── Auto-discovery ──────────────────────────────────────────────────────────

def discover_captures(root_dir: Path) -> List[TestCapture]:
    """
    Scan root_dir for pair_NN_cam0.png / pair_NN_ir.csv.
    Supports distance-named subdirs (100cm/, 125cm/, ...) or flat layout.
    """
    captures: List[TestCapture] = []
    if not root_dir.exists():
        print(f"[discover] not found: {root_dir}")
        return captures

    def parse_dist(name: str) -> float:
        m = re.match(r"^(\d+(?:\.\d+)?)(cm|m)$", name.lower())
        if not m:
            return 0.0
        v = float(m.group(1))
        return v / 100.0 if m.group(2) == "cm" else v

    def scan(directory: Path, dist: float, prefix: str) -> List[TestCapture]:
        found = []
        for cam in sorted(directory.glob("pair_*_cam0.png")):
            stem   = cam.stem.replace("_cam0", "")
            ir_csv = directory / f"{stem}_ir.csv"
            if not ir_csv.exists():
                continue
            label = f"{prefix}{stem}" if prefix else stem
            found.append(TestCapture(label=label, distance_m=dist,
                                     cam_png=str(cam), ir_csv=str(ir_csv)))
        return found

    subdirs      = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    dist_subdirs = [(d, parse_dist(d.name)) for d in subdirs if parse_dist(d.name) > 0]

    if dist_subdirs:
        print(f"[discover] {len(dist_subdirs)} distance subdirectories in {root_dir}")
        for subdir, dist in dist_subdirs:
            pairs = scan(subdir, dist, prefix=f"{subdir.name}_")
            print(f"  {subdir.name}/  ({dist:.2f} m)  →  {len(pairs)} pairs")
            captures.extend(pairs)
    else:
        print(f"[discover] flat directory {root_dir}")
        pairs = scan(root_dir, 0.0, prefix="")
        print(f"  {len(pairs)} pairs  (distance unknown → 0.00 m)")
        captures.extend(pairs)

    return captures


# ─── Interactive annotation ───────────────────────────────────────────────────

def draw_bbox_on_image(
    window_name: str,
    image: np.ndarray,
    prompt: str,
    existing_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Generic click-drag bbox annotation on a single image.
    Returns (x0, y0, x1, y1) or None if the user pressed 's' to skip.
    If existing_bbox is set, it is shown and ENTER immediately confirms it.

    Important Raspberry Pi / OpenCV note:
    Some HighGUI backends, especially the Qt backend on Raspberry Pi, can fail
    to look up a window if the window title contains non-ASCII characters such
    as an em dash. The caller may pass a descriptive title, but internally we
    use a simple ASCII-only window ID for all HighGUI calls.
    """
    state: dict = {
        "start":   None,
        "end":     None,
        "drawing": False,
        "done":    False,
    }
    if existing_bbox is not None:
        x0, y0, x1, y1 = existing_bbox
        state["start"] = (x0, y0)
        state["end"]   = (x1, y1)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"]   = (x, y)
            state["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"]     = (x, y)
            state["drawing"] = False

    # Keep the HighGUI identifier ASCII-only. Do not use the pretty title
    # passed by the caller, because titles like "Camera bbox — 100cm_pair_06"
    # can produce a NULL window handler on OpenCV's Qt backend.
    win = "bbox_annotation"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowTitle(win, window_name)
    except cv2.error:
        # Cosmetic only; the internal ASCII name above is what matters.
        pass

    # Show once and pump the event loop so the backend has created the native
    # window before the callback is registered.
    cv2.imshow(win, image)
    for _ in range(10):
        cv2.waitKey(1)

    try:
        cv2.setMouseCallback(win, on_mouse)
    except cv2.error as e:
        cv2.destroyWindow(win)
        raise RuntimeError(
            "OpenCV created no usable GUI window for mouse input. "
            "Run from a real desktop/VNC/X11 session, make sure DISPLAY is set, "
            "and use opencv-python rather than opencv-python-headless. "
            f"Original OpenCV error: {e}"
        )

    print(f"\n  {prompt}")
    print("  ENTER = accept,  r = redo,  s = skip,  q = quit")

    while True:
        vis = image.copy()
        if state["start"] is not None and state["end"] is not None:
            cv2.rectangle(vis, state["start"], state["end"], (0, 220, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # ENTER
            if state["start"] is not None and state["end"] is not None:
                x0 = min(state["start"][0], state["end"][0])
                y0 = min(state["start"][1], state["end"][1])
                x1 = max(state["start"][0], state["end"][0])
                y1 = max(state["start"][1], state["end"][1])
                cv2.destroyWindow(win)
                return x0, y0, x1, y1
            else:
                print("  Draw a box first.")
        elif key == ord('r'):
            state["start"] = None
            state["end"]   = None
            state["drawing"] = False
        elif key == ord('s'):
            cv2.destroyWindow(win)
            return None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("User quit annotation.")


def ir_pixel_from_display(
    display_x: int,
    display_y: int,
    scale: int,
) -> Tuple[int, int]:
    """Convert a pixel position in the upscaled IR display to an IR pixel coordinate."""
    return int(display_x / scale), int(display_y / scale)


def draw_ir_bbox(
    window_name: str,
    ir_temps: np.ndarray,
    scale: int,
    existing_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Draw a bbox on the upscaled IR heatmap. Coordinates are stored in
    IR pixel space (0..31, 0..23), not in display pixels.
    Returns (ir_x0, ir_y0, ir_x1, ir_y1) or None if skipped.
    """
    heatmap_orig = render_ir_heatmap(ir_temps, scale)

    # Convert existing IR bbox to display coords for pre-population
    existing_display = None
    if existing_bbox is not None:
        ix0, iy0, ix1, iy1 = existing_bbox
        existing_display = (ix0 * scale, iy0 * scale,
                            (ix1 + 1) * scale - 1, (iy1 + 1) * scale - 1)

    result = draw_bbox_on_image(window_name, heatmap_orig,
                                "Draw a box around the hot object in the IR frame.",
                                existing_display)
    if result is None:
        return None

    dx0, dy0, dx1, dy1 = result
    ir_x0 = int(np.clip(dx0 // scale, 0, IR_WIDTH  - 1))
    ir_y0 = int(np.clip(dy0 // scale, 0, IR_HEIGHT - 1))
    ir_x1 = int(np.clip(dx1 // scale, 0, IR_WIDTH  - 1))
    ir_y1 = int(np.clip(dy1 // scale, 0, IR_HEIGHT - 1))

    # Ensure at least a 1×1 region
    if ir_x1 < ir_x0: ir_x1 = ir_x0
    if ir_y1 < ir_y0: ir_y1 = ir_y0

    return ir_x0, ir_y0, ir_x1, ir_y1


def interactive_annotate(captures: List[TestCapture], scale: int) -> None:
    """
    For each capture that is missing either bbox, open the camera image and
    IR heatmap side-by-side and collect both annotations.
    Captions indicate which annotation is expected.
    """
    need_annotation = [
        c for c in captures
        if c.cam_bbox_px is None or c.ir_gt_bbox_px is None
    ]

    if not need_annotation:
        print("[interactive] all captures already annotated — nothing to do.")
        return

    print(f"\n[interactive] annotating {len(need_annotation)} captures.")
    print("  You will be asked to draw TWO bboxes per capture:")
    print("    1. Camera image — the hot object's bounding box")
    print("    2. IR heatmap   — where the object actually appears in the IR\n")
    print("  Controls:  click-drag = draw  |  ENTER = accept  |  r = redo  |  s = skip\n")

    for cap in need_annotation:
        print(f"─── {cap.label}  ({cap.distance_m:.2f} m) ─────────────────────────")

        # ── Step 1: camera bbox ───────────────────────────────────────────────
        if cap.cam_bbox_px is None:
            cam_img = cv2.imread(cap.cam_png)
            if cam_img is None:
                print(f"  skip: cannot read {cap.cam_png}")
                continue
            try:
                result = draw_bbox_on_image(
                    f"[1/2] Camera bbox — {cap.label}",
                    cam_img,
                    "Draw a box around the hot object in the CAMERA image.",
                    existing_bbox=None,
                )
            except KeyboardInterrupt:
                print("\nAnnotation quit by user.")
                return
            if result is None:
                print(f"  skipped camera bbox for {cap.label}")
                continue
            cap.cam_bbox_px = result
            print(f"  cam bbox: {cap.cam_bbox_px}")

        # ── Step 2: IR ground-truth bbox ──────────────────────────────────────
        if cap.ir_gt_bbox_px is None:
            try:
                ir_temps = load_ir_csv(cap.ir_csv)
            except Exception as e:
                print(f"  skip IR annotation: {e}")
                continue
            try:
                result = draw_ir_bbox(
                    f"[2/2] IR ground truth — {cap.label}",
                    ir_temps,
                    scale,
                    existing_bbox=None,
                )
            except KeyboardInterrupt:
                print("\nAnnotation quit by user.")
                return
            if result is None:
                print(f"  skipped IR bbox for {cap.label}")
                cap.cam_bbox_px = None   # roll back — keep pair consistent
                continue
            cap.ir_gt_bbox_px = result
            print(f"  IR gt bbox: {cap.ir_gt_bbox_px}")

    cv2.destroyAllWindows()
    print("\n[interactive] annotation complete.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cam0→IR homography accuracy across multiple depths."
    )
    parser.add_argument("--yaml",         default="ir_alignment.yaml",
                        help="Primary calibration YAML (used for d <= switch-dist)")
    parser.add_argument("--yaml2",        default=None,
                        help="Secondary YAML (used for d > switch-dist).")
    parser.add_argument("--switch-dist",  type=float, default=1.25, metavar="METRES",
                        help="Switch distance in metres (default 1.25). "
                             "Only used when --yaml2 is specified.")
    parser.add_argument("--discover-dir", default=None, metavar="DIR",
                        help="Auto-discover captures from distance subdirs "
                             "(100cm/, 125cm/, ...) or flat layout.")
    parser.add_argument("--out-dir",      default="ir_test_results",
                        help="Output directory for PNGs, summary.txt, and plot.")
    parser.add_argument("--interactive",  action="store_true",
                        help="Annotate camera and IR bboxes interactively.")
    parser.add_argument("--scale",        type=int, default=IR_DISPLAY_SCALE,
                        help=f"IR upscale factor in annotation/output windows "
                             f"(default {IR_DISPLAY_SCALE})")
    args = parser.parse_args()

    # ── Load calibration(s) ───────────────────────────────────────────────────
    try:
        H1, d1 = load_homography(args.yaml)
    except RuntimeError as e:
        print(f"ERROR: {e}"); return 1
    print(f"Loaded H1 from {args.yaml}  (calibrated at {d1:.2f} m)")

    H2, yaml2_path = None, None
    if args.yaml2:
        try:
            H2, d2 = load_homography(args.yaml2)
            yaml2_path = args.yaml2
        except RuntimeError as e:
            print(f"ERROR loading yaml2: {e}"); return 1
        print(f"Loaded H2 from {args.yaml2}  (calibrated at {d2:.2f} m)")
        print(f"Switch: <= {args.switch_dist:.2f} m → H1,  > {args.switch_dist:.2f} m → H2")

    switch_dist = args.switch_dist if H2 is not None else None

    # ── Build capture list ────────────────────────────────────────────────────
    captures = list(TEST_CAPTURES)
    if args.discover_dir is not None:
        discovered = discover_captures(Path(args.discover_dir))
        if discovered:
            captures.extend(discovered)
        else:
            print(f"[discover] no valid pairs found in {args.discover_dir}")

    if not captures:
        print("\nNo captures. Use --discover-dir or edit TEST_CAPTURES.")
        return 1

    # ── Interactive annotation ────────────────────────────────────────────────
    if args.interactive:
        interactive_annotate(captures, args.scale)

    # ── Process ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[CaptureResult] = []
    print(f"\nProcessing {len(captures)} captures...")

    for i, cap in enumerate(captures):
        H, yaml_used = select_homography(
            cap.distance_m, H1, args.yaml, H2, yaml2_path, args.switch_dist
        )
        print(f"  [{i+1:2d}/{len(captures)}] {cap.label:<18}  "
              f"d={cap.distance_m:.2f}m  H={Path(yaml_used).stem}")

        r = process_capture(cap, H, yaml_used)
        if r is None:
            continue
        results.append(r)

        out_img  = make_result_image(r, scale=args.scale)
        img_path = out_dir / f"{cap.label}_result.png"
        cv2.imwrite(str(img_path), out_img)
        print(f"       centre_err={r.centre_error_px:.3f} px  → {img_path.name}")

    if not results:
        print("\nNo results. Run with --interactive to annotate bboxes first.")
        return 1

    # ── Summary + plot ────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    stats = print_and_save_summary(results, out_dir, switch_dist)
    save_error_plot(stats, out_dir, switch_dist, args.yaml, yaml2_path)

    print(f"\nAll outputs saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
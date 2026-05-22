#!/usr/bin/env python3
"""
Microparticle analysis for 7 user-annotated targets (regionprops / MATLAB-equivalent).

Annotation 2 is a conjoined trio of small spheres; watershed splits them into
three separate measurements while preserving annotation_id=2.

Outputs:
  - output/particles/particle_XXX.png (and particle_002_01..03 for the trio)
  - output/particle_measurements.xlsx
  - output/particle_metrics_visualization.png
  - output/segmentation_overview.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage import color, filters, io, measure, morphology, segmentation
from skimage.feature import canny, peak_local_max
from skimage.morphology import disk
from skimage.transform import hough_circle, hough_circle_peaks

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "input_image.png"
OUTPUT_DIR = ROOT / "output"
PARTICLE_DIR = OUTPUT_DIR / "particles"

SCALE_UM = 100.0
CROP_FRACTION = 0.88

# User annotation search windows (y0, y1, x0, x1).
PARTICLE_WINDOWS: dict[int, tuple[int, int, int, int]] = {
    1: (0, 120, 0, 160),
    2: (60, 180, 120, 300),  # conjoined trio
    3: (240, 420, 0, 100),
    4: (240, 400, 450, 600),
    5: (320, 480, 350, 500),
    # Bottom edge: large sphere partially cut off by frame (not the dim patch above).
    6: (580, 676, 680, 880),
    7: (150, 280, 700, 950),
}

# Annotations that contain multiple touching spheres.
COMBINED_ANNOTATIONS: set[int] = {2}

# Spheres truncated by image borders.
CLIP_LEFT_ANNOTATIONS: set[int] = {3}
CLIP_BOTTOM_ANNOTATIONS: set[int] = {6}


def load_grayscale(path: Path) -> np.ndarray:
    raw = io.imread(path)
    if raw.ndim == 3 and raw.shape[-1] == 4:
        alpha = raw[..., 3].astype(np.float64) / 255.0
        gray = color.rgb2gray(raw[..., :3])
        gray = gray * alpha + (1.0 - alpha)
    elif raw.ndim == 3:
        gray = color.rgb2gray(raw)
    else:
        gray = raw.astype(np.float64)
        if gray.max() > 1:
            gray /= 255.0
    return gray.astype(np.float64)


def detect_scale_um_per_pixel(gray: np.ndarray) -> tuple[float, int]:
    h, _ = gray.shape
    bar_strip = gray[int(h * 0.90) :, :]
    dark = bar_strip < 0.35
    best_len = 90
    for y in range(dark.shape[0]):
        row = dark[y]
        diff = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if 40 <= length <= 250 and abs(length - 90) < abs(best_len - 90):
                best_len = length
    return SCALE_UM / best_len, best_len


def detect_hough_circles(roi: np.ndarray) -> list[tuple[float, int, int, int]]:
    edges = canny(roi, sigma=2, low_threshold=0.08, high_threshold=0.2)
    radii = np.arange(8, 72, 2)
    hspaces = hough_circle(edges, radii)
    acc, cx, cy, r = hough_circle_peaks(
        hspaces, radii, total_num_peaks=30, min_xdistance=28, min_ydistance=28
    )
    return [(float(a), int(cy_i), int(cx_i), int(r_i)) for a, cx_i, cy_i, r_i in zip(acc, cx, cy, r)]


def best_hough_in_window(
    circles: list[tuple[float, int, int, int]],
    window: tuple[int, int, int, int],
) -> tuple[int, int, int] | None:
    y0, y1, x0, x1 = window
    best = None
    for acc, cy, cx, r in circles:
        if y0 <= cy < y1 and x0 <= cx < x1:
            if best is None or acc > best[0]:
                best = (acc, cy, cx, r)
    if best is None:
        return None
    return best[1], best[2], best[3]


def estimate_radius_background_edge(
    roi: np.ndarray, cy: int, cx: int, r_min: int, r_max: int
) -> int:
    """Radius where radial mean intensity drops to background level."""
    bg = float(np.percentile(roi, 30))
    thr = bg + 0.06
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    h, w = roi.shape
    for r in range(r_max, r_min - 1, -1):
        vals = []
        for a in angles:
            y = int(round(cy + r * np.sin(a)))
            x = int(round(cx + r * np.cos(a)))
            if 0 <= y < h and 0 <= x < w:
                vals.append(roi[y, x])
        if vals and np.mean(vals) >= thr:
            return int(np.clip(r, r_min, r_max))
    return r_min


def circle_mask(
    shape: tuple[int, int],
    cy: int,
    cx: int,
    radius: int,
    *,
    clip_left: bool = False,
    clip_bottom: bool = False,
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    if clip_left:
        mask &= xx >= 0
    if clip_bottom:
        mask &= yy < shape[0]
    return mask


def find_bright_centroid(
    roi: np.ndarray, window: tuple[int, int, int, int], min_brightness: float = 0.35
) -> tuple[int, int]:
    """Brightest point in window (for dim or edge-truncated spheres)."""
    y0, y1, x0, x1 = window
    patch = filters.gaussian(roi, sigma=1.2)[y0:y1, x0:x1]
    idx = np.unravel_index(int(np.argmax(patch)), patch.shape)
    cy, cx = y0 + idx[0], x0 + idx[1]
    if patch[idx] < min_brightness:
        raise RuntimeError(f"No bright sphere found in window {window}")
    return cy, cx


def split_trio_annotation(
    roi: np.ndarray, window: tuple[int, int, int, int]
) -> list[tuple[np.ndarray, int, int, int]]:
    """Watershed split for three conjoined spheres; returns (mask, cy, cx, radius)."""
    y0, y1, x0, x1 = window
    patch = roi[y0:y1, x0:x1]
    tophat = morphology.white_tophat((patch * 255).astype(np.uint8), disk(5)).astype(
        np.float64
    ) / 255.0
    peaks = peak_local_max(
        tophat, min_distance=12, threshold_abs=0.04, num_peaks=5
    )
    if len(peaks) < 3:
        peaks = peak_local_max(
            tophat, min_distance=10, threshold_abs=0.03, num_peaks=5
        )

    bw = tophat > 0.03
    bw = morphology.opening(bw, disk(1))
    markers = np.zeros_like(bw, dtype=np.int32)
    peak_vals = [(tophat[y, x], y, x) for y, x in peaks]
    peak_vals.sort(reverse=True)
    for i, (_, y, x) in enumerate(peak_vals[:3], start=1):
        markers[y, x] = i

    dist = ndi.distance_transform_edt(bw)
    ws = segmentation.watershed(-dist, markers, mask=bw)

    spheres: list[tuple[np.ndarray, int, int, int]] = []
    for lab in range(1, 4):
        comp = ws == lab
        if not comp.any() or comp.sum() < 30:
            continue
        props = measure.regionprops(comp.astype(np.uint8))[0]
        cy = int(props.centroid[0] + y0)
        cx = int(props.centroid[1] + x0)
        radius = int(max(6, min(22, np.sqrt(props.area / np.pi) + 1)))
        full = np.zeros(roi.shape, dtype=bool)
        full[y0:y1, x0:x1] = comp
        spheres.append((full, cy, cx, radius))

    if len(spheres) < 3:
        circles = detect_hough_circles(roi)
        h = best_hough_in_window(circles, window)
        if h:
            cy, cx, radius = h
            spheres = [(circle_mask(roi.shape, cy, cx, radius), cy, cx, radius)]
    return spheres


def assign_mask(labels: np.ndarray, mask: np.ndarray, label_id: int) -> None:
    """Assign only unlabeled pixels so overlapping circles do not corrupt neighbors."""
    labels[(mask) & (labels == 0)] = label_id


def build_masks_and_metadata(
    roi: np.ndarray, circles: list[tuple[float, int, int, int]]
) -> tuple[np.ndarray, list[dict]]:
    labels = np.zeros(roi.shape, dtype=np.int16)
    meta: list[dict] = []
    next_label = 1

    for ann_id in sorted(PARTICLE_WINDOWS):
        window = PARTICLE_WINDOWS[ann_id]

        if ann_id in COMBINED_ANNOTATIONS:
            spheres = split_trio_annotation(roi, window)
            for sub_idx, (mask, cy, cx, radius) in enumerate(spheres, start=1):
                assign_mask(labels, mask, next_label)
                meta.append(
                    {
                        "label": next_label,
                        "annotation_id": ann_id,
                        "sub_particle": sub_idx,
                        "is_combined_annotation": True,
                        "notes": "Conjoined trio member (annotation 2)",
                        "cy": cy,
                        "cx": cx,
                        "radius_px": radius,
                    }
                )
                next_label += 1
            continue

        if ann_id == 6:
            # Bottom cut-off sphere: bright spot is the upper surface; center sits below
            # the frame edge, with the sphere tangent to the bottom of the ROI.
            cap_y, cap_x = find_bright_centroid(roi, window, min_brightness=0.45)
            cx = cap_x
            radius = estimate_radius_background_edge(roi, cap_y, cap_x, 16, 45)
            hough = best_hough_in_window(circles, window)
            if hough:
                hcy, hcx, hr = hough
                if abs(hcy - cap_y) < 55 and abs(hcx - cap_x) < 55:
                    radius = max(radius, hr)
            # Widen radius using visible bright cap extent (arc is wider than radial drop).
            y0, y1, x0, x1 = window
            patch = roi[y0:y1, x0:x1] > 0.42
            if patch.any():
                ys, xs = np.where(patch)
                cap_radius = int(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 6)
                radius = max(radius, cap_radius)
            cy = roi.shape[0] - radius - 1
            method = "bottom_truncated_sphere"
        else:
            hough = best_hough_in_window(circles, window)
            if hough:
                cy, cx, radius = hough
                method = "hough"
            else:
                smooth = filters.gaussian(roi, sigma=1.5)
                sub = smooth[window[0] : window[1], window[2] : window[3]]
                idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
                cy = window[0] + idx[0]
                cx = window[2] + idx[1]
                r_max = 70 if ann_id in (3, 4, 7) else 40
                radius = estimate_radius_background_edge(roi, cy, cx, 8, r_max)
                method = "radial_background_edge"

        mask = circle_mask(
            roi.shape,
            cy,
            cx,
            radius,
            clip_left=ann_id in CLIP_LEFT_ANNOTATIONS,
            clip_bottom=ann_id in CLIP_BOTTOM_ANNOTATIONS,
        )
        assign_mask(labels, mask, next_label)
        meta.append(
            {
                "label": next_label,
                "annotation_id": ann_id,
                "sub_particle": np.nan,
                "is_combined_annotation": False,
                "notes": f"Segmentation: {method}",
                "cy": cy,
                "cx": cx,
                "radius_px": radius,
            }
        )
        next_label += 1

    return labels, meta


def regionprops_table(
    labels: np.ndarray, intensity: np.ndarray, um_per_px: float, meta: list[dict]
) -> pd.DataFrame:
    props = measure.regionprops_table(
        labels,
        intensity_image=intensity,
        properties=(
            "label",
            "area",
            "perimeter",
            "bbox",
            "centroid",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "eccentricity",
            "solidity",
            "extent",
            "equivalent_diameter_area",
            "euler_number",
            "feret_diameter_max",
            "inertia_tensor_eigvals",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
        ),
    )
    df = pd.DataFrame(props)
    # Diameter from segmented pixel area (robust for circle masks).
    df["equivalent_diameter"] = 2.0 * np.sqrt(df["area"] / np.pi)

    meta_df = pd.DataFrame(meta)
    df = df.merge(meta_df, on="label", how="left")
    df = df.sort_values(["annotation_id", "sub_particle"], na_position="first").reset_index(
        drop=True
    )

    df["display_id"] = df.apply(
        lambda r: (
            f"{int(r['annotation_id'])}-{int(r['sub_particle'])}"
            if pd.notna(r["sub_particle"])
            else str(int(r["annotation_id"]))
        ),
        axis=1,
    )

    px_to_um = um_per_px
    px2_to_um2 = um_per_px**2
    for col in (
        "major_axis_length",
        "minor_axis_length",
        "equivalent_diameter",
        "feret_diameter_max",
        "perimeter",
    ):
        df[f"{col}_um"] = df[col] * px_to_um
    df["area_um2"] = df["area"] * px2_to_um2
    df["aspect_ratio"] = df["major_axis_length"] / np.maximum(
        df["minor_axis_length"], 1e-9
    )
    df["circularity"] = (4.0 * math.pi * df["area"]) / np.maximum(
        df["perimeter"] ** 2, 1e-9
    )
    df["orientation_deg"] = np.degrees(df["orientation"])
    df["inertia_ratio"] = df["inertia_tensor_eigvals-0"] / np.maximum(
        df["inertia_tensor_eigvals-1"], 1e-9
    )
    return df


def annotation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ann_id, grp in df.groupby("annotation_id"):
        rows.append(
            {
                "annotation_id": int(ann_id),
                "n_spheres_measured": len(grp),
                "combined_cluster": bool(grp["is_combined_annotation"].iloc[0]),
                "mean_diameter_um": grp["equivalent_diameter_um"].mean(),
                "total_area_um2": grp["area_um2"].sum(),
                "notes": (
                    "Three conjoined spheres split by watershed"
                    if ann_id == 2
                    else "Single sphere"
                ),
            }
        )
    return pd.DataFrame(rows)


def axis_endpoints(
    cy: float, cx: float, length: float, orientation_rad: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = 0.5 * length * math.cos(orientation_rad)
    dy = 0.5 * length * math.sin(orientation_rad)
    return (cx - dx, cy - dy), (cx + dx, cy + dy)


def png_name(row: pd.Series) -> str:
    if pd.notna(row["sub_particle"]):
        return f"particle_{int(row['annotation_id']):03d}_{int(row['sub_particle']):02d}.png"
    return f"particle_{int(row['annotation_id']):03d}.png"


def save_particle_png(roi: np.ndarray, row: pd.Series, out_path: Path, pad: int = 20) -> None:
    min_row, min_col, max_row, max_col = (
        int(row["bbox-0"]),
        int(row["bbox-1"]),
        int(row["bbox-2"]),
        int(row["bbox-3"]),
    )
    min_row = max(0, min_row - pad)
    min_col = max(0, min_col - pad)
    max_row = min(roi.shape[0], max_row + pad)
    max_col = min(roi.shape[1], max_col + pad)

    crop = roi[min_row:max_row, min_col:max_col]
    if "cy" in row and pd.notna(row.get("cy")):
        cy, cx = float(row["cy"]), float(row["cx"])
    else:
        cy, cx = row["centroid-0"], row["centroid-1"]
    cy_local, cx_local = cy - min_row, cx - min_col
    title = f"Particle {row['display_id']}"

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(crop, cmap="gray", vmin=0, vmax=1)

    bb = Rectangle(
        (row["bbox-1"] - min_col, row["bbox-0"] - min_row),
        row["bbox-3"] - row["bbox-1"],
        row["bbox-2"] - row["bbox-0"],
        fill=False,
        edgecolor="lime",
        linewidth=1.8,
    )
    ax.add_patch(bb)

    maj_p0, maj_p1 = axis_endpoints(
        cy_local, cx_local, row["major_axis_length"], row["orientation"]
    )
    min_p0, min_p1 = axis_endpoints(
        cy_local, cx_local, row["minor_axis_length"], row["orientation"] + math.pi / 2
    )
    ax.plot(
        [maj_p0[0], maj_p1[0]],
        [maj_p0[1], maj_p1[1]],
        color="red",
        linewidth=2.0,
        label=f"Major: {row['major_axis_length_um']:.1f} µm",
    )
    ax.plot(
        [min_p0[0], min_p1[0]],
        [min_p0[1], min_p1[1]],
        color="cyan",
        linewidth=2.0,
        label=f"Minor: {row['minor_axis_length_um']:.1f} µm",
    )
    ax.scatter([cx_local], [cy_local], s=18, c="yellow", edgecolors="black", linewidths=0.5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_overview(roi: np.ndarray, labels: np.ndarray, df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.imshow(roi, cmap="gray", vmin=0, vmax=1)
    for _, row in df.iterrows():
        lab = int(row["label"])
        mask = labels == lab
        ax.contour(mask, levels=[0.5], colors=["lime"], linewidths=1.2)
        y0, x0, y1, x1 = row["bbox-0"], row["bbox-1"], row["bbox-2"], row["bbox-3"]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "lime", linewidth=1.0, alpha=0.7)
        if "cy" in row and pd.notna(row.get("cy")):
            cy, cx = float(row["cy"]), float(row["cx"])
        else:
            cy, cx = row["centroid-0"], row["centroid-1"]
        maj_p0, maj_p1 = axis_endpoints(cy, cx, row["major_axis_length"], row["orientation"])
        min_p0, min_p1 = axis_endpoints(
            cy, cx, row["minor_axis_length"], row["orientation"] + math.pi / 2
        )
        ax.plot([maj_p0[0], maj_p1[0]], [maj_p0[1], maj_p1[1]], "r-", linewidth=1.2)
        ax.plot([min_p0[0], min_p1[0]], [min_p0[1], min_p1[1]], "c-", linewidth=1.2)
        ax.text(
            x0,
            y0 - 4,
            str(row["display_id"]),
            color="yellow",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Annotated microparticles (trio 2-1/2-2/2-3 split from conjoined cluster)")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics_visualization(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)

    axes[0, 0].hist(
        df["equivalent_diameter_um"],
        bins=max(5, len(df)),
        color="#4C72B0",
        edgecolor="white",
    )
    axes[0, 0].set_xlabel("Equivalent diameter (µm)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Size distribution (individual spheres)")

    lim = max(df["major_axis_length_um"].max(), df["minor_axis_length_um"].max()) * 1.08
    axes[0, 1].scatter(
        df["minor_axis_length_um"],
        df["major_axis_length_um"],
        c=df["annotation_id"],
        cmap="tab10",
        s=90,
        edgecolors="k",
        linewidths=0.4,
    )
    axes[0, 1].plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1)
    axes[0, 1].set_xlabel("Minor axis (µm)")
    axes[0, 1].set_ylabel("Major axis (µm)")
    axes[0, 1].set_title("Major vs minor axis")
    for _, row in df.iterrows():
        axes[0, 1].annotate(
            row["display_id"],
            (row["minor_axis_length_um"], row["major_axis_length_um"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    colors = df["is_combined_annotation"].map({True: "#DD8452", False: "#4C72B0"})
    axes[1, 0].scatter(
        df["area_um2"],
        df["equivalent_diameter_um"],
        c=colors,
        s=90,
        edgecolors="k",
        linewidths=0.4,
    )
    axes[1, 0].set_xlabel("Area (µm²)")
    axes[1, 0].set_ylabel("Equivalent diameter (µm)")
    axes[1, 0].set_title("Area vs diameter (orange = trio members)")

    summary = annotation_summary(df)
    axes[1, 1].bar(
        summary["annotation_id"].astype(str),
        summary["mean_diameter_um"],
        color="#55A868",
        edgecolor="white",
    )
    axes[1, 1].set_xlabel("User annotation ID")
    axes[1, 1].set_ylabel("Mean diameter (µm)")
    axes[1, 1].set_title("Mean diameter per annotation (trio averaged)")

    fig.suptitle("Microparticle metrics (7 annotations, 9 spheres measured)", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(image_path: Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure annotated spherical microparticles in SEM micrographs."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=image_path or DEFAULT_IMAGE,
        help="Input micrograph (PNG/TIF). Default: ./input_image.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for results (default: ./output)",
    )
    args = parser.parse_args()
    image_file = args.image
    out_dir = args.output_dir
    particle_dir = out_dir / "particles"

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_file}\n"
            "Place your micrograph at input_image.png or pass --image PATH"
        )

    out_dir.mkdir(exist_ok=True)
    if particle_dir.exists():
        for old_png in particle_dir.glob("particle_*.png"):
            old_png.unlink()
    particle_dir.mkdir(exist_ok=True)

    gray = load_grayscale(image_file)
    um_per_px, scale_px = detect_scale_um_per_pixel(gray)
    crop_h = int(gray.shape[0] * CROP_FRACTION)
    roi = gray[:crop_h, :]

    hough_circles = detect_hough_circles(roi)
    labels, sphere_meta = build_masks_and_metadata(roi, hough_circles)
    df = regionprops_table(labels, roi, um_per_px, sphere_meta)
    ann_summary = annotation_summary(df)

    for _, row in df.iterrows():
        save_particle_png(roi, row, particle_dir / png_name(row))

    excel_path = out_dir / "particle_measurements.xlsx"
    meta = pd.DataFrame(
        {
            "parameter": [
                "source_image",
                "annotation",
                "scale_bar_um",
                "scale_bar_pixels",
                "um_per_pixel",
                "annotations_count",
                "spheres_measured",
                "combined_annotation_ids",
                "segmentation_method",
            ],
            "value": [
                image_file.name,
                "User labels 1-7; label 2 = conjoined trio split by watershed",
                SCALE_UM,
                scale_px,
                um_per_px,
                7,
                len(df),
                "2",
                "Hough circles + background radial edge + watershed (trio)",
            ],
        }
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="metadata", index=False)
        df.to_excel(writer, sheet_name="measurements", index=False)
        ann_summary.to_excel(writer, sheet_name="annotation_summary", index=False)

    save_overview(roi, labels, df, out_dir / "segmentation_overview.png")
    save_metrics_visualization(df, out_dir / "particle_metrics_visualization.png")

    print(f"Calibration: {um_per_px:.4f} µm/pixel ({scale_px} px = {SCALE_UM} µm)")
    print(f"Measured {len(df)} spheres across 7 user annotations")
    print(f"  (annotation 2 split into {len(df[df['annotation_id']==2])} conjoined spheres)")
    print(f"Excel: {excel_path}")
    print(f"PNGs: {particle_dir}")


if __name__ == "__main__":
    main()

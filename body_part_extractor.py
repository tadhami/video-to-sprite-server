"""Extract individual body part PNGs from a magenta-bounded sheet.

The input is an image where each body part is enclosed by a solid colored
rectangle (default: magenta #FF00FF) on a white background. This is the
asset-prep format we use for Spine 2D rigging: a single AI-generated sheet
of all the character's body parts that needs to be sliced into separate
PNGs with transparent backgrounds.

Output: one PNG per detected rectangle, cropped tight to the part's content
with a small transparent padding, plus a labeled debug image showing which
rectangle got which number (for human spot-check / renaming).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _hue_mask(
    arr: np.ndarray,
    target_rgb: tuple[int, int, int],
    hue_tolerance_deg: float = 30.0,
    min_saturation: float = 0.35,
) -> np.ndarray:
    """Boolean mask: pixels whose hue is within `hue_tolerance_deg` of the
    target color's hue AND have at least `min_saturation` saturation.

    Hue-based detection is far more robust to AI-rendering color drift than
    RGB-distance: a target of pure magenta `#FF00FF` (hue 300°) will match
    any rendered shade in roughly [270°, 330°] regardless of how saturated
    or bright the actual pixels are. Critically, it does NOT match pinkish
    skin tones — those have low saturation, so the saturation gate
    excludes them.
    """
    # Compute target hue
    tr, tg, tb = target_rgb
    target_max = max(tr, tg, tb)
    target_min = min(tr, tg, tb)
    target_diff = target_max - target_min
    if target_diff == 0:
        # gray/white target — fall back to RGB distance
        r = arr[..., 0].astype(int); g = arr[..., 1].astype(int); b = arr[..., 2].astype(int)
        return (np.abs(r - tr) + np.abs(g - tg) + np.abs(b - tb)) < 80
    if target_max == tr:
        target_hue = 60.0 * (((tg - tb) / target_diff) % 6)
    elif target_max == tg:
        target_hue = 60.0 * ((tb - tr) / target_diff + 2)
    else:
        target_hue = 60.0 * ((tr - tg) / target_diff + 4)

    # Vectorized HSV for the image
    r = arr[..., 0].astype(float)
    g = arr[..., 1].astype(float)
    b = arr[..., 2].astype(float)
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    diff = max_c - min_c
    saturation = np.where(max_c > 0, diff / np.maximum(max_c, 1), 0)

    # Pixel hue (degrees)
    hue = np.zeros_like(max_c)
    mr = (max_c == r) & (diff > 0)
    mg = (max_c == g) & (diff > 0)
    mb = (max_c == b) & (diff > 0)
    hue[mr] = 60.0 * (((g[mr] - b[mr]) / diff[mr]) % 6)
    hue[mg] = 60.0 * ((b[mg] - r[mg]) / diff[mg] + 2)
    hue[mb] = 60.0 * ((r[mb] - g[mb]) / diff[mb] + 4)

    # Circular hue distance
    hue_dist = np.abs(hue - target_hue)
    hue_dist = np.minimum(hue_dist, 360.0 - hue_dist)

    return (hue_dist < hue_tolerance_deg) & (saturation >= min_saturation)


def _bboxes_overlap_or_touch(a: dict, b: dict, max_gap: int = 0) -> tuple[bool, bool]:
    """Return (x_overlap, y_overlap) where each is True if the bboxes overlap
    or are within `max_gap` pixels on that axis."""
    x_overlap = not (a["x_max"] + max_gap < b["x_min"] or b["x_max"] + max_gap < a["x_min"])
    y_overlap = not (a["y_max"] + max_gap < b["y_min"] or b["y_max"] + max_gap < a["y_min"])
    return x_overlap, y_overlap


def _merge_fragmented_bboxes(
    rects: list[dict],
    max_gap: int = 25,
    fragment_thresh: int = 50,
) -> list[dict]:
    """Merge bboxes that are fragments of the same broken rectangle.

    Two fragments belong to the same rectangle when their bboxes overlap in
    one axis AND have a small gap (< `max_gap` pixels) in the other. This
    catches the common failure mode where a rectangle outline has a gap in
    its perimeter caused by anti-aliasing or model rendering noise, which
    splits the outline into two connected components stacked next to each
    other.

    Pure overlap (one bbox inside another) is NOT merged here — that's the
    "phantom component" case, handled by `_dedup_contained_bboxes` next.
    """
    if len(rects) <= 1:
        return rects

    # Build union-find for merge groups
    parent = list(range(len(rects)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def _is_thin(r: dict) -> bool:
        """A component is a 'fragment' if its smaller bbox dim is tiny.
        Plausible body part rectangles have both dims well above this threshold."""
        w = r["x_max"] - r["x_min"]
        h = r["y_max"] - r["y_min"]
        return min(w, h) < fragment_thresh

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            # Only consider merging if AT LEAST ONE bbox looks like a fragment.
            # Two plausibly-sized rectangles next to each other are real
            # neighbors (e.g., head sitting just above the torso), NOT
            # fragments of one rectangle — never merge those.
            if not (_is_thin(a) or _is_thin(b)):
                continue
            x_overlap, y_overlap = _bboxes_overlap_or_touch(a, b, max_gap=max_gap)
            # Only merge "side-by-side" fragments: one axis overlaps and the
            # other has a small gap. Don't merge if one fully contains the other.
            x_inside = (a["x_min"] >= b["x_min"] and a["x_max"] <= b["x_max"]) or \
                       (b["x_min"] >= a["x_min"] and b["x_max"] <= a["x_max"])
            y_inside = (a["y_min"] >= b["y_min"] and a["y_max"] <= b["y_max"]) or \
                       (b["y_min"] >= a["y_min"] and b["y_max"] <= a["y_max"])
            if x_inside and y_inside:
                continue  # one is contained in the other — dedup will handle
            if x_overlap and y_overlap:
                # touching or overlapping but not fully contained — likely a
                # fragmented rectangle outline
                union(i, j)

    # Build merged groups
    groups: dict[int, list[int]] = {}
    for i in range(len(rects)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged: list[dict] = []
    for indices in groups.values():
        members = [rects[i] for i in indices]
        merged.append({
            "id": members[0]["id"],
            "x_min": min(m["x_min"] for m in members),
            "x_max": max(m["x_max"] for m in members),
            "y_min": min(m["y_min"] for m in members),
            "y_max": max(m["y_max"] for m in members),
        })
    if len(merged) < len(rects):
        logger.info(
            "_merge_fragmented_bboxes: merged %d fragments into %d rectangles",
            len(rects), len(merged),
        )
    return merged


def _dedup_contained_bboxes(rects: list[dict]) -> list[dict]:
    """Drop bboxes whose center falls inside a larger bbox.

    Phantom components created by character pixels matching the boundary
    color (e.g. button eyes inside the head's rectangle) produce small bboxes
    that sit fully inside the real rectangle's bbox. We sort by area
    descending and drop any subsequent bbox whose center is inside a kept
    (larger) bbox.
    """
    if len(rects) <= 1:
        return rects

    sorted_rects = sorted(
        rects,
        key=lambda r: -((r["x_max"] - r["x_min"]) * (r["y_max"] - r["y_min"])),
    )
    kept: list[dict] = []
    for r in sorted_rects:
        cx = (r["x_min"] + r["x_max"]) // 2
        cy = (r["y_min"] + r["y_max"]) // 2
        is_phantom = any(
            k["x_min"] <= cx <= k["x_max"] and k["y_min"] <= cy <= k["y_max"]
            for k in kept
        )
        if is_phantom:
            continue
        kept.append(r)
    if len(kept) < len(rects):
        logger.info(
            "_dedup_contained_bboxes: dropped %d phantom bbox(es) contained in larger ones",
            len(rects) - len(kept),
        )
    return kept


def extract_body_parts_from_sheet(
    input_path: str,
    output_dir: str,
    boundary_color: tuple[int, int, int] = (255, 0, 255),
    boundary_tolerance: int = 80,
    padding: int = 10,
    background_threshold: int = 240,
    min_rect_pixels: int = 200,
) -> dict:
    """Slice a body-parts sheet into individual transparent PNGs.

    Algorithm:
      1. Find pixels close to `boundary_color` (within `boundary_tolerance`).
      2. Morphologically close small gaps in the boundary lines.
      3. Connected-components labeling — each label = one rectangle outline.
         Drop components smaller than `min_rect_pixels` to ignore noise specks.
      4. For each rectangle's bbox, slice the image's interior (inset a few
         pixels to skip the boundary line).
      5. Within the interior, convert near-white AND any residual boundary-color
         pixels to alpha=0; everything else stays opaque.
      6. Crop tight to the alpha bbox, add `padding` transparent pixels around.
      7. Save as `part_NN.png` numbered in reading order (top-to-bottom,
         left-to-right). Generate a labeled debug image alongside.

    Returns a dict with: count, parts[], debug_image, output_dir.
    Each parts[i] entry has: path, index, rectangle_bbox, content_size.

    Errors raise `RuntimeError` with actionable messages.
    """
    from scipy.ndimage import label, binary_closing

    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]

    # 1. Hue-based boundary detection. RGB-distance from a target color is
    #    confounded by lighting/saturation drift and tends to match pinkish
    #    skin tones at high tolerance. Hue is invariant: magenta sits at
    #    ~300°, character colors (browns, blues, greens) are far from there.
    #    Build a boundary mask = pixels whose hue is within `hue_tolerance`
    #    of the target color's hue AND have meaningful saturation.
    boundary_mask = _hue_mask(arr, boundary_color, hue_tolerance_deg=30, min_saturation=0.35)

    if not boundary_mask.any():
        raise RuntimeError(
            f"No boundary pixels detected for color {boundary_color}. Is the "
            f"boundary really drawn in this hue? For a magenta boundary "
            f"pass boundary_color=(255, 0, 255)."
        )

    # 2. Light morphological closing to bridge anti-aliasing gaps in the outlines
    boundary_mask = binary_closing(boundary_mask, iterations=2)

    # 3. Label connected components of the boundary mask — each labeled blob
    #    is one rectangle outline (or a fragment of one).
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_components = label(boundary_mask, structure=structure)

    rectangles: list[dict] = []
    for component_id in range(1, n_components + 1):
        component_pixels = labeled == component_id
        if component_pixels.sum() < min_rect_pixels:
            continue
        ys, xs = np.where(component_pixels)
        rectangles.append({
            "id": component_id,
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
        })

    if not rectangles:
        raise RuntimeError(
            f"Detected boundary-colored pixels but no rectangles of at least "
            f"{min_rect_pixels} px were found. Lower `min_rect_pixels` or "
            f"check that the rectangles are unbroken."
        )

    # 3a. Merge bboxes that look like fragments of the same rectangle.
    rectangles = _merge_fragmented_bboxes(rectangles, max_gap=18)

    # 3b. Drop bboxes whose center falls inside a larger bbox.
    rectangles = _dedup_contained_bboxes(rectangles)

    # 4. Sort by row (bucketed by y_min // row_height) then by x_min
    # Row height heuristic: 1/3 of the median rectangle height groups same-row
    # pieces even when their y_min varies slightly.
    median_h = int(np.median([r["y_max"] - r["y_min"] for r in rectangles]))
    row_bucket = max(50, median_h // 3)
    rectangles.sort(key=lambda r: (r["y_min"] // row_bucket, r["x_min"]))

    # 5. Extract each rectangle's content
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: list[dict] = []
    inset = 5  # skip the magenta line + a couple of safety pixels
    for i, rect in enumerate(rectangles):
        y0 = max(0, rect["y_min"] + inset)
        y1 = min(H, rect["y_max"] - inset + 1)
        x0 = max(0, rect["x_min"] + inset)
        x1 = min(W, rect["x_max"] - inset + 1)
        if y1 - y0 < 5 or x1 - x0 < 5:
            logger.warning("Rectangle %d too small after inset (%dx%d) — skipping",
                           rect["id"], x1 - x0, y1 - y0)
            continue

        interior = arr[y0:y1, x0:x1]

        # Determine alpha: near-white background OR residual boundary color
        ir = interior[..., 0].astype(int)
        ig = interior[..., 1].astype(int)
        ib = interior[..., 2].astype(int)
        is_white = (ir >= background_threshold) & (ig >= background_threshold) & (ib >= background_threshold)
        is_boundary = (
            np.abs(ir - boundary_color[0])
            + np.abs(ig - boundary_color[1])
            + np.abs(ib - boundary_color[2])
        ) < boundary_tolerance
        is_transparent = is_white | is_boundary
        alpha = np.where(is_transparent, 0, 255).astype(np.uint8)
        rgba = np.dstack([interior, alpha])

        # Crop to opaque bbox
        opaque_ys, opaque_xs = np.where(alpha > 0)
        if len(opaque_ys) == 0:
            logger.warning("Rectangle %d had no opaque content — skipping", rect["id"])
            continue
        cy0, cy1 = int(opaque_ys.min()), int(opaque_ys.max() + 1)
        cx0, cx1 = int(opaque_xs.min()), int(opaque_xs.max() + 1)
        cropped = rgba[cy0:cy1, cx0:cx1]

        # Transparent padding
        if padding > 0:
            ch, cw = cropped.shape[:2]
            padded = np.zeros((ch + 2 * padding, cw + 2 * padding, 4), dtype=np.uint8)
            padded[padding:padding + ch, padding:padding + cw] = cropped
        else:
            padded = cropped

        part_path = out_dir / f"part_{i + 1:02d}.png"
        Image.fromarray(padded, "RGBA").save(part_path)

        parts.append({
            "path": str(part_path),
            "index": i + 1,
            "rectangle_bbox": [rect["x_min"], rect["y_min"], rect["x_max"], rect["y_max"]],
            "content_size": [int(padded.shape[1]), int(padded.shape[0])],
        })

    # 6. Labeled debug overview — original sheet with each detected rect numbered
    debug_img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(24, H // 50))
    except Exception:
        font = ImageFont.load_default()
    for i, rect in enumerate(rectangles):
        label_text = f"#{i + 1}"
        # yellow tag in top-left of the rectangle
        tag_w, tag_h = max(50, H // 30), max(28, H // 50)
        tx0 = rect["x_min"] + 3
        ty0 = rect["y_min"] + 3
        draw.rectangle([tx0, ty0, tx0 + tag_w, ty0 + tag_h], fill="yellow", outline="black")
        draw.text((tx0 + 4, ty0 + 2), label_text, fill="black", font=font)
    debug_path = out_dir / "_debug_labeled.png"
    debug_img.save(debug_path)

    logger.info(
        "extract_body_parts: %d rectangles extracted from %s → %s",
        len(parts), input_path, out_dir,
    )

    return {
        "count": len(parts),
        "parts": parts,
        "debug_image": str(debug_path),
        "output_dir": str(out_dir),
    }

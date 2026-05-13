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

    # 1. Boundary-color mask
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    color_dist = (
        np.abs(r - boundary_color[0])
        + np.abs(g - boundary_color[1])
        + np.abs(b - boundary_color[2])
    )
    boundary_mask = color_dist < boundary_tolerance

    # 2. Close anti-aliasing gaps so each rectangle outline is one connected blob
    boundary_mask = binary_closing(boundary_mask, iterations=2)

    if not boundary_mask.any():
        raise RuntimeError(
            f"No boundary pixels detected with color {boundary_color} "
            f"(tolerance {boundary_tolerance}). Are the rectangles drawn in "
            f"the expected color? Try a different boundary_color or a "
            f"higher boundary_tolerance."
        )

    # 3. Label connected components
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
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

"""Contact sheet generator: tile sprite frames horizontally into one PNG."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

GAP_PX = 8
INDICATOR_H = 10   # height of the pass/fail bar at the bottom of each cell
_COLOR_PASS   = (50, 200, 80, 255)    # green
_COLOR_FAIL   = (220, 50, 50, 255)    # red
_COLOR_UNKNOWN = (120, 120, 120, 255) # grey (no validation data)


def make_contact_sheet(
    frame_paths: list[str],
    output_path: str,
    validation: list[dict] | None = None,
) -> str:
    """Tile frames horizontally with a small transparent gap between each.

    If `validation` is supplied (one dict per frame, each with a ``passed``
    bool key), a coloured bar is drawn at the bottom of every cell:
    green = passed, red = failed, grey = unknown.
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty")

    frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    cell_w = max(f.width for f in frames)
    cell_h = max(f.height for f in frames)

    bar_h = INDICATOR_H if validation else 0
    n = len(frames)
    total_w = n * cell_w + (n - 1) * GAP_PX
    sheet = Image.new("RGBA", (total_w, cell_h + bar_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    x = 0
    for i, f in enumerate(frames):
        offset_x = x + (cell_w - f.width) // 2
        offset_y = (cell_h - f.height) // 2
        sheet.paste(f, (offset_x, offset_y), f)

        if validation:
            v = validation[i] if i < len(validation) else None
            if v is None:
                color = _COLOR_UNKNOWN
            elif v.get("passed"):
                color = _COLOR_PASS
            else:
                color = _COLOR_FAIL
            draw.rectangle(
                [x, cell_h, x + cell_w - 1, cell_h + bar_h - 1],
                fill=color,
            )

        x += cell_w + GAP_PX

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path

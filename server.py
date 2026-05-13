"""MCP server exposing a 2D game asset generator backed by OpenRouter."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import re

import httpx
import numpy as np
from PIL import Image
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

import body_part_extractor
import contact_sheet
import sprite_processor

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.1-flash-image-preview"
OUTPUT_DIR = Path(os.environ.get("GAME_ASSETS_DIR", "~/game-assets")).expanduser()

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

SYSTEM_PROMPT = """\
You are a 2D game sprite reproducer — your job is pixel-accurate character copying, NOT creative design.

When a reference image is provided:
- Your ONLY task is to reproduce the EXACT character shown in the reference image. Every detail must match: face shape, eye shape, eye color, hair type and length, clothing, colors, proportions, outline weight, and art style.
- Do NOT redesign, simplify, or "improve" any part of the character. If the reference shows button eyes (large round circles), reproduce them as large round circles — NOT as glasses, goggles, or spectacles.
- Do NOT add elements that are not in the reference (no glasses, no extra accessories, no different hairstyle).
- Apply ONLY the pose/action changes explicitly stated in the user prompt. Change NOTHING else about the character's appearance.
- Do NOT produce character turnaround sheets, multi-view layouts, or model sheets unless explicitly asked.
- Do NOT add text, labels, annotations, borders, or reference grids.

When no reference image is provided:
- Follow the user prompt exactly to generate the requested asset.
"""

mcp = FastMCP("video-to-sprite-server")


def _postprocess_image(image_bytes: bytes) -> bytes:
    """Remove the white background and crop to content with transparent padding.

    Steps:
    1. Convert to RGBA.
    2. Flood-fill from all four corners, replacing white/near-white pixels
       (tolerance 30) with transparency.
    3. Crop to the tight bounding box of non-transparent content.
    4. Add 8 % transparent padding on each side.
    """
    TOLERANCE = 30
    PADDING_RATIO = 0.08

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    def is_near_white(r: int, g: int, b: int) -> bool:
        return r >= 255 - TOLERANCE and g >= 255 - TOLERANCE and b >= 255 - TOLERANCE

    # BFS flood-fill from each corner
    visited = [[False] * height for _ in range(width)]
    queue: list[tuple[int, int]] = []
    for cx, cy in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
        r, g, b, a = pixels[cx, cy]
        if is_near_white(r, g, b):
            queue.append((cx, cy))
            visited[cx][cy] = True

    while queue:
        x, y = queue.pop()
        r, g, b, a = pixels[x, y]
        if not is_near_white(r, g, b):
            continue
        pixels[x, y] = (r, g, b, 0)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                visited[nx][ny] = True
                nr, ng, nb, na = pixels[nx, ny]
                if is_near_white(nr, ng, nb):
                    queue.append((nx, ny))

    bbox = img.getbbox()
    if bbox is None:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    img = img.crop(bbox)
    cw, ch = img.size

    pad_w = int(cw * PADDING_RATIO)
    pad_h = int(ch * PADDING_RATIO)
    padded = Image.new("RGBA", (cw + 2 * pad_w, ch + 2 * pad_h), (0, 0, 0, 0))
    padded.paste(img, (pad_w, pad_h), img)

    buf = io.BytesIO()
    padded.save(buf, format="PNG")
    return buf.getvalue()


def _validate_reference_image(path_str: str) -> Path:
    """Validate a user-provided reference image path.

    Resolves ~ and symlinks, then ensures the target is an existing regular
    file with a recognized image extension. Resolving before checking the
    extension prevents tricks like trailing slashes or relative escapes from
    bypassing the suffix check.
    """
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"Reference image not found or not a regular file: {path}"
        )
    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Reference image must have one of these extensions: "
            f"{sorted(ALLOWED_IMAGE_EXTENSIONS)} (got {path.suffix!r})"
        )
    return path


@mcp.tool()
async def generate_2d_asset(
    prompt: str, reference_image_path: str | None = None
) -> list[ImageContent | TextContent]:
    """Generate a 2D game asset image from a text prompt.

    Sends the prompt to OpenRouter using the OpenAI chat completions format.
    The resulting PNG is written to the directory configured by the
    GAME_ASSETS_DIR env var (default ~/game-assets/) with a timestamped
    filename. Returns the image as an MCP ImageContent block (so Claude
    can see and analyze it) plus a TextContent block with the saved path.

    If `reference_image_path` is provided, the file is read from disk,
    base64-encoded, and sent alongside the prompt as a visual starting
    point for the model — useful for "modify this image" or "generate a
    variation of this" workflows. Supports PNG and JPEG; the mime type
    is inferred from the file extension. The path may use ~ for the
    home directory.
    """
    if reference_image_path:
        path = _validate_reference_image(reference_image_path)
        ref_bytes = path.read_bytes()
        ref_b64 = base64.b64encode(ref_bytes).decode()
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{ref_b64}"},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    else:
        user_content = prompt

    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "modalities": ["image", "text"],
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "image_config": {"aspect_ratio": "1:1"},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()

        logger.debug("Full OpenRouter response: %s", json.dumps(data, indent=2))
        logger.debug("Model returned by API: %s", data.get("model", "<not present>"))

        image_bytes = await _extract_image_bytes(data, client)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"asset_{timestamp}.png"

    # Save raw bytes first, then post-process in-place
    output_path.write_bytes(image_bytes)
    processed_bytes = _postprocess_image(image_bytes)
    output_path.write_bytes(processed_bytes)

    return [
        ImageContent(
            type="image",
            data=base64.b64encode(processed_bytes).decode(),
            mimeType="image/png",
        ),
        TextContent(
            type="text",
            text=f"Image saved to {output_path}",
        ),
    ]


async def _extract_image_bytes(data: dict, client: httpx.AsyncClient) -> bytes:
    """Extract image bytes from an OpenAI-format chat completion response.

    Handles:
    - choices[].message.content as a list of parts (image_url or image or inline_data types)
    - choices[].message.content as a plain string (data URL or https URL)
    - top-level data[].b64_json / data[].url  (image generation endpoint compat)
    """
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("API returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content")

    logger.debug("choices[0].message.content type: %s", type(content).__name__)
    logger.debug(
        "choices[0].message.content: %s",
        json.dumps(content, indent=2) if not isinstance(content, bytes) else repr(content),
    )

    # OpenRouter non-standard field: image data lives in message.images, not content
    images_field = message.get("images", [])
    logger.debug("choices[0].message.images: %s", json.dumps(images_field, indent=2))
    if images_field:
        for img in images_field:
            img_url_val = img.get("image_url", {})
            url = img_url_val.get("url") if isinstance(img_url_val, dict) else img_url_val
            if url:
                return await _fetch_or_decode(url, client)

    # Content is a list of multimodal parts
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")

            if ptype == "image_url":
                # image_url value may be a dict {"url": "..."} or a bare string
                img_url_val = part.get("image_url")
                if isinstance(img_url_val, dict):
                    return await _fetch_or_decode(img_url_val["url"], client)
                if isinstance(img_url_val, str):
                    return await _fetch_or_decode(img_url_val, client)

            if ptype == "image":
                # Anthropic/Gemini style: {"type":"image","source":{"type":"base64","data":"..."}}
                source = part.get("source", {})
                if source.get("type") == "base64":
                    return base64.b64decode(source["data"])
                if source.get("type") == "url":
                    return await _fetch_or_decode(source["url"], client)

            if ptype == "inline_data":
                # Gemini native format: {"type":"inline_data","inline_data":{"mime_type":"image/png","data":"<b64>"}}
                inline = part.get("inline_data", {})
                if inline.get("data"):
                    return base64.b64decode(inline["data"])

    # Content is a bare string: data URL or https URL
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("data:") or stripped.startswith("http"):
            return await _fetch_or_decode(stripped, client)

    # Fallback: OpenAI /images/generations compat (data[].b64_json / data[].url)
    images = data.get("data", [])
    if images:
        img = images[0]
        if "b64_json" in img:
            return base64.b64decode(img["b64_json"])
        if "url" in img:
            resp = await client.get(img["url"])
            resp.raise_for_status()
            return resp.content

    raise RuntimeError(
        f"Could not extract image from API response. "
        f"Keys: {list(data.keys())} | "
        f"content type: {type(content).__name__} | "
        f"content value: {json.dumps(content)[:500] if not isinstance(content, bytes) else repr(content[:200])}"
    )


async def _fetch_or_decode(url: str, client: httpx.AsyncClient) -> bytes:
    """Decode a base64 data URL, or fetch a remote URL."""
    if url.startswith("data:"):
        # data:<mime>;base64,<data>
        _, encoded = url.split(",", 1)
        return base64.b64decode(encoded)
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.content


# ── Frame validation ─────────────────────────────────────────────────────────

def _get_char_metrics(image_path: str) -> dict | None:
    """Return character bounding-box metrics for a processed frame.

    Returns None if the frame contains no visible character pixels.
    All measurements are in canvas pixels (before any in-engine scaling).
    Uses pure Pillow — no numpy required.
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = img.split()[-1]
    # Threshold: only count pixels with alpha > 20
    alpha_thresh = alpha.point(lambda v: 255 if v > 20 else 0)
    bbox = alpha_thresh.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    w, h = img.size
    return {
        "top": top, "bottom": bottom, "left": left, "right": right,
        "height": bottom - top,
        "width":  right - left,
        "canvas_h": h, "canvas_w": w,
        "head_frac": top / h,       # head position as fraction of canvas height
        "foot_frac": bottom / h,    # foot position as fraction of canvas height
    }


def _validate_one_frame(
    frame_path: str,
    ref: dict,
    size_tol: float = 0.18,
    head_tol: float = 0.08,
) -> dict:
    """Validate a single processed frame against reference metrics.

    Checks:
    - Character height within ±size_tol of reference height
    - Character width within ±size_tol*1.5 of reference (poses vary more laterally)
    - Head vertical position within ±head_tol of reference (detects shrunken / floated chars)

    Returns a dict with ``passed`` bool, ``issues`` list, and raw measurements.
    """
    m = _get_char_metrics(frame_path)
    if m is None:
        return {"passed": False, "issues": ["no character detected"], "metrics": None}

    issues: list[str] = []

    h_ratio = m["height"] / max(ref["height"], 1)
    if abs(h_ratio - 1.0) > size_tol:
        issues.append(
            f"height {m['height']}px vs ref {ref['height']}px "
            f"({h_ratio:.0%} — expected {100*(1-size_tol):.0f}–{100*(1+size_tol):.0f}%)"
        )

    w_ratio = m["width"] / max(ref["width"], 1)
    if abs(w_ratio - 1.0) > size_tol * 1.5:
        issues.append(
            f"width {m['width']}px vs ref {ref['width']}px ({w_ratio:.0%})"
        )

    head_diff = abs(m["head_frac"] - ref["head_frac"])
    if head_diff > head_tol:
        issues.append(
            f"head at {m['head_frac']:.0%} from top vs ref {ref['head_frac']:.0%} "
            f"(diff {head_diff:.0%})"
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "metrics": m,
        "height_ratio": h_ratio,
        "width_ratio": w_ratio,
    }


def _validate_frames(
    processed_paths: list[str],
    reference_image_path: str,
    size_tol: float = 0.18,
    head_tol: float = 0.08,
) -> dict:
    """Validate all generated frames against the reference image.

    Returns a dict with per-frame results and a human-readable summary.
    This is always called after sprite_processor runs so that every batch
    of generated art is checked for size and position consistency before
    being shown or written to assets.
    """
    ref = _get_char_metrics(reference_image_path)
    if ref is None:
        logger.warning("validate_frames: could not extract metrics from reference %s", reference_image_path)
        return {
            "reference_ok": False,
            "frames": [{"passed": True, "issues": ["reference unreadable"]} for _ in processed_paths],
            "all_passed": True,
            "failed_indices": [],
            "summary": "validation skipped — reference unreadable",
        }

    frame_results = [
        _validate_one_frame(p, ref, size_tol, head_tol)
        for p in processed_paths
    ]
    failed = [i for i, r in enumerate(frame_results) if not r["passed"]]
    n = len(processed_paths)
    summary_lines = [f"{n - len(failed)}/{n} frames passed validation"]
    for i in failed:
        for issue in frame_results[i]["issues"]:
            summary_lines.append(f"  frame_{i}: {issue}")

    logger.info("Frame validation: %s", summary_lines[0])
    for line in summary_lines[1:]:
        logger.warning("Frame validation%s", line)

    return {
        "reference_ok": True,
        "reference_metrics": ref,
        "frames": frame_results,
        "all_passed": len(failed) == 0,
        "failed_indices": failed,
        "summary": "\n".join(summary_lines),
    }


_DEFAULT_CONFIG = {
    "godot_executable": "/Applications/Godot.app/Contents/MacOS/Godot",
    "game_project_path": "",
    "removebg_api_key": "",
    "defaults": {
        "canvas_width": 100,
        "canvas_height": 250,
        "char_width": 90,
        "foot_anchor_y": 238,
        "animation_fps": 8,
    },
    "characters": {},
}


def _load_config() -> dict:
    """Load config.json. Discovery: $GAME_ASSET_SERVER_CONFIG → ./config.json → defaults."""
    candidates: list[Path] = []
    env_path = os.environ.get("GAME_ASSET_SERVER_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "config.json")
    candidates.append(Path(__file__).parent / "config.json")

    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception as exc:
                logger.warning("Failed to read config at %s: %s", path, exc)
    return dict(_DEFAULT_CONFIG)


def _resolve_dimensions(
    config: dict,
    character: str,
    overrides: dict,
) -> dict:
    """Three-layer precedence: tool-call overrides > character config > defaults."""
    defaults = {**_DEFAULT_CONFIG["defaults"], **config.get("defaults", {})}
    char_cfg = config.get("characters", {}).get(character, {}) or {}
    eff = dict(defaults)
    for k, v in char_cfg.items():
        if v is not None:
            eff[k] = v
    for k, v in overrides.items():
        if v is not None:
            eff[k] = v
    return eff


async def _generate_one_image(
    prompt: str,
    client: httpx.AsyncClient,
    reference_image_path: str | Path | None = None,
) -> bytes:
    """Generate a single 2D asset image and return raw PNG bytes.

    If `reference_image_path` is provided, the image is base64-encoded and
    sent as a multimodal `image_url` part alongside the text prompt.
    """
    if reference_image_path:
        ref_path = _validate_reference_image(str(reference_image_path))
        ref_b64 = base64.b64encode(ref_path.read_bytes()).decode()
        mime = mimetypes.guess_type(str(ref_path))[0] or "image/png"
        user_content: object = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{ref_b64}"},
            },
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "modalities": ["image", "text"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
    }
    response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API returned {response.status_code}: {response.text[:500]}"
        )
    return await _extract_image_bytes(response.json(), client)


def _build_sheet_prompt(
    animation: str,
    frame_count: int,
    frame_size: int,
    character_description: str | None = None,
) -> str:
    """Build a sprite-sheet generation prompt.

    `animation` is the caller-supplied animation-specific frame breakdown
    (e.g., per-frame pose descriptions for a run cycle). It is embedded
    verbatim in the middle of the prompt.

    `character_description` is an optional explicit list of key character
    features for the model to preserve exactly — useful for overriding model
    misinterpretations of art-style elements (e.g. "button eyes, NOT glasses").
    """
    sheet_width = frame_size * frame_count
    sheet_height = frame_size

    char_desc_section = ""
    if character_description:
        char_desc_section = (
            f"\nCRITICAL — preserve these character features EXACTLY as described "
            f"(do not reinterpret them):\n{character_description}\n"
        )

    return (
        f"Using the provided reference image as the base character, generate a sprite "
        f"sheet of exactly {frame_count} frames arranged horizontally left-to-right on "
        f"a single canvas of {sheet_width}x{sheet_height} pixels (each frame is exactly "
        f"{frame_size}x{frame_size} pixels, evenly spaced with no gaps, no borders, no "
        f"labels).\n\n"
        f"This is a reproduction task, NOT a design task. Copy the character from the "
        f"reference image pixel-perfectly — same face, same eyes, same hair, same "
        f"clothing, same colors, same proportions, same outline weight, same art style. "
        f"Do NOT redesign, simplify, or add anything that is not in the reference.\n"
        f"{char_desc_section}\n"
        f"{animation}\n\n"
        f"Consistency rule: The character's appearance (head, face, eyes, hair, "
        f"clothing, colors, outline) must be identical across all {frame_count} frames. "
        f"Only the pose changes described per-frame above may differ.\n\n"
        f"Output: the sprite sheet image only. Transparent background. No frame "
        f"numbers. No labels. No borders between frames."
    )


def _slice_sheet(sheet_path: str, frame_count: int, output_dir: str) -> list[str]:
    """Slice a horizontal sprite sheet into `frame_count` frames.

    Uses content-aware slicing: finds vertical "gap" columns (columns where
    pixel content is sparse) to locate character boundaries, then selects the
    `frame_count` best-spaced cut points. Falls back to equal-width slicing if
    gap detection fails or finds too few gaps.

    Frames are saved as `raw_frame_N.png` in `output_dir`.
    """
    img = Image.open(sheet_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]  # shape: (height, width)

    # Column content score: fraction of pixels with meaningful alpha
    col_fill = (alpha > 20).sum(axis=0) / alpha.shape[0]

    # Smooth slightly to avoid noise
    kernel_size = max(3, img.width // 200)
    kernel = np.ones(kernel_size) / kernel_size
    col_smooth = np.convolve(col_fill, kernel, mode="same")

    # Find gap columns: below a threshold, not near the image edges
    gap_threshold = col_smooth.max() * 0.08
    is_gap = col_smooth < gap_threshold

    # Find contiguous gap regions and pick their centers
    gap_centers: list[int] = []
    in_gap = False
    gap_start = 0
    margin = img.width // (frame_count * 4)  # ignore near-edge columns
    for x in range(margin, img.width - margin):
        if is_gap[x] and not in_gap:
            in_gap = True
            gap_start = x
        elif not is_gap[x] and in_gap:
            in_gap = False
            gap_centers.append((gap_start + x) // 2)

    # We need exactly (frame_count - 1) cut points between frames
    cut_points: list[int] = []
    if len(gap_centers) >= frame_count - 1:
        # If more gaps than needed, pick the frame_count-1 most evenly spaced ones
        # Strategy: greedily pick gaps that divide the sheet most evenly
        ideal_spacing = img.width / frame_count
        ideal_cuts = [int(ideal_spacing * (i + 1)) for i in range(frame_count - 1)]
        for ideal in ideal_cuts:
            closest = min(gap_centers, key=lambda g: abs(g - ideal))
            cut_points.append(closest)
        cut_points = sorted(set(cut_points))

    # Fall back to equal-width if we couldn't find enough gaps
    if len(cut_points) < frame_count - 1:
        logger.info(
            "_slice_sheet: content-aware detection found %d gaps (need %d), "
            "falling back to equal-width slicing",
            len(gap_centers),
            frame_count - 1,
        )
        frame_w = img.width // frame_count
        cut_points = [frame_w * (i + 1) for i in range(frame_count - 1)]

    boundaries = [0] + cut_points + [img.width]
    paths: list[str] = []
    for i in range(frame_count):
        x0, x1 = boundaries[i], boundaries[i + 1]
        frame = img.crop((x0, 0, x1, img.height))
        out = os.path.join(output_dir, f"raw_frame_{i}.png")
        frame.save(out)
        paths.append(out)
    return paths


_GIF_BG_COLOR = (30, 30, 30, 255)  # dark background — shows sprite colors clearly


def _stitch_gif(frame_paths: list[str], output_path: str, fps: int) -> str:
    """Combine frames into a looping animated GIF using Pillow.

    GIF supports only 1-bit transparency, so raw RGBA frames produce
    palette-colour artifacts for transparent regions. We composite each frame
    onto a solid dark background first so the character is clean against a
    consistent colour instead of a random palette entry.
    """
    composited: list[Image.Image] = []
    for p in frame_paths:
        frame = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", frame.size, _GIF_BG_COLOR)
        bg.paste(frame, mask=frame.split()[3])
        composited.append(bg.convert("RGB"))

    duration_ms = max(1, int(round(1000 / fps)))
    composited[0].save(
        output_path,
        save_all=True,
        append_images=composited[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def _try_godot_preview(
    processed_frames: list[str],
    output_gif_path: Path,
    config: dict,
    fps: int,
) -> str | None:
    """Run Godot headless to render a preview, then stitch its frames into a GIF.

    Returns the GIF path on success, None if Godot or the preview scene is
    unavailable (logs a warning in that case — never raises).
    """
    sidecar_path = Path("/tmp/sprite_preview_config.json")
    sidecar_path.write_text(json.dumps({
        "frames": processed_frames,
        "fps": fps,
        "loop": True,
        "duration_seconds": 3,
    }))

    godot_exec = config.get("godot_executable", "")
    game_path = config.get("game_project_path", "")
    if not godot_exec or not Path(godot_exec).exists():
        logger.warning("Godot executable not found at %s — skipping GIF preview", godot_exec)
        return None
    if not game_path:
        logger.warning("game_project_path not set in config — skipping GIF preview")
        return None

    scene_path = Path(game_path) / "tests/integration/sprite_preview.tscn"
    if not scene_path.exists():
        logger.warning(
            "sprite_preview.tscn not found at %s — skipping GIF preview", scene_path
        )
        return None

    render_dir = Path("/tmp") / f"sprite_preview_render_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    render_dir.mkdir(parents=True, exist_ok=True)
    movie_target = render_dir / "frame.png"

    cmd = [
        godot_exec,
        "--path", game_path,
        "--write-movie", str(movie_target),
        "--fixed-fps", str(fps),
        "res://tests/integration/sprite_preview.tscn",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("Godot preview run failed (%s) — skipping GIF", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "Godot exited %s — skipping GIF. stderr: %s",
            result.returncode, result.stderr[:500],
        )
        return None

    rendered = sorted(render_dir.glob("*.png"))
    if not rendered:
        logger.warning("Godot produced no frames in %s — skipping GIF", render_dir)
        return None

    return _stitch_gif([str(p) for p in rendered], str(output_gif_path), fps)


@mcp.tool()
async def generate_game_sprite(
    prompt: str,
    animation: str = "idle",
    character: str = "player",
    frame_count: int = 8,
    frame_size: int = 128,
    preview: bool = True,
    write_to_assets: bool = False,
    reference_image: str | None = None,
    character_description: str | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    char_width: int | None = None,
    foot_anchor_y: int | None = None,
) -> dict:
    """Generate an animation strip of game-ready sprite frames.

    `prompt` is the animation-specific frame breakdown (e.g., per-frame pose
    descriptions). It is embedded inside a sprite-sheet prompt that asks the
    model to render all frames on a single horizontal canvas, using
    `reference_image` as the visual ground truth for the character.

    The returned sheet is sliced into `frame_count` equal-width raw frames,
    each of which is post-processed via `sprite_processor.process_image`
    (background removal + canvas placement at the configured anchor). A
    horizontal contact sheet is built from the processed frames.

    If `reference_image` is None, defaults to
    `{game_project_path}/assets/characters/{character}/idle/frame_0.png`.

    `character_description` is an optional plain-text description of the
    character's key visual features that the model must preserve exactly (e.g.
    "button eyes — large round circles, NOT glasses"). When provided it is
    injected into the generation prompt as a CRITICAL preservation list,
    helping the model avoid misinterpreting art-style elements in the reference
    image. Claude should always populate this when calling the tool for a
    known character.

    If `preview=True`, generates an animated GIF from the processed frames.
    First attempts a Godot-rendered preview via
    `tests/integration/sprite_preview.tscn`; if Godot is unavailable or
    fails, falls back to a Pillow-stitched GIF directly from the processed
    frames. The GIF is always produced when `preview=True` and at least one
    frame was processed successfully.

    If `write_to_assets=True`, copies the processed frames to
    `{game_project_path}/assets/characters/{character}/{animation}/frame_N.png`.

    Dimension precedence (highest first): tool-call args, character config,
    defaults from config.json.
    """
    config = _load_config()
    overrides = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "char_width": char_width,
        "foot_anchor_y": foot_anchor_y,
    }
    eff = _resolve_dimensions(config, character, overrides)
    fps = int(eff.get("animation_fps", 8))

    if reference_image is None:
        game_path = config.get("game_project_path", "")
        if not game_path:
            raise ValueError(
                "reference_image not provided and game_project_path is empty in "
                "config — cannot resolve default reference image."
            )
        reference_image = str(
            Path(game_path) / "assets" / "characters" / character / "idle" / "frame_0.png"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = OUTPUT_DIR / "sprites" / character / f"{animation}_{timestamp}"
    raw_dir = work_dir / "raw"
    processed_dir = work_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # If the caller has already supplied a complete sprite-sheet prompt (detected
    # by the canonical "Using the provided reference image as frame 1" opener),
    # pass it through directly rather than wrapping it with _build_sheet_prompt.
    # Wrapping a complete template creates conflicting double-instructions that
    # confuse the model.  We still inject character_description if provided.
    _TEMPLATE_MARKER = "Using the provided reference image as frame 1"
    if prompt.strip().startswith(_TEMPLATE_MARKER):
        sheet_prompt = prompt
    else:
        sheet_prompt = _build_sheet_prompt(prompt, frame_count, frame_size)

    async with httpx.AsyncClient(timeout=120.0) as client:
        sheet_bytes = await _generate_one_image(
            sheet_prompt, client, reference_image_path=reference_image
        )

    sheet_path = work_dir / "sheet.png"
    sheet_path.write_bytes(sheet_bytes)

    # Remove background from the full sheet in one API call, then slice.
    # This gives far cleaner results than per-frame removal — remove.bg sees
    # the whole composition and won't clip edges or eat light-coloured pixels.
    sheet_bg_path = work_dir / "sheet_bg_removed.png"
    await asyncio.to_thread(
        sprite_processor.strip_background,
        str(sheet_path),
        str(sheet_bg_path),
        config.get("removebg_api_key", "") or None,
    )

    raw_paths = await asyncio.to_thread(
        _slice_sheet, str(sheet_bg_path), frame_count, str(raw_dir)
    )

    # Batch canvas-normalize: shared scale factor across all frames keeps
    # character size consistent across poses (no size-pop between frames).
    processed_paths = [str(processed_dir / f"frame_{i}.png") for i in range(len(raw_paths))]
    await asyncio.to_thread(
        sprite_processor.canvas_normalize_batch,
        list(raw_paths),
        processed_paths,
        int(eff["canvas_width"]),
        int(eff["canvas_height"]),
        int(eff["char_width"]),
        int(eff["foot_anchor_y"]),
    )

    # ── Validate frames against reference ────────────────────────────────────
    validation = await asyncio.to_thread(
        _validate_frames, processed_paths, reference_image
    )
    if not validation["all_passed"]:
        logger.warning(
            "generate_game_sprite: %d frame(s) failed validation — "
            "consider regenerating. Details:\n%s",
            len(validation["failed_indices"]),
            validation["summary"],
        )

    contact_sheet_path = work_dir / "contact_sheet.png"
    contact_sheet.make_contact_sheet(
        processed_paths,
        str(contact_sheet_path),
        validation=validation["frames"],
    )

    gif_path: str | None = None
    if preview and processed_paths:
        gif_target = work_dir / "preview.gif"
        # Pillow stitch first — renders frames at their native 100x250 canvas size,
        # so the character is clearly visible. Godot preview was producing a
        # 1280x720 viewport with the sprite as a tiny dot, which is useless for
        # frame review.
        gif_path = None
        try:
            gif_path = await asyncio.to_thread(
                _stitch_gif, processed_paths, str(gif_target), fps
            )
        except Exception as exc:
            logger.warning("Pillow GIF stitch failed (%s) — no preview GIF", exc)

    written = False
    game_path = config.get("game_project_path", "")

    # Always copy contact sheet + GIF into the game project's _preview_tmp/<animation>/
    # so Claude can read them for review regardless of write_to_assets setting.
    if game_path:
        preview_dir = Path(game_path) / "_preview_tmp" / animation
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Step 1: raw model output
        shutil.copy2(str(sheet_path), preview_dir / "step1_raw_sheet.png")
        # Step 2: after bg removal on full sheet
        if sheet_bg_path.exists():
            shutil.copy2(str(sheet_bg_path), preview_dir / "step2_bg_removed_sheet.png")
        # Step 3: individual slices (before canvas normalization)
        for i, raw_p in enumerate(raw_paths):
            shutil.copy2(raw_p, preview_dir / f"step3_raw_frame_{i}.png")
        # Step 4: canvas-normalized frames
        for i, src in enumerate(processed_paths):
            shutil.copy2(src, preview_dir / f"step4_frame_{i}.png")
        # Contact sheet + GIF
        shutil.copy2(str(contact_sheet_path), preview_dir / "contact_sheet.png")
        shutil.copy2(str(sheet_path), preview_dir / "sheet.png")
        for i, src in enumerate(processed_paths):
            shutil.copy2(src, preview_dir / f"frame_{i}.png")
        if gif_path:
            shutil.copy2(gif_path, preview_dir / "preview.gif")

    if write_to_assets:
        if not game_path:
            logger.warning("write_to_assets=True but game_project_path is empty")
        else:
            assets_dir = Path(game_path) / "assets" / "characters" / character / animation
            assets_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(processed_paths):
                shutil.copy2(src, assets_dir / f"frame_{i}.png")
            written = True

    return {
        "frames": processed_paths,
        "sheet": str(sheet_path),
        "contact_sheet": str(contact_sheet_path),
        "gif": gif_path,
        "written_to_assets": written,
        "reference_image": reference_image,
        "effective_config": eff,
        "validation": validation,
        "validation_summary": validation["summary"],
        "validation_passed": validation["all_passed"],
    }


# ── Video-to-sprite pipeline (Sprite-Pipeline approach) ──────────────────────

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}

# Runway URLs we know how to resolve to a downloadable mp4
_RUNWAY_UUID_RE = re.compile(
    r"runwayml\.com/(?:creation|share|shared_assets)/"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/?",
    re.IGNORECASE,
)


def _resolve_video_to_local_path(video_path_or_url: str) -> Path:
    """Accept either a local mp4 path or a remote URL; return a local Path.

    Three cases:
      1. Local path — return as-is (must exist).
      2. Runway URL (app.runwayml.com/{creation,share,shared_assets}/<uuid>):
         hit the public shared-assets API, pull the signed mp4 URL, download
         to a content-addressed cache so repeat runs of the same URL skip
         the network round-trip.
      3. Generic http(s) URL ending in a known video extension: just
         download it.

    The cache lives at `OUTPUT_DIR/runway-cache/<uuid-or-hash>/<filename>` —
    safe to delete by hand to force a re-fetch.
    """
    s = video_path_or_url.strip()
    is_url = s.startswith(("http://", "https://"))
    if not is_url:
        p = Path(s).expanduser().resolve()
        if not p.is_file():
            raise ValueError(f"Video file not found: {p}")
        return p

    cache_root = OUTPUT_DIR / "runway-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Case 2: Runway-managed URL
    m = _RUNWAY_UUID_RE.search(s)
    if m:
        uuid = m.group("uuid").lower()
        cache_dir = cache_root / uuid
        # Skip network if cached
        if cache_dir.is_dir():
            existing = list(cache_dir.glob("*.mp4")) + list(cache_dir.glob("*.mov")) + list(cache_dir.glob("*.webm"))
            if existing:
                logger.info("runway-cache hit: reusing %s", existing[0])
                return existing[0]
        cache_dir.mkdir(parents=True, exist_ok=True)

        meta_url = f"https://api.runwayml.com/v1/shared_assets/{uuid}"
        try:
            resp = httpx.get(meta_url, timeout=30.0, follow_redirects=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reach Runway shared-assets API for {uuid}: {exc}"
            ) from exc
        if resp.status_code == 404:
            raise RuntimeError(
                f"Runway returned 404 for asset {uuid}. The clip's sharing toggle "
                "is probably off — open the clip in Runway, click Share, and turn "
                "on 'Anyone with the link can view', then retry."
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Runway API returned {resp.status_code} for {meta_url}: "
                f"{resp.text[:200]}"
            )
        meta = resp.json()
        artifacts = meta.get("artifacts") or []
        if not artifacts:
            raise RuntimeError(
                f"Runway asset {uuid} exists but has no artifacts — "
                "is generation still running, or did it fail?"
            )
        art = artifacts[0]
        media_url = art.get("url")
        filename = art.get("filename") or f"{uuid}.mp4"
        if not media_url:
            raise RuntimeError(
                f"Runway artifact for {uuid} has no `url` field. Raw: {art}"
            )

        # Sanitize filename — Runway's titles can contain commas, slashes, etc.
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename).strip()
        if not safe_name.lower().endswith(tuple(_VIDEO_EXTENSIONS)):
            safe_name = f"{safe_name}.mp4"
        local = cache_dir / safe_name

        with httpx.stream("GET", media_url, timeout=120.0, follow_redirects=True) as r:
            if r.status_code != 200:
                raise RuntimeError(
                    f"Download from Runway CDN returned {r.status_code} (URL is "
                    f"signed and short-lived; try again)."
                )
            with local.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        logger.info("runway-cache: downloaded %s (%d bytes)", local, local.stat().st_size)
        return local

    # Case 3: Generic video URL — download as-is to a hash-keyed cache subdir
    import hashlib
    url_hash = hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    cache_dir = cache_root / f"url-{url_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Pull a sensible filename out of the URL path
    from urllib.parse import urlparse
    parsed = urlparse(s)
    filename = Path(parsed.path).name or "video.mp4"
    if not filename.lower().endswith(tuple(_VIDEO_EXTENSIONS)):
        raise ValueError(
            f"URL does not look like a direct video file (no .mp4/.mov/.webm/.mkv "
            f"extension on path): {s}. If this is a hosted-page URL rather than "
            "a CDN file URL, paste the direct video URL instead."
        )
    local = cache_dir / filename
    if local.is_file() and local.stat().st_size > 0:
        logger.info("url cache hit: reusing %s", local)
        return local
    with httpx.stream("GET", s, timeout=120.0, follow_redirects=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Download returned {r.status_code} for {s}")
        with local.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    return local

# Chroma-key thresholds. "Keyness" is min(high_channels) - max(low_channels)
# — how much the key-dominant channel(s) overpower the suppressed ones. For a
# pure #00FF00 (green) key it equals `g - max(r, b)`; for a pure #FF00FF
# (magenta) key it equals `min(r, b) - g`; for any chroma color it captures
# how key-like the pixel is.
#
# Three thresholds, decoupling color despill from alpha falloff:
#   - keyness ≥ HARD          → fully transparent (background)
#   - keyness in [ALPHA_SOFT, HARD) → partial alpha (anti-alias edge)
#   - keyness ≥ DESPILL_SOFT  → color neutralized (HIGH channels pulled
#                               down to max(LOW)), independent of alpha
#
# DESPILL_SOFT < ALPHA_SOFT on purpose: pixels with subtle key tint get
# color-neutralized while staying fully opaque. Catches the "hair pink fringe"
# case where back-lighting from the BG bleeds onto dark character pixels just
# enough to be visible (keyness ~8–18), without making hair translucent.
_KEYNESS_HARD = 60
_KEYNESS_ALPHA_SOFT = 20
_KEYNESS_DESPILL_SOFT = 8
_DEFAULT_KEY_COLOR = "#00FF00"  # legacy default — green screen


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse a `#RRGGBB` (or `RRGGBB`) hex string into an (r, g, b) tuple."""
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Expected 6-digit hex color (e.g. '#00FF00'), got {value!r}")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _key_axis(key_rgb: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    """Split key color channels into (high_indices, low_indices).

    A channel is "high" if it carries at least half the key's brightest channel
    (i.e. it's part of the chromatic identity), else "low". For #00FF00 →
    high=[1] (g), low=[0,2] (r,b). For #DB2B83 (magenta-pink) → high=[0,2]
    (r,b), low=[1] (g). For #00FFFF (cyan) → high=[1,2] (g,b), low=[0] (r).
    """
    threshold = max(key_rgb) / 2.0
    high = [i for i, v in enumerate(key_rgb) if v >= threshold]
    low = [i for i, v in enumerate(key_rgb) if v < threshold]
    if not low or not high:
        raise ValueError(
            f"Key color {key_rgb} has no clear chromatic axis (too neutral). "
            "Use a saturated color like #00FF00, #FF00FF, or #00FFFF."
        )
    return high, low


def _check_ffmpeg() -> tuple[str, str]:
    """Locate ffmpeg + ffprobe on PATH or raise an actionable error."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH. Install on macOS with: "
            "`brew install ffmpeg`"
        )
    return ffmpeg, ffprobe


def _probe_video_duration(ffprobe: str, video_path: str) -> float:
    """Return clip duration in seconds. Tries format duration first, then stream."""
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=duration",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    fmt_dur = data.get("format", {}).get("duration")
    if fmt_dur:
        return float(fmt_dur)
    streams = data.get("streams", [])
    if streams and streams[0].get("duration"):
        return float(streams[0]["duration"])
    raise RuntimeError(f"Could not determine duration for {video_path}")


def _extract_frames_from_video(
    video_path: str,
    frame_count: int,
    output_dir: Path,
) -> list[str]:
    """Extract `frame_count` evenly-spaced PNG frames from a video.

    Probes duration with ffprobe, then runs ffmpeg with `fps=count/duration`
    so frames are sampled uniformly across the clip. `-frames:v` caps any
    rounding overshoot. Borrowed from Sprite-Pipeline's
    `tools/extract_frames_ffmpeg.py`, trimmed to the constant-count case.
    """
    ffmpeg, ffprobe = _check_ffmpeg()
    duration = _probe_video_duration(ffprobe, video_path)
    if duration <= 0:
        raise RuntimeError(f"Video reports non-positive duration: {duration}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_fps = frame_count / duration
    pattern = str(output_dir / "extracted_%04d.png")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"fps={target_fps}",
        "-frames:v", str(frame_count),
        "-pix_fmt", "rgb24",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )

    frames = sorted(output_dir.glob("extracted_*.png"))
    if len(frames) < frame_count:
        raise RuntimeError(
            f"ffmpeg extracted {len(frames)} frames, expected {frame_count}. "
            f"Try a longer clip or a smaller frame_count."
        )
    return [str(p) for p in frames[:frame_count]]


def _sample_bg_color(input_path: str, edge_pad: int = 4) -> tuple[int, int, int]:
    """Sample the dominant background color from a frame's outer edge.

    Reads pixels from the four edges (a `edge_pad`-wide strip) and returns
    their median RGB. The edges are almost always pure background unless the
    character touches the frame edge — which is exactly the framing the
    pipeline tells callers to avoid.
    """
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    edge_pixels = np.concatenate([
        arr[:edge_pad, :, :].reshape(-1, 3),
        arr[-edge_pad:, :, :].reshape(-1, 3),
        arr[:, :edge_pad, :].reshape(-1, 3),
        arr[:, -edge_pad:, :].reshape(-1, 3),
    ])
    median = np.median(edge_pixels, axis=0).astype(int)
    return int(median[0]), int(median[1]), int(median[2])


def _chroma_key_distance(
    input_path: str,
    output_path: str,
    key_color: str | None = None,
    hard_threshold: int = 60,
    soft_threshold: int = 90,
    despill_threshold: int = 140,
) -> str:
    """Color-distance chroma keyer — robust on muddy / off-saturation backgrounds.

    Unlike `_chroma_key` (which uses a chromatic-axis "keyness" formula that
    over-keys when the background is desaturated or the character shares an
    axis with the bg), this scores each pixel by its Euclidean RGB distance
    from the key color and only keys pixels close to it:

      • distance ≤ hard_threshold        → fully transparent (background)
      • hard_threshold < d ≤ soft_threshold → partial alpha (anti-alias edge)
      • distance > soft_threshold        → fully opaque (foreground)

    DESPILL: pixels in a wider zone (distance ≤ despill_threshold) get
    color-corrected so the bg's chromatic contribution is removed. For a
    green key, this pulls the green channel down toward max(red,blue) on
    edge/fringe pixels. Without despill, anti-aliased character edges keep
    a visible halo of the bg color (e.g. a green outline around the entire
    character silhouette on a green-screen shoot). Despill operates on
    chromatic axis: HIGH channels of the key color are reduced toward the
    LOW channels' value, so red/skin/hoodie content stays intact.

    If `key_color` is None, auto-samples the dominant edge color from the
    frame — handles the common Seedance/Runway case where the user doesn't
    know the exact backdrop hex.

    This method works on any backdrop color (saturated or desaturated) and
    will not over-key character regions whose chroma axis coincidentally
    matches the bg, because distance is symmetric: a blue hoodie is far from
    pink in RGB space even if both have low green channel values.
    """
    if key_color is None:
        key_rgb = _sample_bg_color(input_path)
    else:
        key_rgb = _parse_hex_color(key_color)

    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[..., :3].astype(np.int32)
    a = arr[..., 3]

    diff = rgb - np.array(key_rgb, dtype=np.int32)
    distance = np.sqrt((diff * diff).sum(axis=-1))

    fully_bg = distance <= hard_threshold
    edge = (distance > hard_threshold) & (distance <= soft_threshold)
    edge_falloff = (distance - hard_threshold) / float(soft_threshold - hard_threshold)
    edge_alpha = np.clip(a.astype(np.float32) * edge_falloff, 0, 255).astype(np.uint8)
    a_new = np.where(fully_bg, np.uint8(0), np.where(edge, edge_alpha, a))

    # Despill: remove the bg color's chromatic contribution from any visible
    # pixel that has the bg's chromatic tint, regardless of overall distance
    # from the bg color. A pixel with R=180, G=200, B=80 is well clear of pure
    # green (#00FF00) by Euclidean distance, but still visibly greenish
    # because G > max(R,B). Distance-based despill misses these.
    #
    # Use keyness (min of HIGH channels minus max of LOW channels) as the
    # despill criterion. Any visible pixel with keyness above a small
    # threshold gets its HIGH channels pulled down toward max(LOW).
    #
    # Skipped if the key color is too neutral to have a chromatic axis
    # (e.g. pure grey backdrop — nothing to despill).
    out_rgb = arr[..., :3].copy()  # uint8
    try:
        high_idx, low_idx = _key_axis(key_rgb)
        high_min = np.minimum.reduce([rgb[..., i] for i in high_idx])
        low_max = np.maximum.reduce([rgb[..., i] for i in low_idx])
        keyness = high_min - low_max
        despill_zone = (keyness > 5) & (a_new > 0)
        for hi in high_idx:
            despilled = np.minimum(rgb[..., hi], low_max)
            out_rgb[..., hi] = np.where(
                despill_zone, despilled.astype(np.uint8), out_rgb[..., hi]
            )
    except ValueError:
        pass  # neutral key color — no despill axis

    out = np.dstack([out_rgb, a_new])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(output_path)
    return output_path


def _chroma_key(input_path: str, output_path: str, key_color: str = _DEFAULT_KEY_COLOR) -> str:
    """Remove a chroma-key background of arbitrary color from an image.

    Generalized algorithm — works for any saturated key color (green, magenta,
    cyan, blue, orange, etc.):

      1. Auto-detect the key's chromatic axis: which channels are HIGH
         (the dominant ones) and which are LOW (the suppressed ones).
      2. Compute keyness = min(HIGH channels) - max(LOW channels). A pixel
         that matches the key's hue scores high. Foreground pixels score low
         or negative (their LOW channels are typically as bright as their HIGH).
      3. Hard kill (alpha→0) for keyness ≥ _KEYNESS_HARD.
      4. Soft edge band (keyness in [SOFT, HARD)): partial alpha proportional
         to how key-like the pixel is, plus despill — pull each HIGH channel
         down toward max(LOW channels) so the residual key-color halo is
         neutralized on character outlines.

    For a green key #00FF00 this reduces exactly to the previous green-only
    formula (g - max(r,b)). For magenta #DB2B83 it becomes min(r,b) - g, which
    correctly catches anti-aliased magenta-tinted edges that the green-only
    formula was leaving behind.
    """
    key_rgb = _parse_hex_color(key_color)
    high_idx, low_idx = _key_axis(key_rgb)

    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    channels: list[np.ndarray] = [arr[..., i].astype(np.int16) for i in range(3)]
    a = arr[..., 3].astype(np.int16)

    high_min = np.minimum.reduce([channels[i] for i in high_idx])
    low_max = np.maximum.reduce([channels[i] for i in low_idx])
    keyness = high_min - low_max

    fully_bg = keyness >= _KEYNESS_HARD
    edge = (keyness >= _KEYNESS_ALPHA_SOFT) & (keyness < _KEYNESS_HARD)

    edge_falloff = (keyness - _KEYNESS_ALPHA_SOFT) / float(_KEYNESS_HARD - _KEYNESS_ALPHA_SOFT)
    edge_alpha = np.clip(a * (1.0 - edge_falloff), 0, 255).astype(np.int16)
    a_new = np.where(fully_bg, 0, np.where(edge, edge_alpha, a))

    # Despill — pull each HIGH channel down to max(LOW) for any pixel with
    # detectable key tint. Triggers at a LOWER threshold than alpha falloff,
    # so subtle BG bleed (e.g. pink reflection in dreadlocks) gets color-
    # neutralized while the pixel stays fully opaque.
    out_channels = list(channels)
    despill_mask = keyness >= _KEYNESS_DESPILL_SOFT
    for hi in high_idx:
        despilled = np.minimum(channels[hi], low_max)
        out_channels[hi] = np.where(despill_mask, despilled, channels[hi])

    out = np.stack([*out_channels, a_new], axis=-1).clip(0, 255).astype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(output_path)
    return output_path


def _strip_floor_rows(
    arr: np.ndarray,
    fill_threshold: float = 0.65,
    pad_x: int = 8,
) -> np.ndarray:
    """Strip wide floor/ground geometry from the bottom of the frame, while
    preserving the character's feet/boots that overlap the floor area.

    A naive "zero whole rows >threshold filled" approach destroys boots when
    the character is standing: at the row where boots meet the floor, the
    boots+floor combined exceed the threshold, and the boot pixels get axed
    along with the floor.

    Smarter approach:
      1. Walk up from the bottom; find the lowest row whose fill drops below
         `fill_threshold`. Call that y the "floor edge" — everything below
         is part of a wide floor surface.
      2. Compute the character's column-extent (x_min, x_max) from all rows
         ABOVE the floor edge. Add a small padding to allow boots that may
         flare slightly wider than the character's calves.
      3. Below the floor edge, zero alpha ONLY for columns OUTSIDE
         [x_min - pad, x_max + pad]. The wide flanking floor strip gets
         removed; the boots (which fall within the character X-range) are
         preserved. A narrow strip of floor immediately under the boots
         survives, but it's narrower than the character body, so it
         visually reads as a shadow base rather than a platform.

    No-op if no row exceeds `fill_threshold` (airborne / no floor in frame).
    Mutates and returns the array.
    """
    alpha = arr[..., 3]
    H, W = alpha.shape
    visible_per_row = (alpha > 0).sum(axis=1)
    threshold_count = int(W * fill_threshold)

    # Walk up from bottom while rows are floor-like; stop at first non-floor row.
    floor_edge_y = H
    for y in range(H - 1, -1, -1):
        if visible_per_row[y] >= threshold_count:
            floor_edge_y = y
        else:
            break

    if floor_edge_y >= H:
        return arr  # no floor strip detected

    # Compute character X-range from all rows above the floor edge
    above = alpha[:floor_edge_y, :] > 0
    if not above.any():
        # Nothing above the floor — frame is entirely floor. Zero everything.
        arr[..., 3] = 0
        return arr

    char_cols = np.any(above, axis=0)
    char_xs = np.where(char_cols)[0]
    x_min = max(0, int(char_xs.min()) - pad_x)
    x_max = min(W, int(char_xs.max()) + pad_x + 1)

    # Below the floor edge: zero only pixels OUTSIDE the character X-range
    arr[floor_edge_y:, :x_min, 3] = 0
    arr[floor_edge_y:, x_max:, 3] = 0
    return arr


def _keep_largest_component(
    input_path: str,
    output_path: str | None = None,
    erode_iters: int = 3,
    strip_floor_rows: bool = True,
) -> str:
    """Drop floor/shadow/set residue, keep only the character.

    AI-generated chroma-key footage often contains lit floor surfaces, ground
    shadows, or other set geometry that is NOT the key color and therefore
    survives `_chroma_key`. These leftovers come in three flavors:

      A. Fully-isolated blobs (easy)            — disconnected from character
      B. Thin contact points (medium)           — feet brushing a shadow
      C. Wide contact / full-width floor (hard) — character standing on floor
                                                  that fills the frame edge-
                                                  to-edge

    Three-phase cleanup handles all three:

      1. **Floor-row strip** (if `strip_floor_rows=True`): walk up from the
         bottom edge zeroing any row that's >65% opaque. Characters never
         fill that much of a row; floors regularly do. This handles the
         wide-floor case (C) directly, regardless of how the floor connects
         to the character.

      2. **Erode** by `erode_iters` pixels. Severs thin connections that
         survived step 1 (case B).

      3. **Connected-components labeling**. Keep only the largest blob in
         the eroded mask, then dilate it back so the character recovers its
         silhouette. Drops case-A blobs and any case-B blob now disconnected.

    No-op when no cleanup is needed.
    """
    from scipy.ndimage import label, binary_erosion, binary_dilation

    target = output_path or input_path
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    full_mask_before = arr[..., 3] > 0
    if not full_mask_before.any():
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        img.save(target)
        return target

    # Phase 1: floor-row strip
    if strip_floor_rows:
        arr = _strip_floor_rows(arr)

    full_mask = arr[..., 3] > 0
    if not full_mask.any():
        # Edge case: floor-strip removed everything (frame was entirely floor)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, "RGBA").save(target)
        return target

    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity

    # Phase 2: erode to sever thin connections
    eroded = binary_erosion(full_mask, structure=structure, iterations=erode_iters) \
        if erode_iters > 0 else full_mask
    if not eroded.any():
        eroded = full_mask  # erosion was too aggressive

    # Phase 3: label, keep largest, dilate back
    labeled, n_components = label(eroded, structure=structure)
    if n_components <= 1:
        keep_mask = full_mask
    else:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_label = int(sizes.argmax())
        largest_mask = labeled == largest_label
        if erode_iters > 0:
            restored = binary_dilation(largest_mask, structure=structure, iterations=erode_iters)
            keep_mask = restored & full_mask
        else:
            keep_mask = largest_mask

    arr[..., 3] = np.where(keep_mask, arr[..., 3], 0)
    dropped_px = int(full_mask_before.sum() - (arr[..., 3] > 0).sum())
    if dropped_px:
        logger.info(
            "_keep_largest_component: dropped %d pixel(s) of floor/shadow/set "
            "residue from %s (n_components=%d, erode_iters=%d)",
            dropped_px,
            input_path,
            n_components,
            erode_iters,
        )

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(target)
    return target


def _check_key_residue(
    frame_paths: list[str],
    key_color: str,
    trace_threshold: int = 5,
    mild_threshold: int = 10,
    strong_threshold: int = 20,
    max_trace_pixels: int = 100,
    max_mild_pixels: int = 5,
) -> dict:
    """Forensically check processed frames for residual key-color tint.

    Uses the same auto-detected chromatic axis as `_chroma_key`: a pixel's
    "keyness" is min(HIGH channels) - max(LOW channels). Higher values mean
    the pixel still leans toward the key color. After full pipeline cleanup
    every visible pixel should have keyness ≤ ~7 (sub-perceptible noise).

    Severity bands per frame:
        - clean   : max keyness ≤ trace_threshold (5)
        - trace   : a few pixels at trace level, fewer than max_trace_pixels
        - mild    : pixels above mild_threshold (10) but fewer than max_mild_pixels
        - severe  : any pixel above strong_threshold (20), OR mild count > max_mild_pixels

    Returns a dict shaped like _validate_frames so callers can render both
    validations the same way. `passed` per frame means severity ≤ trace.
    """
    key_rgb = _parse_hex_color(key_color)
    high_idx, low_idx = _key_axis(key_rgb)
    results = []
    total_residue = 0
    for p in frame_paths:
        img = np.array(Image.open(p).convert("RGBA"))
        chs = [img[..., i].astype(np.int16) for i in range(3)]
        a = img[..., 3]
        visible = a > 0
        if not visible.any():
            results.append({
                "path": p, "passed": True, "severity": "clean",
                "max_keyness": 0, "n_trace": 0, "n_mild": 0, "n_strong": 0,
                "issues": [],
            })
            continue
        high_min = np.minimum.reduce([chs[i] for i in high_idx])
        low_max = np.maximum.reduce([chs[i] for i in low_idx])
        keyness = high_min - low_max
        kv = np.where(visible, keyness, -999)
        max_k = int(kv.max())
        n_trace = int((kv > trace_threshold).sum())
        n_mild = int((kv > mild_threshold).sum())
        n_strong = int((kv > strong_threshold).sum())
        if n_strong > 0 or n_mild > max_mild_pixels:
            severity = "severe"
        elif n_mild > 0:
            severity = "mild"
        elif n_trace > max_trace_pixels:
            severity = "trace"
        else:
            severity = "clean"
        passed = severity in ("clean", "trace")
        issues: list[str] = []
        if not passed:
            issues.append(
                f"residual key tint: max keyness={max_k}, "
                f"{n_strong} strong / {n_mild} mild / {n_trace} trace pixels"
            )
        results.append({
            "path": p,
            "passed": passed,
            "severity": severity,
            "max_keyness": max_k,
            "n_trace": n_trace,
            "n_mild": n_mild,
            "n_strong": n_strong,
            "issues": issues,
        })
        total_residue += n_mild

    failed = [i for i, r in enumerate(results) if not r["passed"]]
    n = len(results)
    summary_lines = [
        f"{n - len(failed)}/{n} frames clean of {key_color} residue "
        f"(thresholds: trace>{trace_threshold}, mild>{mild_threshold}, strong>{strong_threshold})"
    ]
    for i, r in enumerate(results):
        marker = "✓" if r["passed"] else "✗"
        summary_lines.append(
            f"  {marker} frame_{i}: severity={r['severity']:>6} "
            f"max_keyness={r['max_keyness']:>3} "
            f"strong={r['n_strong']:>3} mild={r['n_mild']:>3} trace={r['n_trace']:>4}"
        )
    return {
        "key_color": key_color,
        "all_passed": len(failed) == 0,
        "failed_indices": failed,
        "frames": results,
        "summary": "\n".join(summary_lines),
    }


def _bg_remove_video_frame(
    input_path: str,
    output_path: str,
    api_key: str | None,
    key_color: str = _DEFAULT_KEY_COLOR,
    chroma_method: str = "distance",
) -> str:
    """Background-removal pipeline for a single video frame.

    Order:
      1. Chroma-key against `key_color` — primary, deterministic, handles
         chroma-keyed footage. Default is `#00FF00` (green); pass `#FF00FF`
         (magenta), `#00FFFF` (cyan), or any saturated hex to match other
         shoots.
      2. Keep only the largest connected component — drops floor/shadow/set
         residue that survived chroma-key because it wasn't key-colored.
      3. If chroma-key produced no foreground (frame had no key-colored BG
         at all), fall back to remove.bg if an API key is configured, else
         flood-fill.

    `chroma_method` selects the keyer:
      - "distance" (default): Euclidean RGB distance from key_color. Robust
        on muddy/off-saturation backgrounds (olive green, dark pink, etc.)
        and on characters whose chroma axis overlaps the bg.
      - "axis": original keyness-based logic. Optimal on cleanly-saturated
        backdrops (#00FF00, #FF00FF) when the character has no axis overlap.

    The chroma-key is intentionally always tried first: it precisely targets
    the colored backdrop the caller's video was shot against, and remove.bg
    can produce odd results on subjects that share that channel range.
    """
    if chroma_method == "distance":
        # When the caller hasn't overridden the legacy default ("#00FF00") we
        # treat that as "I don't know the exact bg hex" and auto-sample from
        # the frame's edge. Distance keying only fires within ~60 RGB units
        # of the key color, so passing literal "#00FF00" against an actual
        # backdrop of, say, #62B540 would leave most of the bg untouched.
        sample_bg = key_color in (_DEFAULT_KEY_COLOR, "auto", "", None)
        _chroma_key_distance(input_path, output_path, None if sample_bg else key_color)
    elif chroma_method == "axis":
        _chroma_key(input_path, output_path, key_color)
    else:
        raise ValueError(
            f"chroma_method must be 'distance' or 'axis', got {chroma_method!r}"
        )

    keyed = Image.open(output_path).convert("RGBA")
    if keyed.split()[-1].getbbox() is not None:
        # Drop disconnected floor/shadow blobs; keep only the character.
        _keep_largest_component(output_path)
        return output_path

    logger.warning(
        "Chroma-key found no foreground in %s — falling back to %s",
        input_path,
        "remove.bg" if api_key else "flood-fill",
    )
    if api_key:
        try:
            src = Image.open(input_path)
            result = sprite_processor._remove_bg_api(src, api_key)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            _keep_largest_component(output_path)
            return output_path
        except Exception as exc:
            logger.warning("remove.bg fallback failed (%s) — using flood-fill", exc)
    sprite_processor.strip_background(input_path, output_path, None)
    _keep_largest_component(output_path)
    return output_path


@mcp.tool()
async def generate_sprite_from_video(
    video_path: str,
    animation: str,
    character: str = "player",
    frame_count: int = 8,
    write_to_assets: bool = False,
    reference_image: str | None = None,
    key_color: str = "#00FF00",
    chroma_method: str = "distance",
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    char_width: int | None = None,
    foot_anchor_y: int | None = None,
) -> dict:
    """Process an AI-generated video clip into a Godot-ready sprite animation.

    Inspired by the Sprite-Pipeline approach (LayrKits/Sprite-Pipeline):
    feed in a chroma-keyed MP4 from Seedance/Runway/Kling/etc. and get back
    game-ready frames laid out on the project's standard sprite canvas.

    `video_path` accepts THREE forms:
      • Local file path (existing behavior).
      • Runway URL — `https://app.runwayml.com/creation/<UUID>`,
        `https://app.runwayml.com/share/<UUID>`, or `…/shared_assets/<UUID>`.
        The server hits Runway's public shared-assets API to resolve the
        actual mp4, downloads it, and caches it for repeat runs of the same
        URL. Requires the clip's "Anyone with the link" sharing to be ON.
      • Generic direct video URL — any `https://….mp4|.mov|.webm|.mkv`.
        Just gets downloaded.

    Cache lives at `$GAME_ASSETS_DIR/runway-cache/<uuid>/` (default
    `~/game-assets/runway-cache/`). Re-running the same URL is free.

    Pipeline:
      1. ffmpeg samples `frame_count` evenly-spaced PNGs across the clip.
      2. Each frame's chroma-key background is removed against `key_color`
         (auto-detects the chromatic axis — works for green #00FF00, magenta
         #FF00FF, cyan #00FFFF, etc.). remove.bg → flood-fill are fallbacks
         if the frame turns out to have no key-colored background.
      3. Frames are placed on the standard 100x250 canvas with feet anchored
         at y=238 using a SHARED scale factor across the whole animation —
         so character size stays consistent across poses.
      4. Frames are validated against the reference image for size /
         position drift (same checks as `generate_game_sprite`).
      5. A horizontal contact sheet and an animated GIF preview are built.
      6. Working files are copied to
         `{game_project_path}/_preview_tmp/{animation}/`.
      7. If `write_to_assets=True`, the final frames are written to
         `{game_project_path}/assets/characters/{character}/{animation}/frame_N.png`.

    Requires `ffmpeg` and `ffprobe` on PATH. On macOS:
        brew install ffmpeg

    `video_path` should point to a clip filmed against a solid chroma-key
    backdrop. `key_color` is the dominant hex color of that backdrop — pass
    the actual sampled BG color for best results (e.g. `#DB2B83` for a
    Seedance hot-pink backdrop, not just nominal `#FF00FF`). The character
    should occupy roughly the same area in every frame; the canvas-normalize
    step uses a single scale factor derived from the widest/tallest bbox
    across all frames.

    **Picking a key color:** prefer a color that does NOT appear in your
    character. Magenta (`#FF00FF` or pinkish) is a safer default for
    skin/foliage characters than green; use green for characters with no
    green elements. The algorithm auto-detects the chromatic axis from the
    color you pass.

    If `reference_image` is None, defaults to
    `{game_project_path}/assets/characters/{character}/idle/frame_0.png` —
    matches the default in `generate_game_sprite`.

    Dimension precedence (highest first): tool-call args, character config,
    defaults from config.json.
    """
    _check_ffmpeg()  # fail fast with an actionable message if ffmpeg is missing

    config = _load_config()
    overrides = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "char_width": char_width,
        "foot_anchor_y": foot_anchor_y,
    }
    eff = _resolve_dimensions(config, character, overrides)
    fps = int(eff.get("animation_fps", 8))

    video_p = await asyncio.to_thread(_resolve_video_to_local_path, video_path)
    if video_p.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension {video_p.suffix!r}; expected one of "
            f"{sorted(_VIDEO_EXTENSIONS)}"
        )
    if frame_count < 2:
        raise ValueError(f"frame_count must be ≥ 2, got {frame_count}")

    if reference_image is None:
        game_path = config.get("game_project_path", "")
        if not game_path:
            raise ValueError(
                "reference_image not provided and game_project_path is empty in "
                "config — cannot resolve default reference image."
            )
        reference_image = str(
            Path(game_path) / "assets" / "characters" / character / "idle" / "frame_0.png"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = OUTPUT_DIR / "sprites_video" / character / f"{animation}_{timestamp}"
    extracted_dir = work_dir / "extracted"
    keyed_dir = work_dir / "keyed"
    processed_dir = work_dir / "processed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    keyed_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths = await asyncio.to_thread(
        _extract_frames_from_video, str(video_p), frame_count, extracted_dir
    )

    # Validate key_color early so a malformed value fails before we burn time
    # on per-frame keying.
    _parse_hex_color(key_color)

    api_key = config.get("removebg_api_key", "") or None
    keyed_paths: list[str] = []
    for i, raw in enumerate(extracted_paths):
        out = str(keyed_dir / f"keyed_{i}.png")
        await asyncio.to_thread(_bg_remove_video_frame, raw, out, api_key, key_color, chroma_method)
        keyed_paths.append(out)

    processed_paths = [str(processed_dir / f"frame_{i}.png") for i in range(len(keyed_paths))]
    await asyncio.to_thread(
        sprite_processor.canvas_normalize_batch,
        keyed_paths,
        processed_paths,
        int(eff["canvas_width"]),
        int(eff["canvas_height"]),
        int(eff["char_width"]),
        int(eff["foot_anchor_y"]),
    )

    validation = await asyncio.to_thread(
        _validate_frames, processed_paths, reference_image
    )
    if not validation["all_passed"]:
        logger.warning(
            "generate_sprite_from_video: %d frame(s) failed validation. Details:\n%s",
            len(validation["failed_indices"]),
            validation["summary"],
        )

    # Forensic key-residue check — confirms the chroma-key BG removal actually
    # cleaned up. Runs against the same key_color the chroma keyer used, so it
    # works for any backdrop hue.
    key_residue = await asyncio.to_thread(
        _check_key_residue, processed_paths, key_color
    )
    if not key_residue["all_passed"]:
        logger.warning(
            "generate_sprite_from_video: %d frame(s) have residual %s tint. Details:\n%s",
            len(key_residue["failed_indices"]),
            key_color,
            key_residue["summary"],
        )

    contact_sheet_path = work_dir / "contact_sheet.png"
    contact_sheet.make_contact_sheet(
        processed_paths,
        str(contact_sheet_path),
        validation=validation["frames"],
    )

    gif_path: str | None = None
    if processed_paths:
        gif_target = work_dir / "preview.gif"
        try:
            gif_path = await asyncio.to_thread(
                _stitch_gif, processed_paths, str(gif_target), fps
            )
        except Exception as exc:
            logger.warning("Pillow GIF stitch failed (%s) — no preview GIF", exc)

    written = False
    game_path = config.get("game_project_path", "")

    # Always copy intermediate + final artifacts into the game project's
    # _preview_tmp/<animation>/ so Claude can review them regardless of
    # write_to_assets. Step labels are distinct from generate_game_sprite's
    # so reviewers can tell which pipeline produced which directory.
    if game_path:
        preview_dir = Path(game_path) / "_preview_tmp" / animation
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(extracted_paths):
            shutil.copy2(p, preview_dir / f"video_step1_extracted_{i}.png")
        for i, p in enumerate(keyed_paths):
            shutil.copy2(p, preview_dir / f"video_step2_bg_removed_{i}.png")
        for i, p in enumerate(processed_paths):
            shutil.copy2(p, preview_dir / f"video_step3_normalized_{i}.png")
        shutil.copy2(str(contact_sheet_path), preview_dir / "contact_sheet.png")
        for i, p in enumerate(processed_paths):
            shutil.copy2(p, preview_dir / f"frame_{i}.png")
        if gif_path:
            shutil.copy2(gif_path, preview_dir / "preview.gif")

    if write_to_assets:
        if not game_path:
            logger.warning("write_to_assets=True but game_project_path is empty")
        else:
            assets_dir = Path(game_path) / "assets" / "characters" / character / animation
            assets_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(processed_paths):
                shutil.copy2(src, assets_dir / f"frame_{i}.png")
            written = True

    return {
        "frames": processed_paths,
        "video": str(video_p),
        "extracted_frames": extracted_paths,
        "keyed_frames": keyed_paths,
        "contact_sheet": str(contact_sheet_path),
        "gif": gif_path,
        "written_to_assets": written,
        "reference_image": reference_image,
        "effective_config": eff,
        "validation": validation,
        "validation_summary": validation["summary"],
        "validation_passed": validation["all_passed"],
        "key_residue_check": key_residue,
        "key_residue_summary": key_residue["summary"],
        "key_residue_passed": key_residue["all_passed"],
    }


def _split_sheet_by_density(
    image_path: str,
    n_frames: int,
    key_color: str = "#00FF00",
    distance_threshold: int = 60,
) -> list[tuple[int, int]]:
    """Slice a horizontal sprite sheet into `n_frames` cells via density mins.

    Robust to tightly-packed sheets where figures touch each other (no pure
    background gaps between them). Algorithm:

      1. Compute per-column "character density" — count of non-background
         pixels in each column (where non-bg = pixel further than
         `distance_threshold` from `key_color` in RGB space).
      2. Find the leftmost and rightmost columns with any character
         pixels — that's the non-bg span.
      3. For each of N-1 expected boundaries (at i*span_width/N for
         i=1..N-1), search a ±40% window and pick the column with the
         lowest density. That's a "natural seam" between figures —
         where character pixels are sparsest, typically a hair gap or
         shoulder-to-shoulder pinch point.

    Works even when figures literally touch — the seam still has fewer
    character pixels than the figure interiors. Falls back gracefully on
    sheets WITH gaps (gaps trivially win the lowest-density search).
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    key_rgb = _parse_hex_color(key_color)
    diff = arr.astype(np.int32) - np.array(key_rgb, dtype=np.int32)
    distance = np.sqrt((diff * diff).sum(axis=-1))
    non_bg = distance > distance_threshold

    col_has = non_bg.any(axis=0)
    if not col_has.any():
        raise RuntimeError(
            f"No non-background pixels in {image_path} (key={key_color}, "
            f"threshold={distance_threshold}). Wrong key color?"
        )
    starts = np.where(col_has)[0]
    span_start, span_end = int(starts[0]), int(starts[-1]) + 1
    span_w = span_end - span_start

    col_density = non_bg[:, span_start:span_end].sum(axis=0).astype(float)
    expected_w = span_w / n_frames

    boundaries = [span_start]
    for i in range(1, n_frames):
        target = span_start + int(i * expected_w)
        radius = max(5, int(expected_w * 0.4))
        s_lo = max(span_start + 3, target - radius)
        s_hi = min(span_end - 3, target + radius)
        local = col_density[s_lo - span_start : s_hi - span_start]
        if len(local) == 0:
            boundaries.append(target)
        else:
            boundaries.append(s_lo + int(np.argmin(local)))
    boundaries.append(span_end)

    return list(zip(boundaries[:-1], boundaries[1:]))


@mcp.tool()
async def generate_sprite_from_sheet(
    image_path: str,
    animation: str,
    character: str = "player",
    frame_count: int = 8,
    write_to_assets: bool = False,
    reference_image: str | None = None,
    key_color: str = "#00FF00",
    chroma_method: str = "distance",
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    char_width: int | None = None,
    foot_anchor_y: int | None = None,
) -> dict:
    """Process a horizontal sprite sheet image into Godot-ready sprite frames.

    Companion to `generate_sprite_from_video` for image-source pipelines:
    feed in a sheet from Nano Banana / DALL-E / Midjourney / a hand-drawn
    sheet, and get back the same kind of normalized per-frame outputs.

    Pipeline:
      1. `_split_sheet_by_density` slices the sheet into `frame_count`
         cells using density-minimum boundaries (works on tightly-packed
         sheets where figures touch — no green gaps required).
      2. Each cell's chroma background is removed against `key_color`
         (auto-samples per-cell when `chroma_method="distance"`).
      3. Largest-component cleanup drops floor/shadow residue.
      4. Frames placed on the standard canvas with shared scale across
         the animation — same canvas-normalize logic as the video tool.
      5. Validation against reference image, key-residue forensic check,
         contact sheet, GIF preview, all the same artifacts as
         `generate_sprite_from_video`.
      6. If `write_to_assets=True`, frames written to
         `{game_project_path}/assets/characters/{character}/{animation}/frame_N.png`.

    `image_path` should point to a horizontal strip — N figures arranged
    left-to-right in roughly equal-width cells. The cells DO NOT need
    background gaps between them; the density splitter handles tight
    layouts. Cells DO need to be roughly equal width and have the same
    foot baseline across all figures.

    Use `frame_count` to tell the tool how many figures the sheet
    contains. The splitter assumes the count is exact — if you pass 8 and
    the sheet has 6, you'll get 8 cells split arbitrarily.

    Dimension precedence (highest first): tool-call args, character
    config, defaults from config.json.
    """
    if frame_count < 2:
        raise ValueError(f"frame_count must be ≥ 2, got {frame_count}")

    src = Path(image_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Sheet image not found: {src}")

    config = _load_config()
    overrides = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "char_width": char_width,
        "foot_anchor_y": foot_anchor_y,
    }
    eff = _resolve_dimensions(config, character, overrides)
    fps = int(eff.get("animation_fps", 8))

    if reference_image is None:
        game_path = config.get("game_project_path", "")
        if not game_path:
            raise ValueError(
                "reference_image not provided and game_project_path is "
                "empty in config — cannot resolve default reference image."
            )
        reference_image = str(
            Path(game_path) / "assets" / "characters" / character / "idle" / "frame_0.png"
        )

    _parse_hex_color(key_color)  # validate early

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = OUTPUT_DIR / "sprites_sheet" / character / f"{animation}_{timestamp}"
    sliced_dir = work_dir / "sliced"
    keyed_dir = work_dir / "keyed"
    processed_dir = work_dir / "processed"
    sliced_dir.mkdir(parents=True, exist_ok=True)
    keyed_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: density-min split
    boundaries = await asyncio.to_thread(
        _split_sheet_by_density, str(src), frame_count, key_color
    )

    sheet_img = Image.open(src).convert("RGB")
    sheet_arr = np.array(sheet_img)
    sliced_paths: list[str] = []
    for i, (x0, x1) in enumerate(boundaries):
        cell = Image.fromarray(sheet_arr[:, x0:x1, :])
        out = str(sliced_dir / f"sliced_{i}.png")
        cell.save(out)
        sliced_paths.append(out)

    # Step 2 + 3: bg removal + largest-component cleanup
    api_key = config.get("removebg_api_key", "") or None
    keyed_paths: list[str] = []
    for i, raw in enumerate(sliced_paths):
        out = str(keyed_dir / f"keyed_{i}.png")
        await asyncio.to_thread(
            _bg_remove_video_frame, raw, out, api_key, key_color, chroma_method
        )
        keyed_paths.append(out)

    # Step 4: canvas placement with shared scale
    processed_paths = [
        str(processed_dir / f"frame_{i}.png") for i in range(len(keyed_paths))
    ]
    await asyncio.to_thread(
        sprite_processor.canvas_normalize_batch,
        keyed_paths,
        processed_paths,
        int(eff["canvas_width"]),
        int(eff["canvas_height"]),
        int(eff["char_width"]),
        int(eff["foot_anchor_y"]),
    )

    # Step 5: validation + key-residue check
    validation = await asyncio.to_thread(
        _validate_frames, processed_paths, reference_image
    )
    if not validation["all_passed"]:
        logger.warning(
            "generate_sprite_from_sheet: %d frame(s) failed validation. "
            "Details:\n%s",
            len(validation["failed_indices"]),
            validation["summary"],
        )
    key_residue = await asyncio.to_thread(
        _check_key_residue, processed_paths, key_color
    )

    # Contact sheet + GIF
    contact_sheet_path = work_dir / "contact_sheet.png"
    contact_sheet.make_contact_sheet(
        processed_paths,
        str(contact_sheet_path),
        validation=validation["frames"],
    )
    gif_path: str | None = None
    if processed_paths:
        gif_target = work_dir / "preview.gif"
        try:
            gif_path = await asyncio.to_thread(
                _stitch_gif, processed_paths, str(gif_target), fps
            )
        except Exception as exc:
            logger.warning("Pillow GIF stitch failed (%s) — no preview GIF", exc)

    # Step 6: copy to game project
    written = False
    game_path = config.get("game_project_path", "")
    if game_path:
        preview_dir = Path(game_path) / "_preview_tmp" / animation
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(sliced_paths):
            shutil.copy2(p, preview_dir / f"sheet_step1_sliced_{i}.png")
        for i, p in enumerate(keyed_paths):
            shutil.copy2(p, preview_dir / f"sheet_step2_bg_removed_{i}.png")
        for i, p in enumerate(processed_paths):
            shutil.copy2(p, preview_dir / f"sheet_step3_normalized_{i}.png")
        shutil.copy2(str(contact_sheet_path), preview_dir / "contact_sheet.png")
        for i, p in enumerate(processed_paths):
            shutil.copy2(p, preview_dir / f"frame_{i}.png")
        if gif_path:
            shutil.copy2(gif_path, preview_dir / "preview.gif")

    if write_to_assets:
        if not game_path:
            logger.warning("write_to_assets=True but game_project_path is empty")
        else:
            assets_dir = (
                Path(game_path) / "assets" / "characters" / character / animation
            )
            assets_dir.mkdir(parents=True, exist_ok=True)
            for i, src_p in enumerate(processed_paths):
                shutil.copy2(src_p, assets_dir / f"frame_{i}.png")
            written = True

    return {
        "frames": processed_paths,
        "sheet": str(src),
        "boundaries": [{"x0": x0, "x1": x1} for x0, x1 in boundaries],
        "sliced_frames": sliced_paths,
        "keyed_frames": keyed_paths,
        "contact_sheet": str(contact_sheet_path),
        "gif": gif_path,
        "written_to_assets": written,
        "reference_image": reference_image,
        "effective_config": eff,
        "validation": validation,
        "validation_summary": validation["summary"],
        "validation_passed": validation["all_passed"],
        "key_residue_check": key_residue,
        "key_residue_summary": key_residue["summary"],
        "key_residue_passed": key_residue["all_passed"],
    }


@mcp.tool()
async def extract_body_parts_from_sheet(
    input_path: str,
    output_dir: str | None = None,
    boundary_color: str = "#FF00FF",
    boundary_tolerance: int = 80,
    padding: int = 10,
) -> dict:
    """Slice a body-parts reference sheet into individual transparent PNGs.

    The input image is the AI-generated character body parts sheet used to
    prepare assets for Spine 2D skeletal rigging. Each body part should be
    enclosed by a solid colored rectangle (default magenta #FF00FF) on a
    white background. This tool detects every rectangle, extracts each
    part's content, removes the white background and the colored boundary
    line, crops tight to the visible pixels with a transparent padding,
    and saves each part as a separate PNG.

    Numbering is reading order — top-to-bottom, left-to-right. A labeled
    debug image is written alongside the parts so the caller can visually
    confirm which rectangle got which number and rename the files to
    semantic names like `head.png`, `torso.png`, `arm_upper_l.png`, etc.

    Args:
      input_path: Path to the body parts sheet image (PNG/JPEG).
      output_dir: Directory to write the extracted parts. Defaults to
                  `<input_dir>/<input_stem>_parts/` next to the input image.
      boundary_color: Hex color of the rectangle outlines. Default `#FF00FF`.
                      Use any saturated hex (e.g. `#00FF00`, `#00FFFF`).
      boundary_tolerance: How loose the color match is, in summed-channel-
                          distance (0–765). Higher allows more drift from the
                          exact color. Default 80, which handles typical
                          anti-aliasing and slight model drift.
      padding: Transparent pixels added around each extracted part. Default
               10 — gives Spine some breathing room when rotating parts so
               there's no clipping at the bbox edges.

    Returns:
      A dict with `count` (number of parts extracted), `parts` (list with
      `path`, `index`, `rectangle_bbox`, `content_size` per part),
      `debug_image` (path to the labeled overview), and `output_dir`.
    """
    input_p = Path(input_path).expanduser().resolve()
    if not input_p.is_file():
        raise ValueError(f"Input image not found: {input_p}")
    if output_dir is None:
        output_dir = str(input_p.parent / f"{input_p.stem}_parts")

    raw = boundary_color.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Expected 6-digit hex color (e.g. '#FF00FF'), got {boundary_color!r}")
    rgb = tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))

    return await asyncio.to_thread(
        body_part_extractor.extract_body_parts_from_sheet,
        str(input_p),
        output_dir,
        rgb,
        int(boundary_tolerance),
        int(padding),
    )


def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it in the Claude Desktop MCP config for this server."
        )
    mcp.run()


if __name__ == "__main__":
    main()

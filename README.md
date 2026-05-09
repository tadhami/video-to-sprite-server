# video-to-sprite-server

An MCP server that turns AI-generated reference videos (Seedance, Runway, Kling, etc.) into Godot-ready sprite frames. Drop in a chroma-keyed mp4 and get back a sequence of canvas-normalized PNGs with the background fully removed and the character placed at a consistent foot anchor across the whole animation.

Also exposes the underlying image-generation tools (single asset, animated sprite sheet) for cases where you want to author character art without filming it.

---

## What it does

### `generate_sprite_from_video` — the headline feature

**Accepts a Runway URL directly** — no need to download the mp4 first. Paste any of these into `video_path` and the server resolves it to a local mp4 for you:

- `https://app.runwayml.com/creation/<UUID>`
- `https://app.runwayml.com/share/<UUID>`
- A direct CDN URL ending in `.mp4` / `.mov` / `.webm` / `.mkv`
- A regular local path (existing behavior unchanged)

Runway videos must have "Anyone with the link can view" enabled in the share dialog. Downloaded files are cached at `~/game-assets/runway-cache/<uuid>/` so re-running the same URL is free.

Pipeline:

1. Sample N evenly-spaced frames from the video with `ffmpeg`
2. Chroma-key the background using a generalized algorithm that auto-detects the chromatic axis from any saturated key color (green `#00FF00`, magenta `#FF00FF`, cyan `#00FFFF`, anything in between — pass the actual sampled hex)
3. Despill: pull the key-dominant channels down to neutralize subtle BG bleed at character edges
4. Strip the floor / ground geometry that the AI video model rendered (the floor isn't key-colored, so chroma keying alone leaves it behind)
5. Drop disconnected shadow / set-residue blobs via connected-components labeling
6. Place each frame on a configurable canvas (default 100×250) with feet anchored to a fixed y-coordinate, using a SHARED scale factor across the whole animation so character size doesn't pop between poses
7. Forensically verify there's no residual key-color tint (returned in the response, no need to manually inspect)

### `generate_game_sprite` — animated sprite sheets via image gen

Generates a horizontal sprite sheet via OpenRouter (Gemini image preview), then slices it with content-aware gap detection, applies background removal (remove.bg or flood-fill fallback), and runs the same canvas-normalize + validation pipeline as the video tool.

### `generate_2d_asset` — single-shot image gen

Plain prompt → image. Useful for concept art and character turnarounds.

---

## Why it exists

AI video models are great at generating reference footage of a character moving — much better than asking an image model to draw N consistent frames in a sheet. The bottleneck has been getting clean per-frame PNGs out of the video. This server is the bridge: video in, game-ready sprites out, with the chroma-key + cleanup steps you'd otherwise script by hand for every project.

---

## Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- `ffmpeg` and `ffprobe` on PATH (for the video tool)
  - macOS: `brew install ffmpeg`
- An [OpenRouter](https://openrouter.ai/) API key (for the image-gen tools)
- *Optional:* a [remove.bg](https://www.remove.bg/) API key — only used as a fallback when chroma keying fails (e.g. you fed it a non-keyed clip)

---

## Installation

### As an MCP server in Claude Desktop / Claude Code

1. Clone the repo:
   ```bash
   git clone https://github.com/tadhami/video-to-sprite-server.git
   cd video-to-sprite-server
   ```

2. Install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```

3. Copy the example config and fill in your paths:
   ```bash
   cp config.example.json config.json
   # edit config.json — set game_project_path and (optional) removebg_api_key
   ```

4. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "video-to-sprite-server": {
         "command": "/path/to/video-to-sprite-server/.venv/bin/python",
         "args": ["/path/to/video-to-sprite-server/server.py"],
         "env": {
           "OPENROUTER_API_KEY": "sk-or-v1-..."
         }
       }
     }
   }
   ```

5. Restart Claude Desktop / Claude Code.

---

## Configuration

`config.json` controls per-character canvas geometry and the optional remove.bg fallback. Resolution order: tool-call args → per-character override → top-level defaults.

```json
{
  "godot_executable": "/Applications/Godot.app/Contents/MacOS/Godot",
  "game_project_path": "/path/to/your/godot/project",
  "removebg_api_key": "",
  "defaults": {
    "canvas_width": 100,
    "canvas_height": 250,
    "char_width": 90,
    "foot_anchor_y": 238,
    "animation_fps": 8
  },
  "characters": {
    "player": {},
    "boss": { "canvas_width": 160, "canvas_height": 320, "char_width": 130 }
  }
}
```

Discovery: `$GAME_ASSET_SERVER_CONFIG` env var → `config.json` next to `server.py` (works regardless of CWD).

---

## Picking a key color

The chroma-key algorithm auto-detects the chromatic axis from the color you pass. Just sample your video's actual backdrop color and pass that hex. A few rules of thumb:

| Backdrop | Good for | Avoid for |
|---|---|---|
| `#00FF00` green | Characters with no green elements | Plant motifs, green eyes, green clothes |
| `#FF00FF` magenta / hot pink | Most fantasy/sci-fi characters | Characters with red/pink accents |
| `#00FFFF` cyan | Warm-toned characters (skin, fire) | Anything blue or teal |

You don't need the exact hex — the algorithm checks which channels are HIGH (≥half of max) vs LOW. A backdrop that the AI rendered as `#9B205D` will detect the same axis as `#FF00FF` and key correctly.

---

## Output verification

Every video-to-sprite call runs two automated checks and returns the results in the response:

- **Reference validation** — character height, width, and head position vs. a reference idle frame. Catches drift in character proportions.
- **Key-residue check** — pixel-level scan for any leftover BG hue, with severity bands (clean / trace / mild / severe). Uses the same axis-detection as the keyer, so it works for any key color.

Both are returned as `validation_summary` and `key_residue_summary` strings in the tool's response, so you can see at a glance whether anything needs another pass.

---

## License

MIT — see [LICENSE](./LICENSE).

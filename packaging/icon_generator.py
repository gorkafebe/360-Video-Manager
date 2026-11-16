"""Generates the application icon assets: icon.ico, icon.icns, icon_256.png.

Run directly from the repo root:

    python packaging/icon_generator.py
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

_SIZE = 256
_BG_COLOR = (16, 42, 64, 255)         # dark teal/blue
_SPHERE_COLOR = (235, 245, 250, 255)  # near-white
_PLAY_COLOR = (16, 42, 64, 255)       # same as background, reads against the sphere
_RING_COLOR = (110, 190, 230, 255)    # light-blue accent
_ICON_SIZES = (16, 32, 48, 64, 128, 256)

_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _draw_base_icon(size: int = _SIZE) -> Image.Image:
    """Draw a globe-with-play-button-and-orbit-rings icon at *size* px."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size * 0.04
    radius = size * 0.22
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius, fill=_BG_COLOR,
    )

    cx, cy = size / 2, size / 2

    # Orbit rings (tilted ellipses) suggesting 360°/VR motion around the globe.
    ring_w, ring_h = size * 0.78, size * 0.26
    for angle, width in ((-18, size * 0.020), (18, size * 0.014)):
        ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring)
        ring_draw.ellipse(
            [cx - ring_w / 2, cy - ring_h / 2, cx + ring_w / 2, cy + ring_h / 2],
            outline=_RING_COLOR, width=max(1, int(width)),
        )
        ring = ring.rotate(angle, resample=Image.BICUBIC, center=(cx, cy))
        img.alpha_composite(ring)

    # Sphere (globe).
    r = size * 0.30
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_SPHERE_COLOR)

    # Play triangle, optically centered (a centroid-centered triangle looks
    # left-shifted, so its centroid is nudged right by part of its width).
    t = r * 0.62
    offset_x = t * 0.12
    p1 = (cx - t * 0.55 + offset_x, cy - t)
    p2 = (cx - t * 0.55 + offset_x, cy + t)
    p3 = (cx + t * 0.85 + offset_x, cy)
    draw.polygon([p1, p2, p3], fill=_PLAY_COLOR)

    return img


def generate_icons(output_dir: str = _OUTPUT_DIR) -> None:
    base = _draw_base_icon(_SIZE)
    resized = {s: base.resize((s, s), Image.LANCZOS) for s in _ICON_SIZES}

    png_path = os.path.join(output_dir, "icon_256.png")
    resized[256].save(png_path)
    print(f"Wrote {png_path}")

    ico_path = os.path.join(output_dir, "icon.ico")
    resized[256].save(ico_path, format="ICO", sizes=[(s, s) for s in _ICON_SIZES])
    print(f"Wrote {ico_path}")

    icns_path = os.path.join(output_dir, "icon.icns")
    try:
        resized[256].save(icns_path, format="ICNS")
        print(f"Wrote {icns_path}")
    except Exception as exc:
        print(
            f"Skipped {icns_path}: {exc}\n"
            "Generate it on macOS instead (Pillow's ICNS writer needs it there), "
            "or convert icon_256.png with `iconutil`/`png2icns`."
        )


if __name__ == "__main__":
    generate_icons()

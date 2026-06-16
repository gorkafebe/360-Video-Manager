#!/usr/bin/env python3
"""Standalone diagnostic for line detection false-positive analysis.

Samples frames from a video, runs horizontal and vertical seam detection on each,
and prints a summary table with FFT dominance and quality scores. Debug images are
saved for each frame so you can visually inspect candidates.

Usage:
    python scripts/diagnose_line_detection.py /path/to/video.mp4
    python scripts/diagnose_line_detection.py /path/to/video.mp4 --frames 10
    python scripts/diagnose_line_detection.py /path/to/video.mp4 \\
        --debug-dir ./out/ --fft-min-dominance 0.20
"""

import argparse
import logging
import os
import sys

# Allow running from the project root without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from detector.line_detection import detect_horizontal_line, detect_vertical_line, draw_line_debug
from detector.preprocessing import prepare_frame_for_line_detection


def _parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose line detection on a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="Path to input video")
    p.add_argument("--frames", type=int, default=5,
                   help="Number of frames to sample (default: 5)")
    p.add_argument("--debug-dir", default="./line_detection_debug",
                   help="Output directory for debug images (default: ./line_detection_debug)")
    p.add_argument("--center-tolerance-ratio", type=float, default=0.02,
                   metavar="RATIO",
                   help="Max distance from centre as fraction of frame dimension (default: 0.02)")
    p.add_argument("--search-band-ratio", type=float, default=0.02,
                   metavar="RATIO",
                   help="Half-width of centre search band as fraction of frame dimension (default: 0.02)")
    p.add_argument("--fft-min-dominance", type=float, default=0.10,
                   metavar="DOM",
                   help="Minimum FFT horizontal-energy dominance to confirm detection (default: 0.10)")
    p.add_argument("--max-slope", type=float, default=0.05,
                   metavar="SLOPE",
                   help="Maximum allowed line slope (default: 0.05)")
    p.add_argument("--min-coverage-ratio", type=float, default=0.20,
                   metavar="RATIO",
                   help="Minimum seam coverage as fraction of frame dimension (default: 0.20)")
    return p.parse_args()


def _sample_frame_indices(total_frames: int, n: int):
    if total_frames <= 0 or n <= 0:
        return []
    step = max(1, total_frames // (n + 1))
    return [(i + 1) * step for i in range(n)]


def main():
    args = _parse_args()
    os.makedirs(args.debug_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    logger = logging.getLogger("diagnose")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    logger.info("Video: %s  (%dx%d, %.1f fps, %d frames)",
                os.path.basename(args.video), width, height, fps, total_frames)
    logger.info("Parameters: fft_min_dominance=%.2f  center_tolerance_ratio=%.3f  "
                "search_band_ratio=%.3f  max_slope=%.3f  min_coverage_ratio=%.2f",
                args.fft_min_dominance, args.center_tolerance_ratio,
                args.search_band_ratio, args.max_slope, args.min_coverage_ratio)

    frame_indices = _sample_frame_indices(total_frames, args.frames)
    rows = []

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("Could not read frame %d — skipping", frame_idx)
            continue

        prefix = f"f{i:02d}_frame{frame_idx}"

        # Save raw frame thumbnail
        cv2.imwrite(os.path.join(args.debug_dir, f"{prefix}_raw.jpg"), frame)

        # Save preprocessed (grayscale + blur)
        gray = prepare_frame_for_line_detection(frame)
        cv2.imwrite(os.path.join(args.debug_dir, f"{prefix}_gray.jpg"), gray)

        detect_kwargs = dict(
            center_tolerance_ratio=args.center_tolerance_ratio,
            search_band_ratio=args.search_band_ratio,
            max_slope=args.max_slope,
            min_coverage_ratio=args.min_coverage_ratio,
            fft_min_dominance=args.fft_min_dominance,
        )

        # Horizontal detection
        h_result = detect_horizontal_line(frame, **detect_kwargs)
        h_vis = draw_line_debug(frame, h_result)
        cv2.imwrite(os.path.join(args.debug_dir, f"{prefix}_horiz.jpg"), h_vis)

        # Vertical detection
        v_result = detect_vertical_line(frame, **detect_kwargs)

        has_h = bool(h_result.get("has_horizontal_line"))
        has_v = bool(v_result.get("has_vertical_line"))
        fft_dom_h = h_result.get("fft_dominance", float("nan"))
        fft_dom_v = v_result.get("fft_dominance", float("nan"))
        gate = h_result.get("debug_line_info", {}).get("quality_gate") or {}
        quality = gate.get("quality_score", float("nan"))

        rows.append({
            "frame": frame_idx,
            "h_line": "YES" if has_h else "no",
            "v_line": "YES" if has_v else "no",
            "fft_h": f"{fft_dom_h:+.3f}" if fft_dom_h == fft_dom_h else "  n/a",
            "fft_v": f"{fft_dom_v:+.3f}" if fft_dom_v == fft_dom_v else "  n/a",
            "quality": f"{quality:.3f}" if quality == quality else " n/a",
        })

        logger.info("Frame %5d: horiz=%-3s vert=%-3s  fft_h=%+.3f  fft_v=%+.3f  quality=%s",
                    frame_idx, "YES" if has_h else "no", "YES" if has_v else "no",
                    fft_dom_h if fft_dom_h == fft_dom_h else 0.0,
                    fft_dom_v if fft_dom_v == fft_dom_v else 0.0,
                    f"{quality:.3f}" if quality == quality else "n/a")

    cap.release()

    if not rows:
        logger.warning("No frames processed.")
        sys.exit(0)

    # Print summary table
    header = (
        f"{'Frame':>7}  {'H-Line':>7}  {'V-Line':>7}  "
        f"{'FFT-H':>8}  {'FFT-V':>8}  {'Quality':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['frame']:>7}  {r['h_line']:>7}  {r['v_line']:>7}  "
            f"{r['fft_h']:>8}  {r['fft_v']:>8}  {r['quality']:>8}"
        )
    print(sep)

    detected = sum(1 for r in rows if "YES" in (r["h_line"] + r["v_line"]))
    print(f"\n{detected}/{len(rows)} frames had a line detected.")
    print(f"Debug images: {os.path.abspath(args.debug_dir)}/")

    print(
        "\nTip: if false positives appear, try raising --fft-min-dominance (e.g. 0.20) "
        "or set VPD_LINE_FFT_MIN_DOMINANCE in your .env file."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

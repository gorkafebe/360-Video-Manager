"""CLI for 360-Video-Manager.

Usage
-----
.. code-block:: bash

    python -m app.main --url "https://youtu.be/XXXXXXXXXXX" --upload
    python -m app.main --url "360 sunset drone" --no-upload
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.logging_config import setup_logging
from workflows.unified_pipeline import JobOptions, process_video_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vr360-manager",
        description="Download, detect, convert, and upload 360° videos.",
    )
    parser.add_argument(
        "--url",
        metavar="URL_OR_QUERY",
        help="YouTube URL or search query. Required unless --local is used.",
    )
    parser.add_argument(
        "--local",
        metavar="VIDEO_PATH",
        help="Path to a local video file (skips download step).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for downloaded / converted files.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Upload title override.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Upload description.",
    )
    parser.add_argument(
        "--playlist",
        default=None,
        help="Existing MediaCMS playlist name or ID.",
    )
    parser.add_argument(
        "--new-playlist",
        metavar="NAME",
        default=None,
        help="Create a new playlist with this name and add the video to it.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the final asset to MediaCMS.",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip equirectangular conversion even when the projection warrants it.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum detection confidence to trigger conversion (default: 0.5).",
    )
    parser.add_argument(
        "--detection-frames",
        type=int,
        default=10,
        metavar="N",
        help="Number of frames to pass to the detector (default: 10).",
    )
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=5,
        metavar="N",
        help="Number of UI preview frames to extract (default: 5).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not save a JSON job manifest.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.url and not args.local:
        parser.error("Either --url or --local must be specified.")

    options = JobOptions(
        source_url=args.url or None,
        local_video_path=args.local or None,
        output_dir=args.output_dir,
        upload=args.upload,
        convert_if_needed=not args.no_convert,
        confidence_threshold=args.confidence_threshold,
        num_detection_frames=args.detection_frames,
        preview_frames=args.preview_frames,
        upload_title=args.title,
        upload_description=args.description,
        upload_playlist=args.playlist,
        upload_new_playlist=args.new_playlist,
        save_manifest=not args.no_manifest,
    )

    result = process_video_job(options)

    if result.success:
        print(f"[OK]  job_id={result.job_id}")
        print(f"      projection : {result.projection_type}")
        print(f"      confidence : {result.confidence:.1%}")
        if result.converted_video_path:
            print(f"      converted  : {result.converted_video_path}")
        if result.upload_result and result.upload_result.success:
            print(f"      upload     : {result.upload_result.media_url or 'OK'}")
        if result.manifest_path:
            print(f"      manifest   : {result.manifest_path}")
        return 0
    else:
        print(f"[FAILED] {result.error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Optional, Union

import click

from wtools.utils.video import _HWACCEL_DEVICE_MAP, video_to_jpeg_bin

logger = logging.getLogger(__name__)

_HWACCEL_CHOICES = ["auto"] + list(_HWACCEL_DEVICE_MAP.keys()) + ["none"]


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output .bin file path. Defaults to INPUT_PATH with .bin extension.",
)
@click.option(
    "-f",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the output file if it already exists.",
)
@click.option(
    "--sample-fps",
    type=float,
    default=2.0,
    show_default=True,
    help="Sampling frame rate (frames per second to extract).",
)
@click.option(
    "--jpeg-quality",
    type=int,
    default=95,
    show_default=True,
    help="JPEG encoding quality (1-100).",
)
@click.option(
    "--max-size",
    type=int,
    default=None,
    help="Resize frames so the longest side does not exceed this many pixels.",
)
@click.option(
    "--hwaccel",
    type=click.Choice(_HWACCEL_CHOICES, case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Hardware decoding backend: 'auto' (auto-detect), 'cuda', "
        "'videotoolbox', 'qsv', 'vaapi', or 'none' (software only)."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG-level) logging output.",
)
def main(
    input_path: str,
    output: Optional[str],
    overwrite: bool,
    sample_fps: float,
    jpeg_quality: int,
    max_size: Optional[int],
    hwaccel: str,
    verbose: bool,
) -> None:
    """Convert a video file to JPEGBIN1 (.bin) format.

    Extract frames from INPUT_PATH at the given sampling rate, JPEG-encode
    each frame, and write the result to a binary container file. This
    pre-processing step avoids decoding the video codec at training time.

    \b
    Examples:
        video-to-bin input.mp4
        video-to-bin input.mp4 -o output.bin --sample-fps 4.0 --max-size 448
        video-to-bin input.mp4 --hwaccel none
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )

    if not os.path.isfile(input_path):
        raise click.UsageError(f"Input path is not a file: {input_path!r}")

    if output is None:
        base, _ = os.path.splitext(input_path)
        output = base + ".bin"

    if os.path.exists(output) and not overwrite:
        raise click.UsageError(
            f"Output file already exists: {output!r}. "
            "Use --overwrite/-f to overwrite it."
        )

    # Translate CLI choice to the function's hwaccel parameter.
    if hwaccel.lower() == "auto":
        hwaccel_arg: Optional[Union[str, bool]] = None  # auto-detect
    elif hwaccel.lower() == "none":
        hwaccel_arg = False
    else:
        hwaccel_arg = hwaccel.lower()

    logger.info("Converting %s -> %s", input_path, output)
    try:
        info = video_to_jpeg_bin(
            input_path=input_path,
            output_path=output,
            sample_fps=sample_fps,
            jpeg_quality=jpeg_quality,
            max_size=max_size,
            hwaccel=hwaccel_arg,
        )
    except Exception:
        logger.error("Conversion failed; cleaning up partial output %s", output)
        if os.path.exists(output):
            os.remove(output)
        raise

    logger.info(
        "Done: %d frames, %dx%d, source_fps=%.1f, sample_fps=%.1f, file_size=%d bytes",
        info["nframes"],
        info["width"],
        info["height"],
        info["source_fps"],
        info["sample_fps"],
        info["file_size"],
    )


if __name__ == "__main__":
    main()

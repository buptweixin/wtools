#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Any, Optional

import click
import numpy as np
from tqdm import tqdm

from wtools.landmark import calculate_pitch_yaw_roll
from wtools.utils import dump_json, load_pts

logger = logging.getLogger(__name__)


@click.command()
@click.argument("img_pts_list_path", type=click.Path(exists=True))
@click.option(
    "--root_dir",
    type=str,
    default=None,
    help="Dataset root directory. Defaults to the directory of IMG_PTS_LIST_PATH.",
)
@click.option(
    "--pts_format",
    type=click.Choice(["sensetime", "mmc"]),
    default="sensetime",
    show_default=True,
    help="Landmark format used to select facial keypoint indices.",
)
@click.option(
    "--threshold",
    type=float,
    default=20.0,
    show_default=True,
    help="Pitch angle (degrees) separating low-pose from large-pose buckets.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the resampled img pts list. "
    "Defaults to '<root_dir>/img_pts_list_half_large_pose.txt'.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible np.random.choice sampling.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG-level) logging output.",
)
def main(
    img_pts_list_path: str,
    root_dir: Optional[str],
    pts_format: str,
    threshold: float,
    output: Optional[str],
    seed: int,
    verbose: bool,
) -> None:
    """Generate pose annotations from landmark files and resample by angle bucket.

    Read an image-points list, compute pitch/yaw/roll for each entry, bucket the
    samples by absolute pitch angle (low vs. large pose), and draw an evenly
    sized random subset per bucket. The resampled list is written to OUTPUT
    (or the default path inside ROOT_DIR).

    IMG_PTS_LIST_PATH is a text file where each line is
    ``<img_path> <pts_path> [...]``.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )

    np.random.seed(seed)

    if root_dir is None:
        root_dir = os.path.dirname(img_pts_list_path)

    if output is None:
        output = os.path.join(root_dir, "img_pts_list_half_large_pose.txt")

    with open(img_pts_list_path, "r") as f:
        lines = [l.strip().split() for l in f if l.strip()]

    total_num = len(lines)
    max_num_per_class = max(total_num // 3, 1)

    if total_num == 0:
        logger.warning("Input list is empty, nothing to do.")
        return

    logger.info(
        "Loaded %d entries from %s (max %d per bucket)",
        total_num,
        img_pts_list_path,
        max_num_per_class,
    )

    selected_indices: Any
    if pts_format == "sensetime":
        selected_indices = [33, 67, 68, 42, 52, 55, 58, 61, 47, 51, 84, 90, 93, 16]
    elif pts_format == "mmc":
        selected_indices = [33, 37, 42, 46, 51, 55, 61, 65, 78, 82, 86, 92, 95, 16]
    else:
        raise NotImplementedError(
            f"pts_format '{pts_format}' is not supported. "
            "Supported formats: 'sensetime', 'mmc'."
        )

    all_pose: list = []
    all_pose_dict: dict = {}
    for line in tqdm(lines, desc="Calculating poses"):
        pts_name = line[1]
        pts_path = os.path.join(root_dir, pts_name)
        # Guard against path traversal: if pts_name contains ".." or an
        # absolute path, the resolved path could escape root_dir.
        normalized = os.path.normpath(pts_path)
        root_normalized = os.path.normpath(root_dir)
        if not normalized.startswith(root_normalized + os.sep) and normalized != root_normalized:
            raise ValueError(
                f"Path traversal detected: {pts_name!r} resolves outside "
                f"root_dir {root_dir!r}."
            )
        pts = load_pts(pts_path, verbose=False)
        pts = pts[selected_indices]
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        all_pose_dict[pts_name] = {"pitch": pitch, "yaw": yaw, "roll": roll}
        pose = abs(pitch)
        all_pose.append(pose)

    dump_json(all_pose_dict, img_pts_list_path + ".json")
    logger.debug("Dumped per-sample pose dict to %s.json", img_pts_list_path)

    low_angle, large_angle = [], []
    for idx, pose in enumerate(all_pose):
        if pose < threshold:
            low_angle.append(idx)
        else:
            large_angle.append(idx)

    logger.info(
        "Bucket sizes (threshold=%.1f): low_angle=%d, large_angle=%d",
        threshold,
        len(low_angle),
        len(large_angle),
    )

    all_indices: list = []
    for name, poses in [("low_angle", low_angle), ("large_angle", large_angle)]:
        if len(poses) >= max_num_per_class:
            selected = np.random.choice(poses, max_num_per_class, replace=False)
        else:
            selected = np.random.choice(poses, max_num_per_class, replace=True)
            logger.warning(
                "%s has only %d samples (< %d); sampling with replacement",
                name,
                len(poses),
                max_num_per_class,
            )
        all_indices.append(selected)
    all_indices_np = np.concatenate(all_indices)

    logger.info(
        "Final sampled %d indices (%d unique)",
        len(all_indices_np),
        len(set(all_indices_np.tolist())),
    )

    new_lines = np.array(lines)[all_indices_np].tolist()
    with open(output, "w") as f:
        f.write("\n".join([" ".join(l) for l in new_lines]))
    logger.info("Dumped resampled img pts list to %s", output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from email.policy import default

import click
import numpy as np
from tqdm import tqdm

from wtools.landmark import calculate_pitch_yaw_roll
from wtools.utils import dump_json, load_pts


@click.command()
@click.argument("img_pts_list_path", type=click.Path(exists=True))
@click.option("--root_dir", type=str, default=None)
@click.option(
    "--pts_format", type=click.Choice(["sensetime", "mmc"]), default="sensetime"
)
def main(img_pts_list_path, root_dir, pts_format):
    """generate pose from landmark

    Arguments:
        root_dir {_type_} -- datset root dir
        input_file {_type_} -- image pts list file
        pts_format {_type_} -- pts format
    """

    if root_dir is None:
        root_dir = os.path.dirname(img_pts_list_path)
    img_pts_pose_list_path = os.path.join(root_dir, "img_pts_list_half_large_pose.txt")
    with open(img_pts_list_path, "r") as f:
        lines = [l.strip().split() for l in f]

    total_num = len(lines)
    max_num_per_class = total_num // 3

    if pts_format == "sensetime":
        selected_indices = [33, 67, 68, 42, 52, 55, 58, 61, 47, 51, 84, 90, 93, 16]
    elif pts_format == "mmc":
        selected_indices = [33, 37, 42, 46, 51, 55, 61, 65, 78, 82, 86, 92, 95, 16]
    else:
        raise NotImplementedError("Only sensetime format is supported.")

    all_pose = []
    all_pose_dict = {}
    for i in tqdm(range(len(lines))):
        pts_name = lines[i][1]
        pts_path = os.path.join(root_dir, pts_name)
        pts = load_pts(pts_path, verbose=False)
        pts = pts[selected_indices]
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        all_pose_dict[pts_name] = {"pitch": pitch, "yaw": yaw, "roll": roll}
        pose = abs(pitch)  # + abs(yaw)
        all_pose.append(pose)

    dump_json(all_pose_dict, img_pts_list_path + ".json")

    low_angle, mid_angle, large_angle = [], [], []
    for idx in range(len(all_pose)):
        pose = all_pose[idx]
        if pose < 20:
            low_angle.append(idx)
        else:
            large_angle.append(idx)
        # elif pose < 40:
        #     mid_angle.append(idx)

    all_indices = []
    for poses in [low_angle, large_angle]:
        if len(poses) >= max_num_per_class:
            selected_indices = np.random.choice(poses, max_num_per_class, replace=False)
        else:
            selected_indices = np.random.choice(poses, max_num_per_class, replace=True)
        all_indices.append(selected_indices)
    all_indices = np.concatenate(all_indices)
    print(
        f"num samples: {len(all_indices)}, dedumplicated samples: {len(set(all_indices))}"
    )
    new_lines = np.array(lines)[all_indices].tolist()
    with open(img_pts_pose_list_path, "w") as f:
        f.write("\n".join([" ".join(l) for l in new_lines]))
    print("dump resampled img pts list to {}".format(img_pts_pose_list_path))


if __name__ == "__main__":
    main()

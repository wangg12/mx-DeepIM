# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import division, print_function

import os
import sys
from math import pi
import numpy as np
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, ".."))
from lib.pair_matching.RT_transform import (
    euler2quat,
    mat2euler,
    calc_rt_dist_m,
    euler2mat,
)
from lib.utils.mkdir_if_missing import mkdir_if_missing


np.random.seed(2333)

# =================== global settings ======================
idx2class = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}

LM6d_occ_root = os.path.join(
    cur_dir, "../data/LINEMOD_6D/LM6d_converted/LM6d_occ_render_v1"
)
observed_set_root = os.path.join(LM6d_occ_root, "image_set/observed")
observed_data_root = os.path.join(LM6d_occ_root, "data/observed")

# gt_observed
gt_observed_root = os.path.join(LM6d_occ_root, "data/gt_observed/occ_test")

# output path
pose_dir = os.path.join(LM6d_occ_root, "rendered_poses_test")
mkdir_if_missing(pose_dir)
print("target path: {}".format(pose_dir))

sel_classes = idx2class.values()
num_rendered_per_observed = 1
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

angle_std, angle_max, x_std, y_std, z_std = [15.0, 45.0, 0.01, 0.01, 0.05]
print(
    "angle_std={}, angle_max={}, x_std={}, y_std={}, z_std={}".format(
        angle_std, angle_max, x_std, y_std, z_std
    )
)

image_set = "val"
for cls_idx, cls_name in idx2class.items():
    print(cls_idx, cls_name)
    # if cls_name != 'ape': # NOTE:comment here to generate for all classes
    #     continue
    rd_stat = []
    td_stat = []
    pose_observed = []
    pose_rendered = []

    sel_set_file = os.path.join(
        observed_set_root, "{}_{}.txt".format(cls_name, image_set)
    )
    with open(sel_set_file) as f:
        image_list = [x.strip() for x in f.readlines()]

    for observed_idx in tqdm(image_list):
        # get src_pose_m
        video_name, prefix = observed_idx.split("/")
        pose_path = os.path.join(
            gt_observed_root, cls_name, "{}-pose.txt".format(prefix)
        )
        src_pose_m = np.loadtxt(pose_path, skiprows=1)

        src_euler = np.squeeze(mat2euler(src_pose_m[:3, :3]))
        src_quat = euler2quat(src_euler[0], src_euler[1], src_euler[2]).reshape(1, -1)
        src_trans = src_pose_m[:, 3]
        pose_observed.append((np.hstack((src_quat, src_trans.reshape(1, 3)))))

        for rendered_idx in range(num_rendered_per_observed):
            tgt_euler = src_euler + np.random.normal(0, angle_std / 180 * pi, 3)
            x_error = np.random.normal(0, x_std, 1)[0]
            y_error = np.random.normal(0, y_std, 1)[0]
            z_error = np.random.normal(0, z_std, 1)[0]
            tgt_trans = src_trans + np.array([x_error, y_error, z_error])
            tgt_pose_m = np.hstack(
                (
                    euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]),
                    tgt_trans.reshape((3, 1)),
                )
            )
            r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
            transform = np.matmul(K, tgt_trans.reshape(3, 1))
            center_x = transform[0] / transform[2]
            center_y = transform[1] / transform[2]
            count = 0
            while r_dist > angle_max or not (
                16 < center_x < (640 - 16) and 16 < center_y < (480 - 16)
            ):
                tgt_euler = src_euler + np.random.normal(0, angle_std / 180 * pi, 3)
                x_error = np.random.normal(0, x_std, 1)[0]
                y_error = np.random.normal(0, y_std, 1)[0]
                z_error = np.random.normal(0, z_std, 1)[0]
                tgt_trans = src_trans + np.array([x_error, y_error, z_error])
                tgt_pose_m = np.hstack(
                    (
                        euler2mat(tgt_euler[0], tgt_euler[1], tgt_euler[2]),
                        tgt_trans.reshape((3, 1)),
                    )
                )
                r_dist, t_dist = calc_rt_dist_m(tgt_pose_m, src_pose_m)
                transform = np.matmul(K, tgt_trans.reshape(3, 1))
                center_x = transform[0] / transform[2]
                center_y = transform[1] / transform[2]
                count += 1
                if count == 100:
                    print(rendered_idx)

            tgt_quat = euler2quat(tgt_euler[0], tgt_euler[1], tgt_euler[2]).reshape(
                1, -1
            )
            pose_rendered.append(np.hstack((tgt_quat, tgt_trans.reshape(1, 3))))
            rd_stat.append(r_dist)
            td_stat.append(t_dist)
    rd_stat = np.array(rd_stat)
    td_stat = np.array(td_stat)
    print("r dist: {} +/- {}".format(np.mean(rd_stat), np.std(rd_stat)))
    print("t dist: {} +/- {}".format(np.mean(td_stat), np.std(td_stat)))

    output_file_name = os.path.join(
        pose_dir, "LM6d_occ_v1_{}_test_rendered_pose_{}.txt".format(image_set, cls_name)
    )
    with open(output_file_name, "w") as text_file:
        for x in pose_rendered:
            text_file.write("{}\n".format(" ".join(map(str, np.squeeze(x)))))

    print("{} finished".format(__file__))

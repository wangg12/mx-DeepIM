# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
"""
input: gt observed poses
generate rendered poses,
"""
from __future__ import division, print_function
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, ".."))

from lib.pair_matching.RT_transform import (
    mat2euler,
    euler2mat,
    euler2quat,
    calc_rt_dist_m,
)
from math import pi
from lib.utils.mkdir_if_missing import mkdir_if_missing
from tqdm import tqdm

np.random.seed(2333)

gt_observed_dir = os.path.join(
    cur_path, "..", "data", "LINEMOD_6D/LM6d_occ_ds_multi/data/render_observed"
)
observed_set_dir = os.path.join(
    cur_path, "..", "data/LINEMOD_6D/LM6d_occ_ds_multi/image_set/observed"
)

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
classes = idx2class.values()
classes = sorted(classes)


def class2idx(class_name, idx2class=idx2class):
    for k, v in idx2class.items():
        if v == class_name:
            return k


LM6d_occ_dsm_root = os.path.join(
    cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_occ_dsm"
)

# output dir
pose_dir = os.path.join(LM6d_occ_dsm_root, "ds_rendered_poses")
mkdir_if_missing(pose_dir)

sel_classes = classes
num_rendered_per_observed = 1  # 10
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
version = "v1"
angle_std, angle_max, x_std, y_std, z_std = [15.0, 45.0, 0.01, 0.01, 0.05]
print(angle_std, angle_max, x_std, y_std, z_std)
image_set = "NDtrain"


def main():
    for cls_name in tqdm(sel_classes):
        print(cls_name)
        # if cls_name != 'driller':
        #     continue
        rd_stat = []
        td_stat = []
        pose_observed = []
        pose_rendered = []

        observed_set_file = os.path.join(
            observed_set_dir, "NDtrain_observed_{}.txt".format(cls_name)
        )
        with open(observed_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]

        for data in image_list:
            pose_observed_path = os.path.join(
                gt_observed_dir, cls_name, data + "-pose.txt"
            )
            src_pose_m = np.loadtxt(pose_observed_path, skiprows=1)

            src_euler = np.squeeze(mat2euler(src_pose_m[:, :3]))
            src_quat = euler2quat(src_euler[0], src_euler[1], src_euler[2]).reshape(
                1, -1
            )
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
                    48 < center_x < (640 - 48) and 48 < center_y < (480 - 48)
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
                        print(
                            "{}: {}, {}, {}, {}".format(
                                data, r_dist, t_dist, center_x, center_y
                            )
                        )
                        print(
                            "count: {}, image_path: {}, rendered_idx: {}".format(
                                count,
                                pose_observed_path.replace("pose.txt", "color.png"),
                                rendered_idx,
                            )
                        )

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
            pose_dir,
            "LM6d_occ_dsm_{}_NDtrain_rendered_pose_{}.txt".format(version, cls_name),
        )
        with open(output_file_name, "w") as text_file:
            for x in pose_rendered:
                text_file.write("{}\n".format(" ".join(map(str, np.squeeze(x)))))
    print("{} finished".format(__file__))


if __name__ == "__main__":
    main()

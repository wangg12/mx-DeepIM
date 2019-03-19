# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
from __future__ import division, print_function

import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, ".."))
from lib.pair_matching import RT_transform
from lib.render_glumpy.render_py import Render_Py
from lib.utils.mkdir_if_missing import mkdir_if_missing

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
# config for render machine
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000

LM6d_occ_root = os.path.join(
    cur_dir, "../data/LINEMOD_6D/LM6d_converted/LM6d_occ_render_v1"
)
observed_set_root = os.path.join(LM6d_occ_root, "image_set/observed")
rendered_pose_path = "%s/LM6d_occ_v1_{}_test_rendered_pose_{}.txt" % (
    os.path.join(LM6d_occ_root, "rendered_poses_test"),
)

# output_path
rendered_root_dir = os.path.join(LM6d_occ_root, "data/rendered/occ_test")
pair_set_dir = os.path.join(LM6d_occ_root, "image_set")
mkdir_if_missing(rendered_root_dir)
mkdir_if_missing(pair_set_dir)
print("target path: {}".format(rendered_root_dir))
print("target path: {}".format(pair_set_dir))


def main():
    gen_images = True
    for class_idx, class_name in idx2class.items():
        val_pair = []
        print("start ", class_idx, class_name)
        if gen_images:
            # init renderer
            model_dir = os.path.join(LM6d_occ_root, "../models", class_name)
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        for set_type in ["val"]:
            with open(
                os.path.join(observed_set_root, "{}_{}.txt".format(class_name, "val")),
                "r",
            ) as f:
                test_observed_list = [x.strip() for x in f.readlines()]

            with open(rendered_pose_path.format(set_type, class_name)) as f:
                str_rendered_pose_list = [x.strip().split(" ") for x in f.readlines()]
            rendered_pose_list = np.array(
                [[float(x) for x in each_pose] for each_pose in str_rendered_pose_list]
            )
            rendered_per_observed = 1
            assert len(rendered_pose_list) == len(
                test_observed_list
            ), "{} vs {}".format(len(rendered_pose_list), len(test_observed_list))
            for idx, observed_index in enumerate(tqdm(test_observed_list)):
                video_name, observed_prefix = observed_index.split("/")
                rendered_dir = os.path.join(rendered_root_dir, class_name)
                mkdir_if_missing(rendered_dir)
                for inner_idx in range(rendered_per_observed):
                    if gen_images:
                        image_file = os.path.join(
                            rendered_dir,
                            "{}_{}-color.png".format(observed_prefix, inner_idx),
                        )
                        depth_file = os.path.join(
                            rendered_dir,
                            "{}_{}-depth.png".format(observed_prefix, inner_idx),
                        )
                        rendered_idx = idx * rendered_per_observed + inner_idx
                        pose_rendered_q = rendered_pose_list[rendered_idx]

                        rgb_gl, depth_gl = render_machine.render(
                            pose_rendered_q[:4], pose_rendered_q[4:]
                        )
                        rgb_gl = rgb_gl.astype("uint8")

                        depth_gl = (depth_gl * depth_factor).astype(np.uint16)

                        cv2.imwrite(image_file, rgb_gl)
                        cv2.imwrite(depth_file, depth_gl)

                        pose_rendered_file = os.path.join(
                            rendered_dir,
                            "{}_{}-pose.txt".format(observed_prefix, inner_idx),
                        )
                        text_file = open(pose_rendered_file, "w")
                        text_file.write("{}\n".format(class_idx))
                        pose_rendered_m = np.zeros((3, 4))
                        pose_rendered_m[:, :3] = RT_transform.quat2mat(
                            pose_rendered_q[:4]
                        )
                        pose_rendered_m[:, 3] = pose_rendered_q[4:]
                        pose_ori_m = pose_rendered_m
                        pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}".format(
                            pose_ori_m[0, 0],
                            pose_ori_m[0, 1],
                            pose_ori_m[0, 2],
                            pose_ori_m[0, 3],
                            pose_ori_m[1, 0],
                            pose_ori_m[1, 1],
                            pose_ori_m[1, 2],
                            pose_ori_m[1, 3],
                            pose_ori_m[2, 0],
                            pose_ori_m[2, 1],
                            pose_ori_m[2, 2],
                            pose_ori_m[2, 3],
                        )
                        text_file.write(pose_str)

                    if observed_index in test_observed_list:
                        if inner_idx == 0:
                            val_pair.append(
                                "{} {}/{}/{}_{}".format(
                                    observed_index,
                                    "occ_test",
                                    class_name,
                                    observed_prefix,
                                    inner_idx,
                                )
                            )

            test_pair_set_file = os.path.join(
                pair_set_dir, "my_val_{}.txt".format(class_name)
            )
            val_pair = sorted(val_pair)
            with open(test_pair_set_file, "w") as text_file:
                for x in val_pair:
                    text_file.write("{}\n".format(x))
        print(class_name, " done")


def check_observed_rendered():
    from lib.utils.utils import read_img
    import matplotlib.pyplot as plt

    observed_dir = os.path.join(LM6d_occ_root, "data/observed")

    for class_idx, class_name in idx2class.items():
        if class_name != "duck":
            continue
        print(class_name)
        observed_list_path = os.path.join(
            observed_set_root, "{}_val.txt".format(class_name)
        )
        with open(observed_list_path, "r") as f:
            observed_list = [x.strip() for x in f.readlines()]
        for idx, observed_index in enumerate(observed_list):
            print(observed_index)
            prefix = observed_index.split("/")[1]
            color_observed = read_img(
                os.path.join(observed_dir, observed_index + "-color.png"), 3
            )
            color_rendered = read_img(
                os.path.join(rendered_root_dir, class_name, prefix + "_0-color.png"), 3
            )
            fig = plt.figure()
            plt.axis("off")
            plt.subplot(1, 2, 1)
            plt.imshow(color_observed[:, :, [2, 1, 0]])
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(color_rendered[:, :, [2, 1, 0]])
            plt.axis("off")

            plt.show()


if __name__ == "__main__":
    main()
    # check_observed_rendered()
    print("{} finished".format(__file__))

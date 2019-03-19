# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
"""
generate data syn observed images, depths, labels, poses
and observed set index files
"""
from __future__ import print_function, division
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, ".."))
from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.pair_matching import RT_transform
from lib.render_glumpy.render_py_light_multi_program import Render_Py_Light_MultiProgram
import cv2
from six.moves import cPickle
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

random.seed(2333)
np.random.seed(1234)

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


def pose_q2m(pose_q):
    pose = np.zeros((3, 4), dtype="float32")
    pose[:3, :3] = RT_transform.quat2mat(pose_q[:4])
    pose[:3, 3] = pose_q[4:]
    return pose


# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  # for lm
ZNEAR = 0.25
ZFAR = 6.0

depth_factor = 1000

num_images = 20000  # 10 * num_observed
sel_classes = classes
num_class = len(sel_classes)
class_indices = [i for i in range(num_class)]

observed_prefix_list = ["{:06d}".format(i + 1) for i in range(num_images)]

LM6d_occ_dsm_root = syn_pose_dir = os.path.join(
    cur_path, "../data/LINEMOD_6D/LM6d_converted/LM6d_occ_dsm"
)
observed_pose_dir = os.path.join(LM6d_occ_dsm_root, "syn_poses_multi")


def gen_observed():
    gen_images = True
    if not gen_images:
        print("just generate observed set index files")
    model_root = os.path.join(cur_path, "../data/LINEMOD_6D/LM6d_converted/models")
    # output dir
    observed_root_dir = os.path.join(LM6d_occ_dsm_root, "data/observed")
    observed_set_dir = os.path.join(LM6d_occ_dsm_root, "image_set/observed")
    mkdir_if_missing(observed_root_dir)
    mkdir_if_missing(observed_set_dir)

    syn_pose_path = os.path.join(
        syn_pose_dir, "LM6d_occ_dsm_train_observed_pose_all.pkl"
    )
    with open(syn_pose_path, "rb") as f:
        syn_pose_dict = cPickle.load(f)

    if gen_images:
        # init render machine
        brightness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4]  # ##################
        model_folder_dict = {}
        for class_name in sel_classes:
            if class_name == "__background__":
                continue
            model_folder_dict[class_name] = os.path.join(model_root, class_name)
        render_machine = Render_Py_Light_MultiProgram(
            sel_classes,
            model_folder_dict,
            K,
            width,
            height,
            ZNEAR,
            ZFAR,
            brightness_ratios,
        )

    observed_set_path = os.path.join(observed_set_dir, "train_observed_{}.txt")

    train_idx_dict = {cls_name: [] for cls_name in classes}

    for idx in tqdm(range(num_images)):
        prefix = observed_prefix_list[idx]
        if len(syn_pose_dict[prefix].keys()) < 3:
            continue
        random.shuffle(class_indices)

        if idx % 500 == 0:
            print("idx: {}, prefix: {}".format(idx, prefix))

        if gen_images:
            # generate random light_position
            if idx % 6 == 0:
                light_position = [1, 0, 1]
            elif idx % 6 == 1:
                light_position = [1, 1, 1]
            elif idx % 6 == 2:
                light_position = [0, 1, 1]
            elif idx % 6 == 3:
                light_position = [-1, 1, 1]
            elif idx % 6 == 4:
                light_position = [-1, 0, 1]
            elif idx % 6 == 5:
                light_position = [0, 0, 1]
            else:
                raise Exception("???")

            # randomly adjust color and intensity for light_intensity
            colors = np.array(
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            )
            intensity = np.random.uniform(0.8, 1.2, size=(3,))
            colors_randk = random.randint(0, colors.shape[0] - 1)
            light_intensity = colors[colors_randk] * intensity

            # randomly choose a render machine(brightness_ratio)
            rm_randk = random.randint(0, len(brightness_ratios) - 1)

            rgb_dict = {}
            label_dict = {}
            depth_dict = {}
            pose_dict = {}
            depth_mean_dict = {}
            classes_in_pose = syn_pose_dict[prefix].keys()
            # print('num classes:', len(classes_in_pose))
            pose_0 = pose_q2m(
                syn_pose_dict[prefix][classes_in_pose[0]]
            )  # the first pose

        for cls_idx in class_indices:
            cls_name = sel_classes[cls_idx]
            if cls_name not in classes_in_pose:
                continue
            train_idx_dict[cls_name].append("./" + prefix)

            if gen_images:
                pose_q = syn_pose_dict[prefix][cls_name]
                # print(pose_q)
                pose = pose_q2m(pose_q)
                pose_dict[cls_name] = pose

                light_position_0 = np.array(light_position) * 0.5
                # inverse yz
                # print('pose_0: ', pose_0)
                light_position_0[0] += pose_0[0, 3]
                light_position_0[1] -= pose_0[1, 3]
                light_position_0[2] -= pose_0[2, 3]

                rgb_gl, depth_gl = render_machine.render(
                    RT_transform.mat2quat(pose[:3, :3]),
                    pose[:, -1],
                    light_position_0,
                    light_intensity,
                    class_name=cls_name,
                    brightness_k=rm_randk,
                )
                rgb_gl = rgb_gl.astype("uint8")

                rgb_dict[cls_name] = rgb_gl

                label_gl = np.zeros(depth_gl.shape)
                label_gl[depth_gl != 0] = 1
                label_dict[cls_name] = label_gl

                depth_dict[cls_name] = depth_gl * depth_factor
                depth_mean_dict[cls_name] = np.mean(depth_gl[depth_gl != 0])

        if gen_images:
            # get rendered results together
            res_img = np.zeros(rgb_gl.shape, dtype="uint8")
            res_depth = np.zeros(depth_gl.shape, dtype="uint16")
            res_label = np.zeros(label_gl.shape)

            means = depth_mean_dict.values()
            gen_indices = np.argsort(np.array(means)[::-1])
            gen_classes = depth_mean_dict.keys()
            for cls_idx in gen_indices:
                cls_name = gen_classes[cls_idx]
                if cls_name not in classes_in_pose:
                    continue
                tmp_rgb = rgb_dict[cls_name]
                tmp_label = label_dict[cls_name]
                for i in range(3):
                    res_img[:, :, i][tmp_label == 1] = tmp_rgb[:, :, i][tmp_label == 1]

                # label
                res_label[tmp_label == 1] = class2idx(cls_name)

                # depth
                tmp_depth = depth_dict[cls_name]
                res_depth[tmp_label == 1] = tmp_depth[tmp_label == 1]

            res_depth = res_depth.astype("uint16")

        def vis_check():
            fig = plt.figure(figsize=(8, 6), dpi=200)
            plt.axis("off")
            fig.add_subplot(1, 3, 1)
            plt.imshow(res_img[:, :, [2, 1, 0]])
            plt.axis("off")
            plt.title("res_img")

            plt.subplot(1, 3, 2)
            plt.imshow(res_depth / depth_factor)
            plt.axis("off")
            plt.title("res_depth")

            plt.subplot(1, 3, 3)
            plt.imshow(res_label)
            plt.axis("off")
            plt.title("res_label")

            fig.suptitle(
                "light position_0: {}\n light_intensity: {}\n brightness: {}".format(
                    light_position_0, light_intensity, brightness_ratios[rm_randk]
                )
            )
            plt.show()

        # vis_check()
        if gen_images:
            # write results -------------------------------------
            observed_color_file = os.path.join(observed_root_dir, prefix + "-color.png")
            observed_depth_file = os.path.join(observed_root_dir, prefix + "-depth.png")
            observed_label_file = os.path.join(observed_root_dir, prefix + "-label.png")
            poses_file = os.path.join(observed_root_dir, prefix + "-poses.npy")

            cv2.imwrite(observed_color_file, res_img)
            cv2.imwrite(observed_depth_file, res_depth)
            cv2.imwrite(observed_label_file, res_label)
            np.save(poses_file, pose_dict)
            # pass

    # write observed set index
    for cls_name in sel_classes:
        train_indices = train_idx_dict[cls_name]
        train_indices = sorted(train_indices)
        with open(observed_set_path.format(cls_name), "w") as f:
            for line in train_indices:
                f.write(line + "\n")
    print("done")


if __name__ == "__main__":
    gen_observed()
    print("{} finished".format(__file__))

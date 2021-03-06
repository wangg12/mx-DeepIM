# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
"""
calculate visible rate for each object in each image, if the visible rate is less than 15%,
remove that img_idx in the train_indices of the object.
"""
from __future__ import print_function, division
import numpy as np
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
from lib.utils.utils import read_img
from tqdm import tqdm


model_root = os.path.join(cur_dir, "..", "data", "LINEMOD_6D/LM6d_converted/models")
LM6d_occ_dsm_root = os.path.join(
    cur_dir, "../data/LINEMOD_6D/LM6d_converted/LM6d_occ_dsm"
)
observed_dir = os.path.join(LM6d_occ_dsm_root, "data/observed")
observed_set_dir = os.path.join(LM6d_occ_dsm_root, "image_set/observed")
gt_observed_dir = os.path.join(LM6d_occ_dsm_root, "data/gt_observed")

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


def class2idx(class_name, idx2class=idx2class):
    for k, v in idx2class.items():
        if v == class_name:
            return k


classes = idx2class.values()
classes = sorted(classes)

# config for renderer
width = 640
height = 480
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ZNEAR = 0.25
ZFAR = 6.0
depth_factor = 1000

sel_classes = classes
num_class = len(sel_classes)


def main():
    for cls_idx, cls_name in enumerate(tqdm(sel_classes)):
        print(cls_idx, cls_name)
        keyframe_path = os.path.join(
            observed_set_dir, "train_observed_{}.txt".format(cls_name)
        )
        with open(keyframe_path) as f:
            observed_index_list = [x.strip() for x in f.readlines()]
        video_name_list = [x.split("/")[0] for x in observed_index_list]  # ./
        observed_prefix_list = [x.split("/")[1] for x in observed_index_list]

        high_occ_list = []
        NDtrain_list = []
        occ_rates = []
        for idx, observed_index in enumerate(tqdm(observed_index_list)):
            prefix = observed_prefix_list[idx]
            video_name = video_name_list[idx]

            observed_label_file = os.path.join(
                observed_dir, video_name, prefix + "-label.png"
            )
            gt_observed_label_file = os.path.join(
                gt_observed_dir, cls_name, video_name, prefix + "-label.png"
            )

            observed_label = read_img(observed_label_file, 1)
            gt_observed_label = read_img(gt_observed_label_file, 1)

            occ_rate = cal_occ_rate(observed_label, gt_observed_label, cls_name)
            occ_rates.append(occ_rate)
            if occ_rate > 0.85:
                high_occ_list.append(observed_index)
            else:
                NDtrain_list.append(observed_index)
        occ_rates = np.array(occ_rates)
        print(
            "occ rate stat, mean: {}, std: {}".format(
                np.mean(occ_rates), np.std(occ_rates)
            )
        )
        print(
            "high occ: {}, NDtrain: {}, all: {}".format(
                len(high_occ_list), len(NDtrain_list), len(observed_index_list)
            )
        )
        with open(
            os.path.join(observed_set_dir, "NDtrain_observed_{}.txt".format(cls_name)),
            "w",
        ) as f:
            for line in NDtrain_list:
                f.write(line + "\n")

        with open(
            os.path.join(
                observed_set_dir, "HighOcc_train_observed_{}.txt".format(cls_name)
            ),
            "w",
        ) as f:
            for line in high_occ_list:
                f.write(line + "\n")

    print("done")


def cal_occ_rate(observed_label, gt_observed_label, cls_name, binary_mask=False):
    observed_label_idx = class2idx(cls_name)
    if binary_mask:
        observed_label_idx = 1
    visible_pixels = np.sum(
        np.logical_and(observed_label == observed_label_idx, gt_observed_label == 1)
    )

    all_pixels = np.sum(gt_observed_label[gt_observed_label == 1])
    if all_pixels == 0 or visible_pixels == 0:
        occ_rate = 1
    else:
        occ_rate = 1 - visible_pixels / all_pixels

    if False:  # occ_rate > 0.85:
        print("cls_name: {}".format(cls_name))
        print("gt_observed_label: {}".format(np.unique(gt_observed_label)))
        print("observed labal: {}".format(np.unique(observed_label)))
        print("observed_label_idx: {}".format(observed_label_idx))
        print(
            "visible: {}, all: {}, occ_rate: {}".format(
                visible_pixels, all_pixels, occ_rate
            )
        )
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.subplot(1, 3, 1)
        tmp = (observed_label == observed_label_idx).astype("uint8")
        plt.imshow(tmp)
        plt.subplot(1, 3, 2)
        plt.imshow(gt_observed_label)
        plt.title("gt observed")
        plt.subplot(1, 3, 3)
        plt.imshow(observed_label)
        plt.title("observed")
        plt.show()
    assert occ_rate >= 0, "visible: {}, all: {}".format(visible_pixels, all_pixels)
    return occ_rate


def check_observed_gt_observed():
    observed_dir = os.path.join(LM6d_occ_dsm_root, "data/observed")
    gt_observed_dir = os.path.join(LM6d_occ_dsm_root, "data/gt_observed")
    cls_name = "can"
    indices = ["{:06d}".format(i) for i in range(1, 20)]

    for prefix in indices:
        gt_observed_color_file = os.path.join(
            gt_observed_dir, cls_name, prefix + "-color.png"
        )
        gt_observed_depth_file = os.path.join(
            gt_observed_dir, cls_name, prefix + "-depth.png"
        )
        gt_observed_label_file = os.path.join(
            gt_observed_dir, cls_name, prefix + "-label.png"
        )

        observed_color_file = os.path.join(observed_dir, prefix + "-color.png")
        observed_depth_file = os.path.join(observed_dir, prefix + "-depth.png")
        observed_label_file = os.path.join(observed_dir, prefix + "-label.png")

        color_r = read_img(observed_color_file, 3)
        depth_r = read_img(observed_depth_file, 1) / depth_factor
        label_r = read_img(observed_label_file, 1)

        color_rr = read_img(gt_observed_color_file, 3)
        depth_rr = read_img(gt_observed_depth_file, 1) / depth_factor
        label_rr = read_img(gt_observed_label_file, 1)

        visible_pixels = np.sum(
            np.logical_and(label_r == classes.index(cls_name) + 1, label_rr == 1)
        )
        all_pixels = np.sum(label_rr[label_rr == 1])
        print("visible: {}, all: {}".format(visible_pixels, all_pixels))

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6), dpi=120)
        plt.axis("off")
        fig.add_subplot(2, 3, 1)
        plt.imshow(color_r[:, :, [2, 1, 0]])
        plt.axis("off")
        plt.title("color observed")

        plt.subplot(2, 3, 2)
        plt.imshow(depth_r)
        plt.axis("off")
        plt.title("depth observed")

        plt.subplot(2, 3, 3)
        plt.imshow(label_r)
        plt.axis("off")

        fig.add_subplot(2, 3, 4)
        plt.imshow(color_rr[:, :, [2, 1, 0]])
        plt.axis("off")
        plt.title("color gt_observed")

        plt.subplot(2, 3, 5)
        plt.imshow(depth_rr)
        plt.axis("off")
        plt.title("depth gt_observed")

        plt.subplot(2, 3, 6)
        plt.imshow(label_rr)
        plt.axis("off")
        plt.title("label gt_observed")
        plt.show()
        # assert visible_pixels <= all_pixels


# def stat_lm6d_occ_rate():
#     LM6d_root = os.path.join(cur_dir, "../data/LINEMOD_6D/LM6d_converted/LM6d_refine")
#     for cls_idx, cls_name in enumerate(tqdm(sel_classes)):
#         print(cls_idx, cls_name)
#         observed_dir = os.path.join(LM6d_root, "data/observed")
#         gt_observed_dir = os.path.join(LM6d_root, "data/gt_observed")
#         set_name = "train"
#         keyframe_path = os.path.join(
#             LM6d_root,
#             "image_set/observed/{}_{}.txt".format(cls_name, set_name),
#         )
#         with open(keyframe_path) as f:
#             observed_index_list = [x.strip() for x in f.readlines()]
#         video_name_list = [x.split("/")[0] for x in observed_index_list]  # ./
#         observed_prefix_list = [x.split("/")[1] for x in observed_index_list]

#         high_occ_list = []
#         NDtrain_list = []
#         occ_rates = []
#         for idx, observed_index in enumerate(tqdm(observed_index_list)):
#             prefix = observed_prefix_list[idx]
#             video_name = video_name_list[idx]

#             observed_label_file = os.path.join(observed_dir, video_name, prefix + "-label.png")
#             gt_observed_label_file = os.path.join(
#                 gt_observed_dir, cls_name, video_name, prefix + "-label.png"
#             )

#             observed_label = read_img(observed_label_file, 1)
#             gt_observed_label = read_img(gt_observed_label_file, 1)

#             occ_rate = cal_occ_rate(
#                 observed_label, gt_observed_label, cls_name, binary_mask=True
#             )
#             occ_rates.append(occ_rate)
#             if occ_rate > 0.85:
#                 high_occ_list.append(observed_index)
#             else:
#                 NDtrain_list.append(observed_index)
#         occ_rates = np.array(occ_rates)
#         print(
#             "occ rate stat, mean: {}, std: {}".format(
#                 np.mean(occ_rates), np.std(occ_rates)
#             )
#         )
#         print(
#             "high occ: {}, NDtrain: {}, all: {}".format(
#                 len(high_occ_list), len(NDtrain_list), len(observed_index_list)
#             )
#         )

#     print("done")


if __name__ == "__main__":
    # stat_lm6d_occ_rate()
    main()
    # check_observed_gt_observed()

# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Gu Wang, Yi Li
# --------------------------------------------------------
"""
generate pair set
"""
from __future__ import print_function, division
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(cur_path, ".."))
from lib.utils.mkdir_if_missing import mkdir_if_missing
import random

random.seed(2333)
np.random.seed(2333)
from tqdm import tqdm

image_set_dir = os.path.join(
    cur_path, "..", "data/LINEMOD_6D/LM6d_converted/LM6d_occ_render_v1/image_set"
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


def main():
    set_type = "train"
    for class_idx, class_name in enumerate(tqdm(classes)):
        print("start ", class_idx, class_name)

        f_observed = os.path.join(
            image_set_dir, "observed/{}_{}.txt".format(class_name, set_type)
        )
        pairs = []
        with open(f_observed, "r") as f:
            for line in f:
                real_idx = line.strip("\r\n")
                real_suffix = real_idx.split("/")[1]
                for rendered_i in range(10):
                    rendered_idx = "{}/{}_{}".format(
                        class_name, real_suffix, rendered_i
                    )
                    pairs.append("{} {}".format(real_idx, rendered_idx))

        with open(
            os.path.join(image_set_dir, "{}_{}.txt".format(set_type, class_name)), "w"
        ) as f:
            for pair in pairs:
                f.write(pair + "\n")


if __name__ == "__main__":
    main()

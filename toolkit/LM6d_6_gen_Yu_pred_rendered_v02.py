# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Gu Wang
# --------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import os
from shutil import copyfile
from lib.utils.mkdir_if_missing import *
from lib.render_glumpy.render_py import Render_Py
from lib.pair_matching import RT_transform
import scipy.io as sio
import cv2
from tqdm import tqdm

if __name__=='__main__':
    GEN_EGGBOX = False
    # GEN_EGGBOX = True # change this line for eggbox, because eggbox in the first version is wrong

    big_idx2class = {
        1: 'ape',
        2: 'benchvise',
        4: 'camera',
        5: 'can',
        6: 'cat',
        8: 'driller',
        9: 'duck',
        10: 'eggbox',
        11: 'glue',
        12: 'holepuncher',
        13: 'iron',
        14: 'lamp',
        15: 'phone'
    }
    class_name_list = big_idx2class.values()
    class_name_list = sorted(class_name_list)

    class2big_idx = {}
    for key in big_idx2class:
        class2big_idx[big_idx2class[key]] = key

    cur_path = os.path.abspath(os.path.dirname(__file__))

    # config for Yu's results
    keyframe_path = "%s/{}_test.txt"%(os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/image_set/real'))
    if not GEN_EGGBOX:
        yu_pred_dir = os.path.join(cur_path, '../data/LINEMOD_6D/results_frcnn_linemod')
    else:
        yu_pred_dir = os.path.join(cur_path, '../data/LINEMOD_6D/frcnn_LM6d_eggbox_yu_val_v02_fix')

    # config for renderer
    width = 640
    height = 480
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    depth_factor = 1000

    gen_images = True
    # output_path
    version = 'rendered_Yu_v02'
    real_root_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/data/real')
    real_meta_path = "%s/{}-meta.mat"%(real_root_dir)
    rendered_root_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/data', version)
    pair_set_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/image_set')
    mkdir_if_missing(rendered_root_dir)
    mkdir_if_missing(pair_set_dir)
    all_pair = []
    for small_class_idx, class_name in enumerate(tqdm(class_name_list)):
        if GEN_EGGBOX:
            if class_name != 'eggbox': # eggbox is wrong in the first version
                continue
        else:
            if class_name == 'eggbox':
                continue
        big_class_idx = class2big_idx[class_name]
        # init render
        model_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_converted/models/{}'.format(class_name))
        if gen_images:
            render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

        with open(keyframe_path.format(class_name)) as f:
            real_index_list = [x.strip() for x in f.readlines()]
        video_name_list = [x.split('/')[0] for x in real_index_list]
        real_prefix_list = [x.split('/')[1] for x in real_index_list]

        all_pair = []
        for idx, real_index in enumerate(real_index_list):
            rendered_dir = os.path.join(rendered_root_dir, video_name_list[idx], class_name)
            mkdir_if_missing(rendered_dir)
            yu_idx = idx
            yu_pred_file = os.path.join(yu_pred_dir, class_name, "{:04d}.mat".format(yu_idx))
            yu_pred = sio.loadmat(yu_pred_file)
            rois = yu_pred['rois']
            if len(rois) != 0:
                labels = rois[:, 0] # 1: found; -1: not found
                if labels != -1:
                    try:
                        meta_data = sio.loadmat(real_meta_path.format(real_index))
                    except:
                        raise Exception(real_index)
                    proposal_idx = np.where(labels == 1)
                    assert len(proposal_idx) == 1
                    pose_ori_q = yu_pred['poses'][proposal_idx].reshape(7)

                    pose_ori_m = RT_transform.se3_q2m(pose_ori_q)
                    pose_ori_q[:4] = RT_transform.mat2quat(pose_ori_m[:, :3])

                    pose_gt = meta_data['poses']
                    if len(pose_gt.shape) > 2:
                        pose_gt = pose_gt[:, :, list(meta_data['cls_indexes'][0]).index(big_class_idx)]
                    print("{}, {:04d}, {}".format(class_name, idx, RT_transform.calc_rt_dist_m(pose_ori_m, pose_gt)))

                    pose_ori_file = os.path.join(rendered_dir, '{}_{}_{}-pose.txt'.format(class_name, real_prefix_list[idx], 0))
                    image_file = os.path.join(rendered_dir, '{}_{}_{}-color.png'.format(class_name, real_prefix_list[idx], 0))
                    depth_file = os.path.join(rendered_dir, '{}_{}_{}-depth.png'.format(class_name, real_prefix_list[idx], 0))
                    segmentation_file = os.path.join(rendered_dir, '{}_{}_{}-label.png'.format(class_name, real_prefix_list[idx], 0))

                    if gen_images:
                        rgb_gl, depth_gl = render_machine.render(pose_ori_q[:4], pose_ori_q[4:])
                        depth_gl = (depth_gl * depth_factor).astype(np.uint16)

                        segmentation = np.zeros(depth_gl.shape)
                        segmentation[depth_gl != 0] = 1
                        cv2.imwrite(image_file, rgb_gl)
                        cv2.imwrite(depth_file, depth_gl)
                        cv2.imwrite(segmentation_file, segmentation)

                    text_file = open(pose_ori_file, 'w')
                    text_file.write("{}\n".format(big_class_idx))
                    pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                        .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                                pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                                pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
                    text_file.write(pose_str)

                    all_pair.append("{} {}/{}/{}_{}_{}"
                                    .format(real_index,
                                            video_name_list[idx], class_name, class_name, real_prefix_list[idx], 0))
                else:
                    print("no yu_pred in {} {} {}, label=-1 ********************".format(class_name, yu_idx, real_index))
            else:
                print("no yu_pred in {}, {:04d}, {}".format(class_name, yu_idx, real_index))
                pose_ori_m = np.zeros((3, 4))
                pose_ori_m[:] = -1

                pose_ori_file = os.path.join(rendered_dir,
                                             '{}_{}_{}-pose.txt'.format(class_name, real_prefix_list[idx], 0))

                image_file = os.path.join(rendered_dir,
                                          '{}_{}_{}-color.png'.format(class_name, real_prefix_list[idx], 0))
                depth_file = os.path.join(rendered_dir,
                                          '{}_{}_{}-depth.png'.format(class_name, real_prefix_list[idx], 0))
                segmentation_file = os.path.join(rendered_dir,
                                                 '{}_{}_{}-label.png'.format(class_name, real_prefix_list[idx], 0))
                if gen_images:
                    rgb_gl = np.zeros((height, width, 3), dtype=np.uint8)
                    depth_gl = np.zeros((height, width), dtype=np.uint16)
                    segmentation = np.zeros((height, width), dtype=np.uint8)

                    cv2.imwrite(image_file, rgb_gl)
                    cv2.imwrite(depth_file, depth_gl)
                    cv2.imwrite(segmentation_file, segmentation)

                text_file = open(pose_ori_file, 'w')
                text_file.write("{}\n".format(big_class_idx))
                pose_str = "{} {} {} {}\n{} {} {} {}\n{} {} {} {}" \
                    .format(pose_ori_m[0, 0], pose_ori_m[0, 1], pose_ori_m[0, 2], pose_ori_m[0, 3],
                            pose_ori_m[1, 0], pose_ori_m[1, 1], pose_ori_m[1, 2], pose_ori_m[1, 3],
                            pose_ori_m[2, 0], pose_ori_m[2, 1], pose_ori_m[2, 2], pose_ori_m[2, 3])
                text_file.write(pose_str)

                all_pair.append("{} {}/{}/{}_{}_{}"
                                .format(real_index,
                                        video_name_list[idx], class_name, class_name, real_prefix_list[idx], 0))

            pair_set_file = os.path.join(pair_set_dir, "yu_val_v02_{}.txt".format(class_name))
            with open(pair_set_file, "w") as text_file:
                for x in all_pair:
                    text_file.write("{}\n".format(x))

        print(class_name, " done")

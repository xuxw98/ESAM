# Copyright (c) OpenMMLab. All rights reserved.
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python demo/online_demo.py
from argparse import ArgumentParser

import pdb
import os
from os import path as osp
from copy import deepcopy
import random
import numpy as np
import torch
import mmengine
from mmdet3d.apis import init_model
from mmengine.dataset import Compose, pseudo_collate
from concurrent import futures as futures
from pathlib import Path

from tools.update_infos_to_v2 import get_empty_standard_data_info, clear_data_info_unused_keys
from oneformer3d.scannet_dataset import ScanNet200SegDataset_
from show_result import show_seg_result


class ScanNet200SVDataConverter(object):
    """ScanNet data.
    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        scannet200 (bool): True for ScanNet200, else for ScanNet.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, cfg):
        self.root_dir = root_path
        self.split_dir = osp.join(root_path)
        
        # Use dataset only for `parse_data_info`, pipeline is explicitly applied
        self.dataset = ScanNet200SegDataset_(
            ann_file='scannet200_sv_oneformer3d_infos_val.pkl',
            data_root=cfg.data_root,
            data_prefix=cfg.data_prefix,
            metainfo=dict(classes=cfg.class_names),
            pipeline=cfg.test_pipeline,
            ignore_index=cfg.num_semantic_classes,
            test_mode=True
        )

    # def get_axis_align_matrix(self, idx):
    #     matrix_file = osp.join(self.root_dir, 'scannet_sv_instance_data',
    #                            f'{idx}_axis_align_matrix.npy')
    #     mmengine.check_file_exist(matrix_file)
    #     return np.load(matrix_file)

    def process_single_scene(self, sample_idx):
        ## Data process
        info = dict()
        pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info
        info['pts_path'] = osp.join('points', f'{sample_idx}.bin')
        info['super_pts_path'] = osp.join('super_points', f'{sample_idx}.bin')
        info['pts_instance_mask_path'] = osp.join(
            'instance_mask', f'{sample_idx}.bin')
        info['pts_semantic_mask_path'] = osp.join(
            'semantic_mask', f'{sample_idx}.bin')

        # annotations = {}
        # axis_align_matrix = self.get_axis_align_matrix(sample_idx)
        # annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
        # info['annos'] = annotations

        ## Update to v2
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['lidar_points']['num_pts_feats'] = info[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            info['pts_path']).name
        if 'pts_semantic_mask_path' in info:
            temp_data_info['pts_semantic_mask_path'] = Path(
                info['pts_semantic_mask_path']).name
        if 'pts_instance_mask_path' in info:
            temp_data_info['pts_instance_mask_path'] = Path(
                info['pts_instance_mask_path']).name
        if 'super_pts_path' in info:
            temp_data_info['super_pts_path'] = Path(
                info['super_pts_path']).name

        # anns = info.get('annos', None)
        # ignore_class_name = set()
        # if anns is not None:
        #     temp_data_info['axis_align_matrix'] = anns[
        #         'axis_align_matrix'].tolist()
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)

        ## Dataset process
        temp_data_info = self.dataset.parse_data_info(temp_data_info)

        return temp_data_info


def inference_detector(model, scene_idx):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    scannetmv_converter = ScanNet200SVDataConverter(root_path="./data/scannet200-sv_full", cfg=cfg)
    data = scannetmv_converter.process_single_scene(scene_idx)
    original_points = np.fromfile(data['lidar_points']['lidar_path'], np.float32).reshape(-1,6)

    data = [test_pipeline(data)]
    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    return results[0], data[0], original_points


def main():
    parser = ArgumentParser()
    scene_name = "scene0667_00_1000"
    parser.add_argument('--scene-idx', type=str, default="%s" % scene_name, help='single scene index')
    parser.add_argument('--config', type=str, default="configs/mixformer3d_sv_1xb4_scannet200.py", help='Config file')
    parser.add_argument('--checkpoint', type=str, default="work_dirs/mf3d_scannet200_sv_128e_v4x3GAP_cat_agnostic/epoch_128_G_s_add.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # model init
    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    # test a single scene
    result, data, ori_points = inference_detector(model=model, scene_idx=args.scene_idx)
    super_points = data['data_samples'].eval_ann_info['sp_pts_mask']
    gt_ins = data['data_samples'].eval_ann_info['pts_instance_mask']
    gt_sem = data['data_samples'].eval_ann_info['pts_semantic_mask']
    pred_ins_mask = result.pred_pts_seg.pts_instance_mask[0]
    pred_ins_score = result.pred_pts_seg.instance_scores
    pred_sem = result.pred_pts_seg.pts_semantic_mask[0]
    palette = [random.sample(range(0, 255), 3) for i in range(200)]

    # Superpoint results
    show_seg_result(ori_points, super_points, super_points, 
        out_dir='./demo/data', filename='%s_superpoint' % scene_name, palette=np.array(palette))

    # Semseg results
    show_seg_result(ori_points, gt_sem, pred_sem, 
        out_dir='./demo/data', filename='%s_semseg' % scene_name, palette=np.array(palette), ignore_index=200)
    
    # Insseg results
    pred_instance_masks_sort = torch.Tensor(pred_ins_mask[pred_ins_score.argsort()])
    pred_instance_masks_label = pred_instance_masks_sort[0].long() - 1
    for i in range(1, pred_instance_masks_sort.shape[0]):
        pred_instance_masks_label[pred_instance_masks_sort[i].bool()] = i

    np.random.seed(0)
    palette = np.random.random((max(max(pred_instance_masks_label) + 2, max(gt_ins) + 2), 3)) * 255
    palette[-1] = 200

    show_seg_result(ori_points, gt_ins, pred_instance_masks_label, 
        out_dir='./demo/data', filename='%s_insseg' % scene_name, palette=palette)


if __name__ == '__main__':
    main()




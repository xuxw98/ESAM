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

from oneformer3d.scannet_dataset import ScanNet200SegMVDataset_
from show_result import show_seg_result, write_oriented_bbox, show_seg_result_ply


class ScanNet200MVDataConverter(object):
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
        self.dataset = ScanNet200SegMVDataset_(
            ann_file='scannet200_mv_oneformer3d_infos_train.pkl',
            data_root=cfg.data_root,
            data_prefix=cfg.data_prefix,
            metainfo=dict(classes=cfg.class_names),
            pipeline=cfg.test_pipeline,
            ignore_index=cfg.num_semantic_classes,
            test_mode=True
        )

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'axis_align_matrix',
                               f'{idx}.npy')
        mmengine.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def process_single_scene(self, sample_idx):
        ## Data process
        info = dict()
        pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info
        files = os.listdir(osp.join(self.root_dir, 'points', sample_idx))
        files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
        info['pts_paths'] = [osp.join('points', sample_idx, file) for file in files]
        info['super_pts_paths'] = [osp.join('super_points', sample_idx, file) for file in files]
        info['pts_instance_mask_paths'] = [osp.join('instance_mask', sample_idx, file) for file in files]
        info['pts_semantic_mask_paths'] = [osp.join('semantic_mask', sample_idx, file) for file in files]

        annotations = {}
        axis_align_matrix = self.get_axis_align_matrix(sample_idx)
        annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
        info['annos'] = annotations

        ## Update to v2
        anns = info.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            info['axis_align_matrix'] = anns[
                'axis_align_matrix'].tolist()
        info.pop('annos')

        ## Dataset process
        info = self.dataset.parse_data_info(info)

        return info


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
    test_pipeline = [
        dict(
            type='LoadAdjacentDataFromFile',
            coord_type='DEPTH',
            shift_height=False,
            use_color=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5],
            num_frames=-1,
            num_sample=20000,
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=True,
            with_seg_3d=True,
            with_sp_mask_3d=True,
            with_rec=True, cat_rec=True,
            dataset_type='scannet200'),
        dict(type='SwapChairAndFloorWithRec'),
        dict(type='PointSegClassMappingWithRec'),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='NormalizePointsColor_',
                    color_mean=(0.47793125906962 * 255, 0.4303257521323044 * 255, 0.3749598901421883 * 255),
                    color_std=(0.2834475483823543 * 255, 0.27566157565723015 * 255, 0.27018971370874995 * 255)),
                dict(
                    type='AddSuperPointAnnotations_Online',
                    num_classes=200,
                    stuff_classes=[0, 1],
                    merge_non_stuff_cls=False,
                    with_rec=True),
                dict(type='BboxCalculation', voxel_size=0.02),
            ]),
        dict(type='Pack3DDetInputs_Online', keys=['points', 'sp_pts_mask', 'gt_bboxes_3d'] + ['gt_sp_masks'])
    ]
    test_pipeline = Compose(test_pipeline)

    scannetmv_converter = ScanNet200MVDataConverter(root_path="./data/scannet200-mv", cfg=cfg)
    data = scannetmv_converter.process_single_scene(scene_idx)

    data = [test_pipeline(data)]
    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    return results[0], data[0]


def main():
    parser = ArgumentParser()
    scene_name = "scene0131_00"
    parser.add_argument('--scene-idx', type=str, default="%s" % scene_name, help='single scene index')
    parser.add_argument('--config', type=str, default="configs/mixformer3d_online_1xb4_scannet200.py", help='Config file')
    parser.add_argument('--checkpoint', type=str, default="work_dirs/mf3d_scannet200_online_128e_final/epoch_128.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # model init
    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    # test a single scene
    result, data = inference_detector(model=model, scene_idx=args.scene_idx)
    ori_points = data['data_samples'].eval_ann_info['rec_xyz']
    gt_ins = data['data_samples'].eval_ann_info['pts_instance_mask']
    gt_sem = data['data_samples'].eval_ann_info['pts_semantic_mask']
    gt_bbox = data['data_samples'].eval_ann_info['gt_bboxes_3d'] # x, y, z, w, h, l, 0
    pred_ins_mask = result.pred_pts_seg.pts_instance_mask[0]
    pred_ins_score = result.pred_pts_seg.instance_scores
    pred_sem = result.pred_pts_seg.pts_semantic_mask[0]
    pred_bbox = result.pred_bbox # x1, y1, z1, x2, y2, z2
    pred_bbox = np.stack([(pred_bbox[:, 0] + pred_bbox[:, 3]) / 2,
                             (pred_bbox[:, 1] + pred_bbox[:, 4]) / 2,
                             (pred_bbox[:, 3] + pred_bbox[:, 5]) / 2,
                             pred_bbox[:, 3] - pred_bbox[:, 0],
                             pred_bbox[:, 4] - pred_bbox[:, 1],
                             pred_bbox[:, 5] - pred_bbox[:, 2],
                             np.zeros_like(pred_bbox[:, 0])], axis=1)

    # Bounding box results
    write_oriented_bbox(gt_bbox, osp.join('./demo/data', f'{scene_name}_box_gt.ply'))
    write_oriented_bbox(pred_bbox, osp.join('./demo/data', f'{scene_name}_box_pred.ply'))

    # Semseg results
    palette = [random.sample(range(0, 255), 3) for i in range(200)]
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
    
    # show_seg_result_ply(ori_points, gt_ins, pred_instance_masks_label, scene_name,
    #     out_dir='./demo/data', filename='%s_insseg' % scene_name, palette=palette)

if __name__ == '__main__':
    main()




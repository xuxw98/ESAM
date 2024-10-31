from argparse import ArgumentParser

import os
from os import path as osp
import random
import numpy as np
import torch
import mmengine
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmengine.dataset import Compose, pseudo_collate
import open3d as o3d
from PIL import Image
from utils.vis_utils import vis_pointcloud,Vis_color

import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))

class DataConverter(object):
    def __init__(self, root_path, cfg):
        self.root_dir = root_path
        self.split_dir = osp.join(root_path)
        
        # Use dataset only for `parse_data_info`, pipeline is explicitly applied
        self.dataset = DATASETS.build(cfg.val_dataloader.dataset)
    
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
        if 'scannet' in self.root_dir:
            info['img_paths'] = [osp.join('2D', sample_idx, 'color', file.replace('bin','jpg')) for file in files]
            axis_align_matrix = self.get_axis_align_matrix(sample_idx)
            info['axis_align_matrix'] = axis_align_matrix.tolist()
        elif '3RScan' in self.root_dir:
            info['img_paths'] = [osp.join('3RScan', sample_idx, 'sequence','frame-' + file.split('.')[0].zfill(6) + '.color.jpg') for file in files]
        elif 'scenenn' in self.root_dir:
            info['img_paths'] = [osp.join('SceneNN', sample_idx, 'image','image'+file.split('.')[0].zfill(5)+'.png') for file in files]

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
    test_pipeline = Compose(cfg.test_pipeline)
    # build the data converter
    data_converter = DataConverter(root_path=cfg.data_root, cfg=cfg)
    # process the single scene data
    data = data_converter.process_single_scene(scene_idx)
    img_paths = data['img_paths']
    data = [test_pipeline(data)]
    collate_data = pseudo_collate(data)
    
    # forward the model
    with torch.no_grad():
        result = model.test_step(collate_data)
    
    return result[0], data[0], img_paths
        
def main():
    parser = ArgumentParser()
    parser.add_argument('--scene_idx', default='scene0011_00', type=str, help='single scene index')
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--use_vis', type=int, default="1")
    args = parser.parse_args()
    # model init
    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    model.map_to_rec_pcd = False
    # test a single scene
    result, data, img_paths = inference_detector(model=model, scene_idx=args.scene_idx)
    points = data['inputs']['points'][:,:,:3]
    pred_ins_mask = result.pred_pts_seg.pts_instance_mask[0]
    pred_ins_score = result.pred_pts_seg.instance_scores

    # Insseg results acquisition
    pred_instance_masks_sorted = torch.Tensor(pred_ins_mask[pred_ins_score.argsort()])
    pred_instance_masks_label = pred_instance_masks_sorted[0].long() - 1
    for i in range(1, pred_instance_masks_sorted.shape[0]):
        pred_instance_masks_label[pred_instance_masks_sorted[i].bool()] = i
        
    np.random.seed(0)
    palette = np.random.random((max(pred_instance_masks_label) + 2, 3)) * 255
    palette[-1] = 200
    
    pred_seg_color = palette[pred_instance_masks_label]
    points_color = pred_seg_color.reshape(points.shape[0], points.shape[1], 3)
    
    # load the scene images
    scene_images = []
    for img_path in img_paths:
        scene_images.append(np.array(Image.open(img_path)))
    scene_images = np.array(scene_images)
        
    # visualize the scene
    vis_p = vis_pointcloud(args.use_vis)
    vis_c = Vis_color(args.use_vis)
    
    for i in range(len(scene_images)):
        vis_p.update(points[i], points_color[i])
        vis_c.update(scene_images[i])
    vis_p.run()
    
    # if you want to save the camera parameters, you can use the following code
    # param = vis_p.vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('temp.json', param)
    # vis_p.vis.destroy_window()
    
if __name__ == '__main__':
    main()
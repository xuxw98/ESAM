from argparse import ArgumentParser
import os
import time
import numpy as np
import torch
import warnings
from mmdet3d.registry import MODELS
from mmengine.registry import init_default_scope
from mmengine.dataset import pseudo_collate
from mmdet3d.structures import Det3DDataSample, PointData
from mmengine.config import Config
from mmengine.runner import load_checkpoint

import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path))) 
from vis_demo.utils.vis_utils import vis_pointcloud,Vis_color
from vis_demo.utils.stream_data_utils import DataPreprocessor, StreamDataloader, StreamBotDataloader 

# deprecated
def init_model(config, checkpoint, device):
    config = Config.fromfile(config)
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    model =  MODELS.build(config.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        warnings.warn('Don\'t suggest using CPU device. '
                    'Some functions are not supported for now.')
    model.to(device)
    model.eval()
    return model

def inference_detector(model, args):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        args (argparse.Namespace): The arguments containing the data root etc.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    intrinsic = np.loadtxt(os.path.join(args.data_root, 'intrinsic.txt'))
    dataloader = StreamDataloader(args.data_root, interval=args.interval)
        
    # build the data preprocessor
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data', 'FastSAM-x.pt')
    data_preprocessor = DataPreprocessor(cfg, ckpt_path, intrinsic=intrinsic)
    all_images = []
    all_points = []
    # process the single scene data
    while True:
        frame_i, color_map, depth_map, pose, end_flag = dataloader.next()
        if end_flag: break
        group_ids, pts = data_preprocessor.process_single_frame(frame_i, color_map, depth_map, pose)
        points = torch.from_numpy(pts).float()
        sp_pts_mask = torch.from_numpy(group_ids).long().to(args.device)
        input_dict = {'points':points.to(args.device)}
        data_sample = Det3DDataSample()
        gt_pts_seg = PointData()
        gt_pts_seg['sp_pts_mask'] = sp_pts_mask
        data_sample.gt_pts_seg = gt_pts_seg
        data = [dict(inputs=input_dict, data_samples=data_sample)]
        collate_data = pseudo_collate(data)
        
        # forward the model
        with torch.no_grad():
            result = model.test_step(collate_data)
        all_images.append(color_map)
        all_points.append(points[:,:3])
    return result[0], all_images, all_points

class StreamDemo:
    def __init__(self, args):
        self.args = args
        self.model = self.init_model()
        self.model.map_to_rec_pcd = False
        np.random.seed(0)
        self.palette = np.random.random((256, 3)) * 255
        self.palette[-1] = 200
        self.palette = self.palette.astype(int)
        self.device = args.device
        self.online_vis = args.online_vis
        self.max_frames = args.max_frames
        self.intrinsic = np.loadtxt(os.path.join(args.data_root, 'intrinsic.txt'))
        self.dataloader = StreamDataloader(args.data_root, args.interval)
        
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data', 'FastSAM-x.pt')
        self.data_preprocessor = DataPreprocessor(self.model.cfg, ckpt_path, intrinsic=self.intrinsic)
        
        self.former_points = np.zeros((0, 3), dtype=np.float32)
    
    def init_model(self):
        args = self.args
        config = Config.fromfile(args.config)
        init_default_scope(config.get('default_scope', 'mmdet3d'))
        model =  MODELS.build(config.model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model.cfg = config
        if args.device != 'cpu':
            torch.cuda.set_device(args.device)
        else:
            warnings.warn('Don\'t suggest using CPU device. '
                        'Some functions are not supported for now.')
        model.to(args.device)
        model.eval()
        return model
    
    def vis(self, cur_points, cur_points_color, cur_image, end_flag=False):
        if not hasattr(self, 'vis_p'):
            self.vis_p = vis_pointcloud(self.args.use_vis)
            self.vis_c = Vis_color(self.args.use_vis)
        if end_flag:
            self.vis_p.run()
            # if you want to save the camera parameters, you can use the following code
            # if self.vis_p.use_vis:
            #     param = self.vis_p.vis.get_view_control().convert_to_pinhole_camera_parameters()
            #     o3d.io.write_pinhole_camera_parameters('temp.json', param)
            #     self.vis_p.vis.destroy_window()
        self.vis_p.update(cur_points, cur_points_color)
        self.vis_c.update(cur_image)
    
    def mask_to_color(self, pred_ins_mask, order=None):
        if order is None:
            idx_mask = np.where(np.any(pred_ins_mask, axis=0), np.argmax(pred_ins_mask, axis=0), -1)
        else:
            idx_mask = -np.ones(pred_ins_mask.shape[1], dtype=int)
            for i in order:
                idx_mask[np.where(pred_ins_mask[i])] = i
        points_color = self.palette[idx_mask]
        return points_color
    
    def run_single_frame(self, color_map, depth_map, pose, intrinsic=None):
        group_ids, pts = self.data_preprocessor.process_single_frame(color_map, depth_map, pose, intrinsic=intrinsic)
        points = torch.from_numpy(pts).float()
        sp_pts_mask = torch.from_numpy(group_ids).long().to(self.device)
        input_dict = {'points':points.to(self.device)}
        data_sample = Det3DDataSample()
        gt_pts_seg = PointData()
        gt_pts_seg['sp_pts_mask'] = sp_pts_mask
        data_sample.gt_pts_seg = gt_pts_seg
        data = [dict(inputs=input_dict, data_samples=data_sample)]
        collate_data = pseudo_collate(data)
        with torch.no_grad():
            result = self.model.test_step(collate_data)
        pred_ins_mask = result[0].pred_pts_seg.pts_instance_mask[0]
        pred_ins_score = result[0].pred_pts_seg.instance_scores
        order = pred_ins_score.argsort()
        points_color = self.mask_to_color(pred_ins_mask, order)
        all_points = np.concatenate([self.former_points, pts[:,:3]], axis=0)
        self.former_points = all_points
        return all_points, points_color, pred_ins_mask
    
    def run(self):
        all_images = []
        all_points = []
        all_points_color = []
        
        time0 = time.time()
        while True:
            frame_i, color_map, depth_map, pose, end_flag = self.dataloader.next()
            end_flag = end_flag or (frame_i >= self.args.max_frames)
            if end_flag:
                self.vis(None, None, None, True)
                break
            group_ids, pts = self.data_preprocessor.process_single_frame(color_map, depth_map, pose)
            points = torch.from_numpy(pts).float()
            sp_pts_mask = torch.from_numpy(group_ids).long().to(self.device)
            input_dict = {'points':points.to(self.device)}
            data_sample = Det3DDataSample()
            gt_pts_seg = PointData()
            gt_pts_seg['sp_pts_mask'] = sp_pts_mask
            data_sample.gt_pts_seg = gt_pts_seg
            data = [dict(inputs=input_dict, data_samples=data_sample)]
            collate_data = pseudo_collate(data)
            with torch.no_grad():
                result = self.model.test_step(collate_data)
            all_images.append(color_map)
            if self.online_vis:
                pred_ins_mask = result[0].pred_pts_seg.pts_instance_mask[0]
                pred_ins_score = result[0].pred_pts_seg.instance_scores
                order = pred_ins_score.argsort()
                points_color = self.mask_to_color(pred_ins_mask, order)
                whole_points = np.concatenate([self.former_points, pts[:,:3]], axis=0)
                self.former_points = whole_points
                
                all_points.append(whole_points)
                all_points_color.append(points_color)
                self.vis(whole_points, points_color, color_map)
            else:
                all_points.append(points[:,:3])
                
        total_time = time.time() - time0
        print(f"Total Time: {total_time:.2f}")
        print(f"Frame Number: {len(all_images)}")
        print(f"FPS: {(len(all_images) / total_time):.2f}")
        
        images = np.array(all_images)
        if not self.online_vis:
            points = torch.stack(all_points)
            pred_ins_mask = result[0].pred_pts_seg.pts_instance_mask[0]
            pred_ins_score = result[0].pred_pts_seg.instance_scores
            pred_ins_masks_sorted = pred_ins_mask[pred_ins_score.argsort()]
            points_color = self.mask_to_color(pred_ins_masks_sorted).reshape(points.shape[0], points.shape[1], 3)
            for i in range(len(all_images)):
                self.vis(points[i], points_color[i], images[i])   
        else:
            points = all_points[-1]
            points_color = all_points_color[-1]
        save_dir = os.path.join(self.args.save_dir, self.args.data_root.split('/')[-1])
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'images.npy'), images)
        np.save(os.path.join(save_dir, 'points.npy'), points)
        np.save(os.path.join(save_dir, 'points_color.npy'), points_color)   

        # point_cloud = np.concatenate([points, points_color], axis=-1)
        # np.savetxt(os.path.join(save_dir, 'point_cloud.npy'), point_cloud.reshape(-1, 6))
    
def main():
    parser = ArgumentParser(add_help=True)
    # args about input/output
    parser.add_argument('--data_root', type=str, default=None, help='Data root')
    parser.add_argument('--save_dir', type=str, default='./vis_demo/results', help='Output directory')
    parser.add_argument('--interval', type=int, default='1', help='Frame processing interval (process every Nth frame)')
    parser.add_argument('--max_frames', type=int, default=10000, help='Max frame number to process')
    # args about model
    parser.add_argument('--config', type=str, default='configs/ESAM-E_CA/ESAM-E_online_stream.py', help='Config file')
    parser.add_argument('--checkpoint', type=str, default='work_dirs/ESAM-E_online_scannet200_CA/epoch_128.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # args about visualization
    parser.add_argument('--use_vis', type=int, default="1", help="Whether to enable visualization, set to 1 to enable")
    parser.add_argument('--online_vis', action='store_true', help="Whether to visualize segmentation results online, store true")
    args = parser.parse_args()
    
    assert args.data_root is not None, "The input data root must be specified"
      
    demo = StreamDemo(args)
    demo.run()

if __name__ == '__main__':
    main()

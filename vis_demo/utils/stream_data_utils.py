import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from fastsam import FastSAM
from tqdm import tqdm
current_dir = os.getcwd()

class DataPreprocessor:
    def __init__(self, cfg, ckpt_path, unify_dim=(640, 480), intrinsic=None):
        self.cfg = cfg
        self.mask_generator = FastSAM(ckpt_path)
        self.unify_dim = unify_dim
        self.intrinsic = intrinsic
        
        self.color_mean = cfg['color_mean']
        self.color_std = cfg['color_std']
    
    def process_single_frame(self, color_map, depth_map, pose, intrinsic=None):
        """
        Process data of single frame, including rgb, depth, pose, intrinsic.
        Args:
            color_map: rgb of this frame, np.ndarray, [H, W, 3]
            depth_map: depth of this frame, np.ndarray, [H, W]
            pose: camera extrinsic(c2w), np.ndarray, [4, 4]
            intrinsic: camera intrinsic, np.ndarray, [3, 3]
        """
        if intrinsic is None:
            intrinsic = self.intrinsic
        assert intrinsic is not None, "Intrinsic is not fixed, please input camera intrinsic of this frame."
        
        bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
        everything_result = self.mask_generator(bgr, device='cuda', retina_masks=True, imgsz=self.unify_dim[0], conf=0.1, iou=0.9)
        try:
            masks = format_result(everything_result[0])
        except:
            everything_result = self.mask_generator(bgr, device='cuda', retina_masks=True, imgsz=self.unify_dim[0], conf=0.1, iou=0.7,)
            masks = format_result(everything_result[0])
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        group_ids = np.full((color_map.shape[0], color_map.shape[1]), -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0
        for i in range(num_masks):
            mask_now = masks[i]["segmentation"]
            group_ids[mask_now] = group_counter
            group_counter += 1
        # convert depth map to point cloud
        height, width = depth_map.shape    
        w_ind = np.arange(width)
        h_ind = np.arange(height)

        ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
        ww_ind = ww_ind.reshape(-1)
        hh_ind = hh_ind.reshape(-1)
        depth_map = depth_map.reshape(-1)
        group_ids = group_ids.reshape(-1)
        color_map = color_map.reshape(-1, 3)
        
        valid = np.where(depth_map > 0.1)[0]
        ww_ind = ww_ind[valid]
        hh_ind = hh_ind[valid]
        depth_map = depth_map[valid]
        group_ids = group_ids[valid]
        rgb = color_map[valid]
        
        # For MV: downsample to 20000
        unaligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, intrinsic, pose)
        if np.isnan(unaligned_xyz).any(): 
            return None
        unaligned_xyz = np.concatenate([unaligned_xyz, rgb], axis=-1)
        xyz_all = np.concatenate([unaligned_xyz, group_ids.reshape(-1,1)], axis=-1)
        xyz_all = random_sampling(xyz_all, 20000)
        unaligned_xyz, group_ids = xyz_all[:, :6], xyz_all[:, 6]
        
        # Get superpoints
        points_without_seg = unaligned_xyz[group_ids == -1]
        if len(points_without_seg) < 20:
            other_ins = np.zeros(len(points_without_seg), dtype=np.int64) + group_ids.max() + 1
        else:
            other_ins = KMeans(n_clusters=20, n_init=10).fit(points_without_seg).labels_ + group_ids.max() + 1
        group_ids[group_ids == -1] = other_ins
        unique_ids = np.unique(group_ids)
        if group_ids.max() != len(unique_ids) - 1:
            new_group_ids = np.zeros_like(group_ids)
            for i, ids in enumerate(unique_ids):
                new_group_ids[group_ids == ids] = i
            group_ids = new_group_ids
            
        # Color Normalization
        unaligned_xyz[:, 3:] = (unaligned_xyz[:, 3:] - self.color_mean) / self.color_std
        
        return group_ids, unaligned_xyz

class StreamDataloader:
    def __init__(self, data_root, interval=1):
        self.data_root = data_root
        self.counter = 0
        self.interval = interval
        self.color_paths = sorted(os.listdir(data_root+'color'), key=lambda f:int(os.path.splitext(f)[0]))
        self.img_nums = len(self.color_paths)
    
    def _get_item(self, i):
        rgb_path = os.path.join(self.data_root, 'color', self.color_paths[i])
        depth_path = rgb_path.replace('color', 'depth').replace('jpg', 'png')
        pose_path = rgb_path.replace('color', 'pose').replace('jpg', 'txt')
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(depth_path, -1)
        pose = np.loadtxt(pose_path)
        return i, rgb_img, depth_img, pose

    def next(self):
        if self.counter >= self.img_nums:
            return None, None, None, None, True
        i, rgb_img, depth_img, pose = self._get_item(self.counter)
        self.counter += self.interval
        return i, rgb_img, depth_img, pose, False

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_item(index)
        else:
            raise TypeError("Index must be int")

class StreamBotDataloader:
    def __init__(self, data_root, interval=1):
        self.data_root = data_root
        self.counter = 0
        self.interval = interval
        self.longest_side = 640
        self.init_data_paths()
    def init_data_paths(self):
        all_files = os.listdir(self.data_root)
        self.color_paths = sorted([f for f in all_files if f.endswith('_rgb.png')],
                                    key=lambda f: int(f.split('_')[0]))
        self.depth_paths = sorted([f for f in all_files if f.endswith('_depth.npy')],
                                    key=lambda f: int(f.split('_')[0]))
        self.pose_paths = sorted([f for f in all_files if f.endswith('_camera_pose.npz')],
                                    key=lambda f: int(f.split('_')[0]))
        self.orig_img_shape = cv2.imread(os.path.join(self.data_root, self.color_paths[0])).shape[:2]
        self.resize_ratio = self.longest_side / max(self.orig_img_shape)
        self.intrinsic = self.build_intrinsic()
        self.img_nums = len(self.color_paths)
        
    def build_intrinsic(self, index=0):
        intrinsic_path = os.path.join(self.data_root, self.pose_paths[index])
        intrinsic_params = np.load(intrinsic_path)['depth_intrinsics.npy']
        if self.resize_ratio != 1:
            intrinsic_params = intrinsic_params * self.resize_ratio
        intrinsic = np.array([[intrinsic_params[0], 0, intrinsic_params[2]],
                                [0, intrinsic_params[1], intrinsic_params[3]],
                                [0, 0, 1]])
        return intrinsic
    def _get_item(self, i):
        rgb_path = os.path.join(self.data_root, self.color_paths[i])
        depth_path = os.path.join(self.data_root, self.depth_paths[i])
        pose_path = os.path.join(self.data_root, self.pose_paths[i])
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = np.load(depth_path)
        if self.resize_ratio != 1:
            rgb_img = cv2.resize(rgb_img, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.resize(depth_img, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_NEAREST)
        
        pos = np.load(pose_path)['position.npy']
        rot = np.load(pose_path)['rotation.npy']
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        return i, rgb_img, depth_img, pose

    def next(self):
        if self.counter >= self.img_nums:
            return None, None, None, None, True
        i, rgb_img, depth_img, pose = self._get_item(self.counter)
        self.counter += self.interval
        return i, rgb_img, depth_img, pose, False

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_item(index)
        else:
            raise TypeError("Index must be int")
    
def format_result(result):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices] 

def convert_from_uvd(u, v, depth, intr, pose):
    # u is width index, v is height index
    depth_scale = 1000
    z = depth / depth_scale

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    
    uv = np.concatenate([u,v,padding], axis=0)
    xyz = (np.linalg.inv(intr[:3,:3]) @ uv) * np.expand_dims(z,axis=0)
    xyz = np.concatenate([xyz,padding], axis=0)
    xyz = pose @ xyz
    xyz[:3,:] /= xyz[3,:] 
    return xyz[:3, :].T

if __name__ == "__main__":
    intrinsic = np.array([[577.870605,0,319.5],[0,577.870605,239.5],[0,0,1]])
    data_preprocessor = DataPreprocessor('/home/xxw/3D/ESAM/data/FastSAM-x.pt', intrinsic=intrinsic)
    dataset_path = '/home/xxw/3D/ESAM/data/scannet200-mv_fast/2D/scene0000_00'
    frame_i = 0
    rgb_path = os.path.join(dataset_path, 'color', f'{frame_i}.jpg')
    depth_path = os.path.join(dataset_path, 'depth', f'{frame_i}.png')
    pose_path = os.path.join(dataset_path, 'pose', f'{frame_i}.txt')
    
    bgr = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, -1)
    pose = np.loadtxt(pose_path)
    group_ids, unaligned_xyz = data_preprocessor.process_single_frame(frame_i, rgb, depth, pose)
    
    import time
    start_time = time.time()
    group_ids, unaligned_xyz = data_preprocessor.process_single_frame(frame_i, rgb, depth, pose)
    print('Time cost per frame: ', time.time() - start_time)
# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np
import pdb


class ScanNetData(object):
    """ScanNet data.
    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        scannet200 (bool): True for ScanNet200, else for ScanNet.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, split='train', scannet200=False, save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.scannet200 = scannet200
        if self.scannet200:
            self.classes = [
                'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
                'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
                'toilet', 'bookshelf', 'monitor', 'curtain', 'book',
                'armchair', 'coffee table', 'box', 'refrigerator', 'lamp',
                'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand',
                'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling',
                'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
                'backpack', 'toilet paper', 'printer', 'tv stand',
                'whiteboard', 'blanket', 'shower curtain', 'trash can',
                'closet', 'stairs', 'microwave', 'stove', 'shoe',
                'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
                'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
                'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
                'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
                'blackboard', 'piano', 'suitcase', 'rail', 'radiator',
                'recycling bin', 'container', 'wardrobe', 'soap dispenser',
                'telephone', 'bucket', 'clock', 'stand', 'light',
                'laundry basket', 'pipe', 'clothes dryer', 'guitar',
                'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
                'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
                'storage bin', 'coffee maker', 'dishwasher',
                'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
                'toaster', 'bulletin board', 'ironing board', 'fireplace',
                'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher',
                'ball', 'hat', 'shower curtain rod', 'water cooler',
                'paper cutter', 'tray', 'shower door', 'pillar', 'ledge',
                'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale',
                'tissue box', 'light switch', 'crate', 'power outlet',
                'decoration', 'sign', 'projector', 'closet door',
                'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
                'headphones', 'dish rack', 'broom', 'guitar case',
                'range hood', 'dustpan', 'hair dryer', 'water bottle',
                'handicap bar', 'purse', 'vent', 'shower floor',
                'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
                'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object',
                'bathroom vanity', 'closet wall', 'laundry hamper',
                'bathroom stall door', 'ceiling light', 'trash bin',
                'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
                'cd case', 'closet rod', 'coffee kettle', 'structure',
                'shower head', 'keyboard piano', 'case of water bottles',
                'coat rack', 'storage organizer', 'folded chair', 'fire alarm',
                'power strip', 'calendar', 'poster', 'potted plant', 'luggage',
                'mattress'
            ]
            self.cat_ids = np.array([
                2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21,
                22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58,
                59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116,
                118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138,
                139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163,
                165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
                202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261,
                264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356,
                370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581,
                609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
                1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180,
                1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190,
                1191
            ])
        else:
            self.classes = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin'
            ]
            self.cat_ids = np.array([
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
            ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_{split}.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmengine.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data',
                            f'{idx}_unaligned_bbox.npy')
        mmengine.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'scannet_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmengine.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_images(self, idx):
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        return paths

    def get_extrinsics(self, idx):
        extrinsics = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.txt') and not file == 'intrinsic.txt':
                extrinsics.append(np.loadtxt(osp.join(path, file)))
        return extrinsics

    def get_intrinsics(self, idx):
        matrix_file = osp.join(self.root_dir, 'posed_images', idx,
                               'intrinsic.txt')
        mmengine.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'scannet_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))
            points.tofile(
                osp.join(self.save_path, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            sp_filename = osp.join(self.root_dir, 'scannet_instance_data',
                                    f'{sample_idx}_sp_label.npy')
            super_points = np.load(sp_filename)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))
            super_points.tofile(
                osp.join(self.save_path, 'super_points', f'{sample_idx}.bin'))
            info['super_pts_path'] = osp.join('super_points', f'{sample_idx}.bin')

            # update with RGB image paths if exist
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                all_extrinsics = self.get_extrinsics(sample_idx)
                all_img_paths = self.get_images(sample_idx)
                # some poses in ScanNet are invalid
                extrinsics, img_paths = [], []
                for extrinsic, img_path in zip(all_extrinsics, all_img_paths):
                    if np.all(np.isfinite(extrinsic)):
                        img_paths.append(img_path)
                        extrinsics.append(extrinsic)
                info['extrinsics'] = extrinsics
                info['img_paths'] = img_paths

            if not self.test_mode:
                pts_instance_mask_path = osp.join(
                    self.root_dir, 'scannet_instance_data',
                    f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(
                    self.root_dir, 'scannet_instance_data',
                    f'{sample_idx}_sem_label.npy')

                pts_instance_mask = np.load(pts_instance_mask_path).astype(
                    np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                    np.int64)

                mmengine.mkdir_or_exist(
                    osp.join(self.save_path, 'instance_mask'))
                mmengine.mkdir_or_exist(
                    osp.join(self.save_path, 'semantic_mask'))

                pts_instance_mask.tofile(
                    osp.join(self.save_path, 'instance_mask',
                             f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(
                    osp.join(self.save_path, 'semantic_mask',
                             f'{sample_idx}.bin'))

                info['pts_instance_mask_path'] = osp.join(
                    'instance_mask', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join(
                    'semantic_mask', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 6
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


class ScanNetSVData(object):
    """ScanNet data.
    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        scannet200 (bool): True for ScanNet200, else for ScanNet.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, split='train', scannet200=False, save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.scannet200 = scannet200
        if self.scannet200:
            self.classes = [
                'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
                'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
                'toilet', 'bookshelf', 'monitor', 'curtain', 'book',
                'armchair', 'coffee table', 'box', 'refrigerator', 'lamp',
                'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand',
                'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling',
                'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
                'backpack', 'toilet paper', 'printer', 'tv stand',
                'whiteboard', 'blanket', 'shower curtain', 'trash can',
                'closet', 'stairs', 'microwave', 'stove', 'shoe',
                'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
                'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
                'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
                'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
                'blackboard', 'piano', 'suitcase', 'rail', 'radiator',
                'recycling bin', 'container', 'wardrobe', 'soap dispenser',
                'telephone', 'bucket', 'clock', 'stand', 'light',
                'laundry basket', 'pipe', 'clothes dryer', 'guitar',
                'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
                'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
                'storage bin', 'coffee maker', 'dishwasher',
                'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
                'toaster', 'bulletin board', 'ironing board', 'fireplace',
                'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher',
                'ball', 'hat', 'shower curtain rod', 'water cooler',
                'paper cutter', 'tray', 'shower door', 'pillar', 'ledge',
                'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale',
                'tissue box', 'light switch', 'crate', 'power outlet',
                'decoration', 'sign', 'projector', 'closet door',
                'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
                'headphones', 'dish rack', 'broom', 'guitar case',
                'range hood', 'dustpan', 'hair dryer', 'water bottle',
                'handicap bar', 'purse', 'vent', 'shower floor',
                'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
                'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object',
                'bathroom vanity', 'closet wall', 'laundry hamper',
                'bathroom stall door', 'ceiling light', 'trash bin',
                'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
                'cd case', 'closet rod', 'coffee kettle', 'structure',
                'shower head', 'keyboard piano', 'case of water bottles',
                'coat rack', 'storage organizer', 'folded chair', 'fire alarm',
                'power strip', 'calendar', 'poster', 'potted plant', 'luggage',
                'mattress'
            ]
            self.cat_ids = np.array([
                2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21,
                22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58,
                59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116,
                118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138,
                139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163,
                165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
                202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261,
                264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356,
                370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581,
                609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
                1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180,
                1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190,
                1191
            ])
        else:
            self.classes = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin'
            ]
            self.cat_ids = np.array([
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
            ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_sv_{split}.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'scannet_sv_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmengine.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'scannet_sv_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename).astype(np.float32)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))
            points.tofile(
                osp.join(self.save_path, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            sp_filename = osp.join(self.root_dir, 'scannet_sv_instance_data',
                                    f'{sample_idx}_sp_label.npy')
            super_points = np.load(sp_filename).astype(np.int64)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))
            super_points.tofile(
                osp.join(self.save_path, 'super_points', f'{sample_idx}.bin'))
            info['super_pts_path'] = osp.join('super_points', f'{sample_idx}.bin')

            prefix = sample_idx[:12]
            img_idx = sample_idx[13:]
            info['img_path'] = osp.join(self.root_dir, '2D', prefix, 'color', f'{img_idx}.jpg')
            try:
                info['pose'] = np.load(os.path.join(self.root_dir, 'pose_centered', prefix, f'{img_idx}.npy'))
            except:
                pass 
            if not self.test_mode:
                pts_instance_mask_path = osp.join(
                    self.root_dir, 'scannet_sv_instance_data',
                    f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(
                    self.root_dir, 'scannet_sv_instance_data',
                    f'{sample_idx}_sem_label.npy')

                pts_instance_mask = np.load(pts_instance_mask_path).astype(
                    np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                    np.int64)

                mmengine.mkdir_or_exist(
                    osp.join(self.save_path, 'instance_mask'))
                mmengine.mkdir_or_exist(
                    osp.join(self.save_path, 'semantic_mask'))

                pts_instance_mask.tofile(
                    osp.join(self.save_path, 'instance_mask',
                             f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(
                    osp.join(self.save_path, 'semantic_mask',
                             f'{sample_idx}.bin'))

                info['pts_instance_mask_path'] = osp.join(
                    'instance_mask', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join(
                    'semantic_mask', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


class ScanNetMVData(object):
    """ScanNet data.
    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        scannet200 (bool): True for ScanNet200, else for ScanNet.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, split='train', scannet200=False, save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.scannet200 = scannet200
        if self.scannet200:
            self.classes = [
                'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
                'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
                'toilet', 'bookshelf', 'monitor', 'curtain', 'book',
                'armchair', 'coffee table', 'box', 'refrigerator', 'lamp',
                'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand',
                'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling',
                'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
                'backpack', 'toilet paper', 'printer', 'tv stand',
                'whiteboard', 'blanket', 'shower curtain', 'trash can',
                'closet', 'stairs', 'microwave', 'stove', 'shoe',
                'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
                'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
                'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
                'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
                'blackboard', 'piano', 'suitcase', 'rail', 'radiator',
                'recycling bin', 'container', 'wardrobe', 'soap dispenser',
                'telephone', 'bucket', 'clock', 'stand', 'light',
                'laundry basket', 'pipe', 'clothes dryer', 'guitar',
                'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
                'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
                'storage bin', 'coffee maker', 'dishwasher',
                'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
                'toaster', 'bulletin board', 'ironing board', 'fireplace',
                'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher',
                'ball', 'hat', 'shower curtain rod', 'water cooler',
                'paper cutter', 'tray', 'shower door', 'pillar', 'ledge',
                'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale',
                'tissue box', 'light switch', 'crate', 'power outlet',
                'decoration', 'sign', 'projector', 'closet door',
                'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
                'headphones', 'dish rack', 'broom', 'guitar case',
                'range hood', 'dustpan', 'hair dryer', 'water bottle',
                'handicap bar', 'purse', 'vent', 'shower floor',
                'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
                'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object',
                'bathroom vanity', 'closet wall', 'laundry hamper',
                'bathroom stall door', 'ceiling light', 'trash bin',
                'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
                'cd case', 'closet rod', 'coffee kettle', 'structure',
                'shower head', 'keyboard piano', 'case of water bottles',
                'coat rack', 'storage organizer', 'folded chair', 'fire alarm',
                'power strip', 'calendar', 'poster', 'potted plant', 'luggage',
                'mattress'
            ]
            self.cat_ids = np.array([
                2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21,
                22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58,
                59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116,
                118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138,
                139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163,
                165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
                202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261,
                264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356,
                370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581,
                609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
                1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180,
                1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190,
                1191
            ])
        else:
            self.classes = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin'
            ]
            self.cat_ids = np.array([
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
            ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_{split}.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'axis_align_matrix',
                               f'{idx}.npy')
        mmengine.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            files = os.listdir(osp.join(self.root_dir, 'points', sample_idx))
            files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
            # interval = 40
            files = [file for file in files if int(file.split('.')[0]) % 40 == 0]
            
            info['pts_paths'] = [osp.join('points', sample_idx, file) for file in files]
            info['super_pts_paths'] = [osp.join('super_points', sample_idx, file) for file in files]
            info['img_paths'] = [osp.join(self.root_dir, '2D', sample_idx, 'color', file.replace('bin','jpg')) for file in files]
            info['poses'] = [np.loadtxt(osp.join(self.root_dir, '2D', sample_idx, 'pose', file.replace('bin','txt'))) for file in files]

            if not self.test_mode:
                info['pts_instance_mask_paths'] = [osp.join('instance_mask', sample_idx, file) for file in files]
                info['pts_semantic_mask_paths'] = [osp.join('semantic_mask', sample_idx, file) for file in files]

            if has_label:
                annotations = {}
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info
        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
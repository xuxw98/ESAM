from os import path as osp
import numpy as np
import random

from mmdet3d.datasets.scannet_dataset import ScanNetSegDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ScanNetSegDataset_(ScanNetSegDataset):
    """We just add super_pts_path."""

    def get_scene_idxs(self, *args, **kwargs):
        """Compute scene_idxs for data sampling."""
        return np.arange(len(self)).astype(np.int32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_path'] = osp.join(
            self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        return info

@DATASETS.register_module()
class ScanNet200SegDataset_(ScanNetSegDataset_):
    # IMPORTANT: the floor and chair categories are swapped.
    METAINFO = {
    'classes': ('wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet',
                'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
                'picture', 'window', 'toilet', 'bookshelf', 'monitor',
                'curtain', 'book', 'armchair', 'coffee table', 'box',
                'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes',
                'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
                'plant', 'ceiling', 'bathtub', 'end table', 'dining table',
                'keyboard', 'bag', 'backpack', 'toilet paper', 'printer',
                'tv stand', 'whiteboard', 'blanket', 'shower curtain',
                'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe',
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
                'mattress'),
    # the valid ids of segmentation annotations
    'seg_valid_class_ids': (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
        23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86,
        87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
        106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131,
        132, 134, 136, 138, 139, 140, 141, 145, 148, 154,155, 156, 157, 159,
        161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
        202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276,
        283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399,
        408, 417, 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163,
        1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175,
        1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
        1189, 1190, 1191),
    'seg_all_class_ids': tuple(range(1, 1358)),
    'palette': [random.sample(range(0, 255), 3) for i in range(200)]}

@DATASETS.register_module()
class ScanNet200SegMVDataset_(ScanNetSegDataset_):
    # IMPORTANT: the floor and chair categories are swapped.
    METAINFO = {
    'classes': ('wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet',
                'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
                'picture', 'window', 'toilet', 'bookshelf', 'monitor',
                'curtain', 'book', 'armchair', 'coffee table', 'box',
                'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes',
                'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
                'plant', 'ceiling', 'bathtub', 'end table', 'dining table',
                'keyboard', 'bag', 'backpack', 'toilet paper', 'printer',
                'tv stand', 'whiteboard', 'blanket', 'shower curtain',
                'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe',
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
                'mattress'),
    # the valid ids of segmentation annotations
    'seg_valid_class_ids': (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
        23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86,
        87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
        106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131,
        132, 134, 136, 138, 139, 140, 141, 145, 148, 154,155, 156, 157, 159,
        161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
        202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276,
        283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399,
        408, 417, 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163,
        1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175,
        1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
        1189, 1190, 1191),
    'seg_all_class_ids': tuple(range(1, 1358)),
    'palette': [random.sample(range(0, 255), 3) for i in range(200)]}

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_paths'] = [osp.join(self.data_root, path) for path in info['super_pts_paths']]
        info['pts_paths'] = [osp.join(self.data_root, path) for path in info['pts_paths']]
        info['pts_instance_mask_paths'] = [osp.join(self.data_root, path) for path in info['pts_instance_mask_paths']]
        info['pts_semantic_mask_paths'] = [osp.join(self.data_root, path) for path in info['pts_semantic_mask_paths']]
        info['img_paths'] = [osp.join(self.data_root, path) for path in info['img_paths']]

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    img_info['img_path'] = osp.join(
                        self.data_prefix.get('img', ''), img_info['img_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.seg_label_mapping

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()

        return info
    
@DATASETS.register_module()
class ScanNetSegMVDataset_(ScanNetSegDataset_):
    METAINFO = {
    'classes': ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
        'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
        'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
        'bathtub', 'otherfurniture'),
    # the valid ids of segmentation annotations
    'seg_valid_class_ids': (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),
    'seg_all_class_ids': tuple(range(41)),
    'palette': [random.sample(range(0, 255), 3) for i in range(200)]}
    
    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_paths'] = [osp.join(self.data_root, path) for path in info['super_pts_paths']]
        info['pts_paths'] = [osp.join(self.data_root, path) for path in info['pts_paths']]
        info['pts_instance_mask_paths'] = [osp.join(self.data_root, path) for path in info['pts_instance_mask_paths']]
        info['pts_semantic_mask_paths'] = [osp.join(self.data_root, path) for path in info['pts_semantic_mask_paths']]
        info['img_paths'] = [osp.join(self.data_root, path) for path in info['img_paths']]

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    img_info['img_path'] = osp.join(
                        self.data_prefix.get('img', ''), img_info['img_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.seg_label_mapping

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()

        return info
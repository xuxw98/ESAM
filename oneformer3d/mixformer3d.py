import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean, scatter
import MinkowskiEngine as ME
import pointops
import pdb, time
from functools import partial
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type, BaseInstance3DBoxes, DepthInstance3DBoxes
from mmengine.structures import InstanceData
from .mask_matrix_nms import mask_matrix_nms
from .oneformer3d import ScanNetOneFormer3DMixin
from .instance_merge import ins_merge_mat, ins_cat, ins_merge, OnlineMerge, GTMerge
import numpy as np
from scipy.optimize import linear_sum_assignment
from .img_backbone import point_sample, bbox3d2result, read_label_mapping
import os

@MODELS.register_module()
class ScanNet200MixFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse())
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        # x = scatter_mean(x, sp_idx, dim=0)  # todo: do we need dim?
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        ## Backbone
        x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## GT-prepare
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        gt_point_instances = []
        for i in range(len(gt_instances)):
            ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            if torch.sum(ins == -1) != 0:
                ins[ins == -1] = torch.max(ins) + 1
                ins = F.one_hot(ins)[:, :-1]
            else:
                ins = F.one_hot(ins)
            ins = ins.bool().T
            gt_point = InstanceData()
            gt_point.sp_masks = ins
            gt_point_instances.append(gt_point)
        queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz)
        ## Decoder
        super_points = [bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples]
        x = self.decoder(x, point_features, queries, super_points, all_xyz_w)
        ## Loss
        return self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        ## Backbone
        x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## Decoder
        super_points = [bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples]
        x = self.decoder(x, point_features, x, super_points, all_xyz_w)
        ## Post-processing
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

@MODELS.register_module()
class ScanNet200MixFormer3D_FF(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(img_backbone)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(960, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
    
    def init_weights(self):
        if hasattr(self, 'img_backbone'):
            self.img_backbone.init_weights()
    
    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # extract image features
        with torch.no_grad():
            img_features = self.img_backbone(batch_inputs_dict['img_path'])
        
        img_metas = []
        for batch_idx in range(len(batch_data_samples)):
            img_meta = {}
            img_meta['depth2img'] = batch_data_samples[batch_idx].depth2img
            img_meta['sample_idx'] = batch_data_samples[batch_idx].sample_idx
            img_meta['img_shape'] = batch_data_samples[batch_idx].img_shape
            img_meta['num_pts_feats'] = batch_data_samples[batch_idx].num_pts_feats
            img_meta['axis_align_matrix'] = batch_data_samples[batch_idx].axis_align_matrix
            
            try:
                img_meta['pcd_rotation'] = batch_data_samples[batch_idx].pcd_rotation
                img_meta['pcd_scale_factor'] = batch_data_samples[batch_idx].pcd_scale_factor
                img_meta['pcd_trans'] = batch_data_samples[batch_idx].pcd_trans
                img_meta['pcd_rotation_angle'] = batch_data_samples[batch_idx].pcd_rotation_angle
                img_meta['pcd_vertical_flip'] = batch_data_samples[batch_idx].pcd_vertical_flip
                img_meta['pcd_horizontal_flip'] = batch_data_samples[batch_idx].pcd_horizontal_flip
                img_meta['transformation_3d_flow'] = batch_data_samples[batch_idx].transformation_3d_flow
            except:
                pass
            img_metas.append(img_meta)
        
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=batch_data_samples[0].img_shape))
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        # x = scatter_mean(x, sp_idx, dim=0)  # todo: do we need dim?
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        ## Backbone
        x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## GT-prepare
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        gt_point_instances = []
        for i in range(len(gt_instances)):
            ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            if torch.sum(ins == -1) != 0:
                ins[ins == -1] = torch.max(ins) + 1
                ins = F.one_hot(ins)[:, :-1]
            else:
                ins = F.one_hot(ins)
            ins = ins.bool().T
            gt_point = InstanceData()
            gt_point.sp_masks = ins
            gt_point_instances.append(gt_point)
        queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz)
        ## Decoder
        super_points = [bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples]
        x = self.decoder(x, point_features, queries, super_points, all_xyz_w)
        ## Loss
        return self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        ## Backbone
        x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## Decoder
        super_points = [bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples]
        x = self.decoder(x, point_features, x, super_points, all_xyz_w)
        ## Post-processing
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

@MODELS.register_module()
class ScanNet200MixFormer3D_Online(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 weights=[0.33,0.33,0.33],
                 thresh=0.55,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.weights = weights
        self.thresh = thresh
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()
    
    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()

    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i):
        """Extract features from sparse tensor.
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            # pdb.set_trace()
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]  # [B, N, 3+D]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        # x = scatter_mean(x, sp_idx, dim=0)  # todo: do we need dim?
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list
    
    def _select_queries(self, x, gt_instances, sp_xyz, frame_i):
        """Select queries for train pass.
        """
        gt_instances_ = []
        for i in range(len(x)):
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(x[i].device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(x[i].device)
            bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(x[i].device)
            temp.bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(x[i].device)])
            gt_instances_.append(temp)

        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                n = max(1,n)
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks[:, ids]
                sp_xyz[i] = sp_xyz[i][ids]
            else:
                queries.append(x[i])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks
        
        return queries, gt_instances_, sp_xyz

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.
        """
        losses, merge_feat_n_frames, ins_masks_query_n_frames = {}, [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        for frame_i in range(num_frames):
            ## Backbone
            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## GT-prepare
            gt_instances = [s.gt_instances_3d for s in batch_data_samples]
            gt_point_instances, ins_masks_query_batch = [], []
            for i in range(len(gt_instances)):
                ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i]
                if torch.sum(ins == -1) != 0:
                    # Use global instance number for each frame
                    ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    ins = F.one_hot(ins)[:, :-1]
                else:
                    ins = F.one_hot(ins)
                    max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    if ins.shape[1] < max_ids:
                        zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                        ins = torch.cat([ins, zero_pad], dim=-1)
                ins = ins.bool().T
                gt_point = InstanceData()
                gt_point.sp_masks = ins
                gt_point_instances.append(gt_point)
            queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz, frame_i)
            ## Decoder
            super_points = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
            if self.decoder_online and frame_i > 0:
                x = self.decoder(x, point_features, queries, super_points, all_xyz_w, prev_queries)
            else:
                x = self.decoder(x, point_features, queries, super_points, all_xyz_w)
            ## Query projector
            for i in range(len(gt_instances)):
                ins_masks_query = gt_instances[i].query_masks[:-self.sem_len, :]
                ins_masks_query = [ins_masks_query[i].nonzero().flatten()
                        for i in range(ins_masks_query.shape[0])]
                ins_masks_query_batch.append(ins_masks_query)
            if hasattr(self, 'merge_head'):
                merge_feat = self.merge_head(x['queries'])
                merge_feat_n_frames.append(merge_feat)
                ins_masks_query_n_frames.append(ins_masks_query_batch)
            ## Query merging
            if self.decoder_online:
                if frame_i == 0: gt_merger = GTMerge()
                prev_queries = gt_merger.merge(x['queries'], ins_masks_query_batch)
                if frame_i == num_frames - 1: gt_merger.clean()
            ## Loss
            loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)
            for key, value in loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        
        ## Query contrast
        if hasattr(self, 'merge_criterion'):
            merge_feat_n_frames = [[frame[i] for frame in merge_feat_n_frames]
                 for i in range(len(merge_feat_n_frames[0]))]
            ins_masks_query_n_frames = [[frame[i] for frame in ins_masks_query_n_frames]
                 for i in range(len(ins_masks_query_n_frames[0]))]
            loss = self.merge_criterion(merge_feat_n_frames, ins_masks_query_n_frames)
            losses.update(loss)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        begin_frame = 0
        if hasattr(self, 'memory'):
            self.memory.reset()
        
        for frame_i in range(begin_frame, num_frames):
            ## Backbone
            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## Decoder
            super_points = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
            if self.decoder_online and frame_i > 0:
                x = self.decoder(x, point_features, x, super_points, all_xyz_w, [mv_queries])
            else:
                x = self.decoder(x, point_features, x, super_points, all_xyz_w)
            ## Post-processing
            pred_pts_seg, mapping = self.predict_by_feat(
                x, batch_data_samples[0].gt_pts_seg.sp_pts_mask[frame_i])
            
            results.append(pred_pts_seg[0])
            ## Query projector, semantic and geometric information
            if hasattr(self, 'merge_head'):
                query_feats = self.merge_head(x['queries'][0])
                query_feats_list.append([query_feats[mapping[0]], query_feats[mapping[1]]])
                sem_preds = x['sem_preds'][0]
                sem_preds_list.append([sem_preds[mapping[0]], sem_preds[mapping[1]]])
                sp_xyz_list.append([sp_xyz[0][mapping[0]], sp_xyz[0][mapping[1]]])
                if self.use_bbox:
                    bbox_preds = x['bboxes'][0] # [N, 6]
                    bboxes_list.append([bbox_preds[mapping[0]], bbox_preds[mapping[1]]])            
            
            ## Online merging
            if self.test_cfg.merge_type == 'learnable_online':
                if frame_i == begin_frame:
                    online_merger = OnlineMerge(self.test_cfg.inscat_topk_insts, self.use_bbox, self.weights, self.thresh, use_inst_label = True)
                mv_mask, mv_labels, mv_scores, mv_queries, mv_bboxes = online_merger.merge(
                    results[-1]['pts_instance_mask'][0],
                    results[-1]['instance_labels'][0],
                    results[-1]['instance_scores'][0],
                    results[-1]['instance_queries'][0],
                    query_feats_list[-1][0],
                    sem_preds_list[-1][0],
                    sp_xyz_list[-1][0],
                    bboxes_list[-1][0] if self.use_bbox else None)           
                
                # Empty cache. Only offline merging requires the whole list.
                query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list = [], [], [], []
                if frame_i == num_frames - 1:
                    online_merger.clean() # Ignore panoptic segmentation
        
        ## Offline merging
        if self.test_cfg.merge_type == 'learnable':
            mv_mask, mv_labels, mv_scores = ins_merge_mat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                [res[0] for res in query_feats_list],
                [res[0] for res in sem_preds_list],
                [res[0] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge_mat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                [res[1] for res in query_feats_list],
                [res[1] for res in sem_preds_list],
                [res[1] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'concat':
            mv_mask, mv_labels, mv_scores = ins_cat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_cat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'geometric':
            mv_mask, mv_labels, mv_scores = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'learnable_online':
            pass
        else:
            raise NotImplementedError("Unknown merge_type.")

        ## Offline panoptic segmentation
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        
        ## Mapping to reconstructed point clouds
        mv_xyz = batch_inputs_dict['points'][0][begin_frame:num_frames, :, :3].reshape(-1, 3)
        rec_xyz = torch.tensor(batch_data_samples[0].eval_ann_info['rec_xyz'])[:, :3]
        target_coord = rec_xyz.to(mv_xyz.device).contiguous().float()
        target_offset = torch.tensor(target_coord.shape[0]).to(mv_xyz.device).float()
        source_coord = mv_xyz.contiguous().float()
        source_offset = torch.tensor(source_coord.shape[0]).to(mv_xyz.device).float()
        indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.reshape(-1).long()
    
        merged_result = PointData(
            pts_semantic_mask=[mv_sem[indices].cpu().numpy()],
            pts_instance_mask=[mv_mask[:, indices].cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())
    
        # Ensemble the predictions with mesh segments (eval_ann_info['segment_ids'])
        if 'segment_ids' in batch_data_samples[0].eval_ann_info:
            merged_result = self.segment_smooth(merged_result, mv_xyz.device,
                batch_data_samples[0].eval_ann_info['segment_ids'])
        
        batch_data_samples[0].pred_pts_seg = merged_result
        if self.use_bbox:
            batch_data_samples[0].pred_bbox = mv_bboxes.cpu().numpy()
        
    
        return batch_data_samples
    
    def segment_smooth(self, results, device, segment_ids):
        unique_ids = np.unique(segment_ids)
        new_segment_ids = np.zeros_like(segment_ids)
        for i, ids in enumerate(unique_ids):
            new_segment_ids[segment_ids == ids] = i
        segment_ids = new_segment_ids
        segment_ids = torch.from_numpy(segment_ids).to(device)
        sem_mask = torch.from_numpy(results.pts_semantic_mask[0]).to(device)
        ins_mask = torch.from_numpy(results.pts_instance_mask[0]).to(device)
        sem_mask = scatter_mean(F.one_hot(sem_mask).float(), segment_ids, dim=0)
        sem_mask = sem_mask.argmax(dim=1)[segment_ids]
        ins_mask = scatter_mean(ins_mask.float(), segment_ids, dim=1)
        ins_mask = (ins_mask > 0.5)[:, segment_ids]
        results.pts_semantic_mask[0] = sem_mask.cpu().numpy()
        results.pts_instance_mask[0] = ins_mask.cpu().numpy()
        return results
    
    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)

        sem_map2 = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        inst_res2 = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)

        pts_semantic_mask = [sem_res, sem_map2]
        pts_instance_mask = [inst_res[0].bool(), inst_res2[0].bool()]
        instance_labels = [inst_res[1], inst_res2[1]]
        instance_scores = [inst_res[2], inst_res2[2]]
        instance_queries = [inst_res[3], inst_res2[3]]
        mapping = [inst_res[4], inst_res2[4]]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=instance_labels,
                instance_scores=instance_scores,
                instance_queries=instance_queries)], mapping
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        mapping = torch.arange(len(out['cls_preds'][0])).to(superpoints.device)
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        queries = out['queries'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        queries = queries[topk_idx]
        mapping = mapping[topk_idx]

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)
            queries = queries[keep_inds]
            mapping = mapping[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries = queries[score_mask]
        mapping = mapping[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries = queries[npoint_mask]
        mapping = mapping[npoint_mask]

        return mask_pred, labels, scores, queries, mapping
    
    def predict_by_feat_panoptic(self, sem_map, mask_pred, labels, scores):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask

        return sem_map, inst_map

@MODELS.register_module()
class ScanNet200MixFormer3D_FF_Online(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(img_backbone)
        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()
        
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(960, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
    
    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()
        if hasattr(self, 'img_backbone'):
            self.img_backbone.init_weights()

    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i):
        """Extract features from sparse tensor.
        """
        # extract image features
        with torch.no_grad():
            img_features = []
            for img_paths in batch_inputs_dict['img_paths']:
                img_features.append(self.img_backbone(img_paths[frame_i])[0])
        
        # TODO check
        img_metas = []
        for batch_data_sample in batch_data_samples:
            img_meta = {}
            img_meta['depth2img'] = batch_data_sample.depth2img[frame_i]
            img_meta['sample_idx'] = batch_data_sample.sample_idx
            img_meta['img_shape'] = batch_data_sample.img_shape
            img_meta['axis_align_matrix'] = batch_data_sample.axis_align_matrix
            
            try:
                img_meta['pcd_rotation'] = batch_data_sample.pcd_rotation
                img_meta['pcd_scale_factor'] = batch_data_sample.pcd_scale_factor
                img_meta['pcd_trans'] = batch_data_sample.pcd_trans
                img_meta['pcd_rotation_angle'] = batch_data_sample.pcd_rotation_angle
                img_meta['pcd_vertical_flip'] = batch_data_sample.pcd_vertical_flip
                img_meta['pcd_horizontal_flip'] = batch_data_sample.pcd_horizontal_flip
                img_meta['transformation_3d_flow'] = batch_data_sample.transformation_3d_flow
            except:
                pass
            img_metas.append(img_meta)
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            # pdb.set_trace()
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=batch_data_samples[0].img_shape),
                          memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]  # [B, N, 3+D]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        # x = scatter_mean(x, sp_idx, dim=0)  # todo: do we need dim?
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list
    
    def _select_queries(self, x, gt_instances, sp_xyz, frame_i):
        """Select queries for train pass.
        """
        gt_instances_ = []
        for i in range(len(x)):
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(x[i].device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(x[i].device)
            bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(x[i].device)
            temp.bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(x[i].device)])
            gt_instances_.append(temp)

        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                n = max(1,n)
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks[:, ids]
                sp_xyz[i] = sp_xyz[i][ids]
            else:
                queries.append(x[i])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks
        
        return queries, gt_instances_, sp_xyz

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x
    

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.
        """
        losses, merge_feat_n_frames, ins_masks_query_n_frames = {}, [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        for frame_i in range(num_frames):
            ## Backbone
            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## GT-prepare
            gt_instances = [s.gt_instances_3d for s in batch_data_samples]
            gt_point_instances, ins_masks_query_batch = [], []
            for i in range(len(gt_instances)):
                ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i]
                if torch.sum(ins == -1) != 0:
                    # Use global instance number for each frame
                    ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    ins = F.one_hot(ins)[:, :-1]
                else:
                    ins = F.one_hot(ins)
                    max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    if ins.shape[1] < max_ids:
                        zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                        ins = torch.cat([ins, zero_pad], dim=-1)
                ins = ins.bool().T
                gt_point = InstanceData()
                gt_point.sp_masks = ins
                gt_point_instances.append(gt_point)
            queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz, frame_i)
            ## Decoder
            super_points = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
            if self.decoder_online and frame_i > 0:
                x = self.decoder(x, point_features, queries, super_points, all_xyz_w, prev_queries)
            else:
                x = self.decoder(x, point_features, queries, super_points, all_xyz_w)
            ## Query projector
            for i in range(len(gt_instances)):
                ins_masks_query = gt_instances[i].query_masks[:-self.sem_len, :]
                ins_masks_query = [ins_masks_query[i].nonzero().flatten()
                        for i in range(ins_masks_query.shape[0])]
                ins_masks_query_batch.append(ins_masks_query)
            if hasattr(self, 'merge_head'):
                merge_feat = self.merge_head(x['queries'])
                merge_feat_n_frames.append(merge_feat)
                ins_masks_query_n_frames.append(ins_masks_query_batch)
            ## Query merging
            if self.decoder_online:
                if frame_i == 0: gt_merger = GTMerge()
                prev_queries = gt_merger.merge(x['queries'], ins_masks_query_batch)
                if frame_i == num_frames - 1: gt_merger.clean()
            ## Loss
            loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)
            for key, value in loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        
        ## Query contrast
        if hasattr(self, 'merge_criterion'):
            merge_feat_n_frames = [[frame[i] for frame in merge_feat_n_frames]
                 for i in range(len(merge_feat_n_frames[0]))]
            ins_masks_query_n_frames = [[frame[i] for frame in ins_masks_query_n_frames]
                 for i in range(len(ins_masks_query_n_frames[0]))]
            loss = self.merge_criterion(merge_feat_n_frames, ins_masks_query_n_frames)
            losses.update(loss)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        for frame_i in range(num_frames):
            ## Backbone
            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## Decoder
            super_points = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
            if self.decoder_online and frame_i > 0:
                x = self.decoder(x, point_features, x, super_points, all_xyz_w, [mv_queries])
            else:
                x = self.decoder(x, point_features, x, super_points, all_xyz_w)
            ## Post-processing
            pred_pts_seg, mapping = self.predict_by_feat(
                x, batch_data_samples[0].gt_pts_seg.sp_pts_mask[frame_i])
            results.append(pred_pts_seg[0])
            ## Query projector, semantic and geometric information
            if hasattr(self, 'merge_head'):
                query_feats = self.merge_head(x['queries'][0])
                query_feats_list.append([query_feats[mapping[0]], query_feats[mapping[1]]])
                sem_preds = x['sem_preds'][0]
                sem_preds_list.append([sem_preds[mapping[0]], sem_preds[mapping[1]]])
                sp_xyz_list.append([sp_xyz[0][mapping[0]], sp_xyz[0][mapping[1]]])
                if self.use_bbox:
                    bbox_preds = x['bboxes'][0] # [N, 6]
                    bboxes_list.append([bbox_preds[mapping[0]], bbox_preds[mapping[1]]])
                cls_preds = x['cls_preds'][0]
                cls_preds_list.append([cls_preds[mapping[0]], cls_preds[mapping[1]]])
            ## Online merging
            if self.test_cfg.merge_type == 'learnable_online':
                if frame_i == 0:
                    online_merger = OnlineMerge(self.test_cfg.inscat_topk_insts, self.use_bbox)
                mv_mask, mv_labels, mv_scores, mv_queries, mv_bboxes = online_merger.merge(
                    results[-1]['pts_instance_mask'][0],
                    results[-1]['instance_labels'][0],
                    # cls_preds_list[-1][0],
                    results[-1]['instance_scores'][0],
                    results[-1]['instance_queries'][0],
                    query_feats_list[-1][0],
                    sem_preds_list[-1][0],
                    sp_xyz_list[-1][0],
                    bboxes_list[-1][0] if self.use_bbox else None)
                # Empty cache. Only offline merging requires the whole list.
                query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list = [], [], [], []
                if frame_i == num_frames - 1:
                    online_merger.clean() # Ignore panoptic segmentation
        
        ## Offline merging
        if self.test_cfg.merge_type == 'learnable':
            mv_mask, mv_labels, mv_scores = ins_merge_mat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                [res[0] for res in query_feats_list],
                [res[0] for res in sem_preds_list],
                [res[0] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge_mat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                [res[1] for res in query_feats_list],
                [res[1] for res in sem_preds_list],
                [res[1] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'concat':
            mv_mask, mv_labels, mv_scores = ins_cat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_cat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'geometric':
            mv_mask, mv_labels, mv_scores = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'learnable_online':
            pass
        else:
            raise NotImplementedError("Unknown merge_type.")

        ## Offline panoptic segmentation
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        
        ## Mapping to reconstructed point clouds
        mv_xyz = batch_inputs_dict['points'][0][:num_frames, :, :3].reshape(-1, 3)
        rec_xyz = torch.tensor(batch_data_samples[0].eval_ann_info['rec_xyz'])[:, :3]
        target_coord = rec_xyz.to(mv_xyz.device).contiguous().float()
        target_offset = torch.tensor(target_coord.shape[0]).to(mv_xyz.device).float()
        source_coord = mv_xyz.contiguous().float()
        source_offset = torch.tensor(source_coord.shape[0]).to(mv_xyz.device).float()
        indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.reshape(-1).long()

    
        merged_result = PointData(
            pts_semantic_mask=[mv_sem[indices].cpu().numpy()],
            pts_instance_mask=[mv_mask[:, indices].cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())

        # ## Ensemble the predictions with mesh segments (eval_ann_info['segment_ids'])
        if 'segment_ids' in batch_data_samples[0].eval_ann_info:
            merged_result = self.segment_smooth(merged_result, mv_xyz.device,
                batch_data_samples[0].eval_ann_info['segment_ids'])
        batch_data_samples[0].pred_pts_seg = merged_result
        if self.use_bbox:
            batch_data_samples[0].pred_bbox = mv_bboxes.cpu().numpy()
        
        return batch_data_samples
    
    def segment_smooth(self, results, device, segment_ids):
        unique_ids = np.unique(segment_ids)
        new_segment_ids = np.zeros_like(segment_ids)
        for i, ids in enumerate(unique_ids):
            new_segment_ids[segment_ids == ids] = i
        segment_ids = new_segment_ids
        segment_ids = torch.from_numpy(segment_ids).to(device)
        sem_mask = torch.from_numpy(results.pts_semantic_mask[0]).to(device)
        ins_mask = torch.from_numpy(results.pts_instance_mask[0]).to(device)
        sem_mask = scatter_mean(F.one_hot(sem_mask).float(), segment_ids, dim=0)
        sem_mask = sem_mask.argmax(dim=1)[segment_ids]
        ins_mask = scatter_mean(ins_mask.float(), segment_ids, dim=1)
        ins_mask = (ins_mask > 0.5)[:, segment_ids]
        results.pts_semantic_mask[0] = sem_mask.cpu().numpy()
        results.pts_instance_mask[0] = ins_mask.cpu().numpy()
        return results
    
    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)

        sem_map2 = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        inst_res2 = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)

        pts_semantic_mask = [sem_res, sem_map2]
        pts_instance_mask = [inst_res[0].bool(), inst_res2[0].bool()]
        instance_labels = [inst_res[1], inst_res2[1]]
        instance_scores = [inst_res[2], inst_res2[2]]
        instance_queries = [inst_res[3], inst_res2[3]]
        mapping = [inst_res[4], inst_res2[4]]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=instance_labels,
                instance_scores=instance_scores,
                instance_queries=instance_queries)], mapping
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        mapping = torch.arange(len(out['cls_preds'][0])).to(superpoints.device)
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        queries = out['queries'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        queries = queries[topk_idx]
        mapping = mapping[topk_idx]

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)
            queries = queries[keep_inds]
            mapping = mapping[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries = queries[score_mask]
        mapping = mapping[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries = queries[npoint_mask]
        mapping = mapping[npoint_mask]

        return mask_pred, labels, scores, queries, mapping
    
    def predict_by_feat_panoptic(self, sem_map, mask_pred, labels, scores):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask

        return sem_map, inst_map

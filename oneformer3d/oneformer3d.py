import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
import pointops
import copy
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmengine.structures import InstanceData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms
import numpy as np
import os

class ScanNetOneFormer3DMixin:
    """Class contains common methods for ScanNet and ScanNet200."""

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
        pan_res = self.predict_by_feat_panoptic(out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy()]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]
    
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

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
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

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
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
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores  = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)
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
    
    def _select_queries(self, x, gt_instances, sp_xyz=None):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
                if sp_xyz is not None:
                    sp_xyz[i] = sp_xyz[i][ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        if sp_xyz is not None:
            return queries, gt_instances, sp_xyz
        return queries, gt_instances


@MODELS.register_module()
class ScanNetOneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    r"""OneFormer3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
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
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 query_thr,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

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
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg

            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_gt_instances = self._select_queries(x, sp_gt_instances)
        x = self.decoder(x, queries)
        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
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
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        x = self.decoder(x, x)

        results_list = self.predict_by_feat(x, sp_pts_masks)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples


@MODELS.register_module()
class ScanNet200OneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
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
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse())
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field).features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x = scatter_mean(x, torch.cat(sp_pts_masks), dim=0)  # todo: do we need dim?

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features

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
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        queries, gt_instances = self._select_queries(x, gt_instances)
        x = self.decoder(x, queries)
        return self.criterion(x, gt_instances)

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
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        x = self.decoder(x, x)
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples

@MODELS.register_module()
class ScanNet200OneFormer3D_Online(ScanNetOneFormer3DMixin, Base3DDetector):
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
                 memory=None,
                 neck=None,
                 decoder=None,
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
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()
            
    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i):
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
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x = scatter_mean(x, torch.cat(sp_pts_masks), dim=0)  # todo: do we need dim?
        
        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features
    
    def _select_queries(self, x, gt_instances, frame_i):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        gt_instances_ = []
        # 取出对应frame的gt_instances
        for i in range(len(x)):
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(x[i].device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(x[i].device)
            gt_instances_.append(temp)

        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks
        
        return queries, gt_instances_
    
    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):    # sv
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
        # breakpoint()
        losses = {}
        merge_feat_n_frames = []
        ins_masks_query_n_frames = []
        if hasattr(self, 'memory'):
            self.memory.reset()
        for frame_i in range(batch_inputs_dict['points'][0].shape[0]):
            x = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            gt_instances = [s.gt_instances_3d for s in batch_data_samples]
            # gt_point_instances = []
            # for i in range(len(gt_instances)):
            #     ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i]
            #     if torch.sum(ins == -1) != 0:
            #         # Use global instance number for each frame
            #         ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - 21
            #         ins = torch.nn.functional.one_hot(ins)[:, :-1]
            #     else:
            #         ins = torch.nn.functional.one_hot(ins)
            #         max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - 21
            #         if ins.shape[1] < max_ids:
            #             zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
            #             ins = torch.cat([ins, zero_pad], dim=-1)   
            #     ins = ins.bool().T
            #     gt_point = InstanceData()
            #     gt_point.sp_masks = ins
            #     gt_point_instances.append(gt_point)
            queries, gt_instances = self._select_queries(x, gt_instances, frame_i)
            x = self.decoder(x, queries)
            loss = self.criterion(x, gt_instances)
            for key, value in loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        return losses

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
        results = []
        if hasattr(self, 'memory'):
            self.memory.reset()
            
        points_per_frame = batch_inputs_dict['points'][0].shape[1]
        
        for frame_i in range(batch_inputs_dict['points'][0].shape[0]):                       
            x = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            x = self.decoder(x, x)
            pred_pts_seg = self.predict_by_feat(
                x, batch_data_samples[0].gt_pts_seg.sp_pts_mask[frame_i])
            results.append(pred_pts_seg[0])
            

        mv_mask, mv_labels, mv_scores = self.ins_merge_dist(
            [points for points in batch_inputs_dict['points'][0]],
            [res['pts_instance_mask'][0] for res in results],
            [res['instance_labels'][0] for res in results],
            [res['instance_scores'][0] for res in results],
            )
        mv_mask2, mv_labels2, mv_scores2 = mv_mask, mv_labels, mv_scores
        
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        mv_sem2 = torch.cat([res['pts_semantic_mask'][1] for res in results])
        pan_sem, pan_inst = self.predict_by_feat_panoptic(mv_sem2, mv_mask2, mv_labels2, mv_scores2)

        # 找到源点云中每个点的最近邻目标点，进行点云配准
        mv_xyz = batch_inputs_dict['points'][0][:, :, :3].reshape(-1, 3)        # 源点云坐标，shape为(n_frames*n_points, 3)
        rec_xyz = torch.tensor(batch_data_samples[0].eval_ann_info['rec_xyz'])  # 目标点云坐标，shape为(n_real_points, 3)
        target_coord = rec_xyz.to(mv_xyz.device).contiguous().float()
        target_offset = torch.tensor(target_coord.shape[0]).to(mv_xyz.device).float()
        source_coord = mv_xyz.contiguous().float()
        source_offset = torch.tensor(source_coord.shape[0]).to(mv_xyz.device).float()
        indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.reshape(-1).long()    # 每个目标点距离最近的源点的索引，shape为(n_real_points)

        # 各项shape分别为：(2, n_real_points), (2, n_instances, n_real_points), (n_instances), (n_instances)
        merged_result = PointData(
            pts_semantic_mask=[mv_sem[indices].cpu().numpy(), pan_sem[indices].cpu().numpy()],
            pts_instance_mask=[mv_mask[:, indices].cpu().numpy(), pan_inst[indices].cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())
        # TODO: ensemble the predictions with mesh segments (eval_ann_info['segment_ids'])
        batch_data_samples[0].pred_pts_seg = merged_result
        return batch_data_samples

    def ins_merge_dist(self, points, masks, labels, scores):
        frame_num = len(points)
        pts_per_frame = points[0].shape[0]
        cur_instances = [InstanceQuery(mask, label, score)
                         for mask, label, score \
                        in zip(masks[0], labels[0], scores[0])]
        cur_points = points[0]          # 第一帧的点云, shape为(n_points, 6)
        for i in range(1, frame_num):
            for mask, label, score \
                in zip(masks[i], labels[i], scores[i]):
                is_merge = False
                for InsQ in cur_instances:
                    # merged ins
                    # 后续帧的每个instance与当前帧的每个instance进行比较，如果满足条件则合并入当前帧inst
                    if InsQ.compare(cur_points, points[i], mask, label, score):
                        InsQ.merge(mask, label, score)
                        is_merge = True
                        break
                # new ins
                # 如果这一后续帧的instance没有与当前帧的instance合并，则认为是新的instance
                if not is_merge:
                    mask = torch.cat([mask.new_zeros(pts_per_frame*i).bool(), mask])    # 在前面padding
                    cur_instances.append(InstanceQuery(mask, label, score))      # 加入新inst
            cur_points = torch.cat([cur_points, points[i]])  # 将后续帧的点云拼接到当前帧的点云后面
            # not merged ins
            for InsQ in cur_instances:
                if len(InsQ.mask) < cur_points.shape[0]:    
                    InsQ.pad(pts_per_frame)    # 将所有的instance都padding，使其点数满足当前累计点云数量
        # 整合所有instance，个数过多则取前topk个
        merged_mask = torch.stack([InsQ.mask for InsQ in cur_instances], dim=0)
        merged_labels = torch.tensor([InsQ.label for InsQ in cur_instances]).to(merged_mask.device)
        merged_scores = torch.tensor([InsQ.score for InsQ in cur_instances]).to(merged_mask.device)
        ## 确实过多，len(merged_scores)有688个，取前100个
        if len(merged_scores) > self.test_cfg.inscat_topk_insts:
            _, kept_ins = merged_scores.topk(self.test_cfg.inscat_topk_insts)
        else:
            kept_ins = ...
        merged_mask, merged_labels, merged_scores = \
            merged_mask[kept_ins], merged_labels[kept_ins], merged_scores[kept_ins]
        
        return merged_mask, merged_labels, merged_scores

    def predict_by_feat(self, out, superpoints):
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
        
        return [PointData(
            pts_semantic_mask=pts_semantic_mask,
            pts_instance_mask=pts_instance_mask,
            instance_labels=instance_labels,
            instance_scores=instance_scores)]

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
    
class InstanceQuery():
    def __init__(self, mask, label, score):
        self.mask = mask
        self.label = label
        self.score = score
        self.merge_count = 1

    def pad(self, pts_num):
        self.mask = torch.cat([self.mask, self.mask.new_zeros(pts_num).bool()]) # 在后面padding
    
    def compare(self, cur_points, points, mask, label, score, pts_thr=0.05, thr=0.1):
        # 如果该instance已经融合过（mask的len会大于cur_points.shape[0]），则不再融合
        if cur_points.shape[0] != len(self.mask):
            return False
        if self.label != label:
            return False

        # 采样后计算距离矩阵
        cur_xyz = cur_points[self.mask, :3].unsqueeze(1) # Mx3，shape为(n_cur_points, 1, 3)
        if cur_xyz.shape[0] > 10000:
            sample_idx = torch.randperm(cur_xyz.shape[0])[:10000]
            cur_xyz = cur_xyz[sample_idx]
        xyz = points[mask, :3].unsqueeze(0) # Nx3，shape为(1, n_points, 3)
        if xyz.shape[0] > 10000:
            sample_idx = torch.randperm(xyz.shape[0])[:10000]
            xyz = xyz[sample_idx]

        dist_mat = cur_xyz - xyz # MxNx3，shape为(n_cur_points, n_points, 3)
        dist_mat = (dist_mat ** 2).sum(-1).sqrt() # MxN, shape为(n_cur_points, n_points)
        min_dist1 = dist_mat.min(-1).values # M, shape为(n_cur_points)
        min_dist2 = dist_mat.min(0).values # N, shape为(n_points)
        ratio1 = (min_dist1 < pts_thr).sum() / len(min_dist1)
        ratio2 = (min_dist2 < pts_thr).sum() / len(min_dist2)
        if max(ratio1, ratio2) > thr:
            return True
        else:
            return False
   
    def merge(self, mask, label, score):
        # mask直接拼接，score取平均
        self.mask = torch.cat([self.mask, mask])
        self.score = (self.score * self.merge_count + score) / (self.merge_count + 1)
        self.merge_count += 1


@MODELS.register_module()
class S3DISOneFormer3D(Base3DDetector):
    pass

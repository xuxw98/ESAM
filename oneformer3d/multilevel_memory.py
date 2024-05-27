# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, constant_init
import pdb


@MODELS.register_module()
class MultilevelMemory(BaseModule):
    def __init__(self, in_channels=[64, 128, 256, 512], scale=2.5, queue=-1, vmp_layer=(0,1,2,3)):
        super().__init__()
        self.scale = scale
        self.queue = queue
        self.vmp_layer = list(vmp_layer)
        self.conv_d1 = nn.ModuleList()
        self.conv_d3 = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        for i, C in enumerate(in_channels):
            if i in self.vmp_layer:
                self.conv_d1.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_d3.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=3,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_convert.append(nn.Sequential(
                    ME.MinkowskiConvolutionTranspose(
                        in_channels=3*C,
                        out_channels=C,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C)))
            else:
                self.conv_d1.append(nn.Identity())
                self.conv_d3.append(nn.Identity())
                self.conv_convert.append(nn.Identity())
        self.relu = ME.MinkowskiReLU()
        self.accumulated_feats = None
        if self.queue > 0:
            self.pruning = ME.MinkowskiPruning()
            self.time_st = 0
            self.accumulated_ts = None
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiConvolutionTranspose):
                constant_init(m.kernel, 0)

            if isinstance(m, ME.MinkowskiBatchNorm):
                constant_init(m.bn.weight, 1)
                constant_init(m.bn.bias, 0)
    
    def reset(self):
        self.accumulated_feats = None
        if self.queue > 0:
            self.time_st = 0
            self.accumulated_ts = None
    
    def global_avg_pool_and_cat(self, feat1, feat2, feat3):
        coords1 = feat1.decomposed_coordinates
        feats1 = feat1.decomposed_features
        coords2 = feat2.decomposed_coordinates
        feats2 = feat2.decomposed_features
        coords3 = feat3.decomposed_coordinates
        feats3 = feat3.decomposed_features
        for i in range(len(coords3)):
            # shape 1 N
            global_avg_feats3 = torch.mean(feats3[i], dim=0).unsqueeze(0).repeat(coords3[i].shape[0],1)
            feats1[i] = torch.cat([feats1[i], feats2[i]], dim=1)     
            feats1[i] = torch.cat([feats1[i], global_avg_feats3], dim=1)      
        coords_sp, feats_sp = ME.utils.sparse_collate(coords1, feats1)
        feat_new = ME.SparseTensor(
            coordinates=coords_sp,
            features=feats_sp,
            tensor_stride=feat1.tensor_stride,
            coordinate_manager=feat1.coordinate_manager
        )
        return feat_new
    
    def accumulate(self, accumulated_feat, accumulated_t, current_feat, index):
        """Accumulate features for a single stage.

        Args:
            accumulated_feat (ME.SparseTensor)
            current_feat (ME.SparseTensor)

        Returns:
            ME.SparseTensor: refined accumulated features
            ME.SparseTensor: current features after accumulation
        """
        if index in self.vmp_layer:
            # VMP
            tensor_stride = current_feat.tensor_stride
            accumulated_feat = ME.TensorField(
                features=torch.cat([current_feat.features, accumulated_feat.features], dim=0),
                coordinates=torch.cat([current_feat.coordinates, accumulated_feat.coordinates], dim=0),
                quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
            ).sparse()
            accumulated_feat = ME.SparseTensor(
                coordinates=accumulated_feat.coordinates,
                features=accumulated_feat.features,
                tensor_stride=tensor_stride,
                coordinate_manager=accumulated_feat.coordinate_manager
            )

            # queued cache
            if self.queue > 0:
                current_t = ME.SparseTensor(
                    coordinates=current_feat.coordinates,
                    features=(torch.ones(current_feat.coordinates.shape[0],1)*self.time_st).to(current_feat.device),
                    tensor_stride=tensor_stride,
                    coordinate_manager=current_feat.coordinate_manager
                )
                accumulated_t = ME.TensorField(
                    features=torch.cat([current_t.features, accumulated_t.features], dim=0),
                    coordinates=torch.cat([current_t.coordinates, accumulated_t.coordinates], dim=0),
                    quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
                ).sparse()
                accumulated_t = ME.SparseTensor(
                    coordinates=accumulated_feat.coordinates,
                    features=accumulated_t.features_at_coordinates(accumulated_feat.coordinates.float()),
                    tensor_stride=tensor_stride,
                    coordinate_manager=accumulated_feat.coordinate_manager
                )
                if (accumulated_t.features.max() - accumulated_t.features.min()) >= (self.queue - 1):
                    mask = (accumulated_t.features != accumulated_t.features.min()).squeeze(1)
                    accumulated_t = self.pruning(accumulated_t, mask)
                    accumulated_feat = self.pruning(accumulated_feat, mask)

            # Select neighbor region for current frame
            accumulated_coords = accumulated_feat.decomposed_coordinates
            current_coords = current_feat.decomposed_coordinates
            accumulated_coords_select_list=[]
            zero_batch_feature_list=[]
            for i in range(len(current_coords)):
                accumulated_coords_batch = accumulated_coords[i]
                current_coords_batch = current_coords[i]
                current_coords_batch_max, _ = torch.max(current_coords_batch,dim=0)
                current_coords_batch_min, _ = torch.min(current_coords_batch,dim=0)
                current_box_size = current_coords_batch_max - current_coords_batch_min
                current_box_add = ((self.scale-1)/2) * current_box_size
                margin_positive = accumulated_coords_batch-current_coords_batch_max-current_box_add
                margin_negative = accumulated_coords_batch-current_coords_batch_min+current_box_add
                in_criterion = torch.mul(margin_positive,margin_negative)
                zero = torch.zeros_like(in_criterion)
                one = torch.ones_like(in_criterion)
                in_criterion = torch.where(in_criterion<=0,one,zero)
                mask = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2]
                mask = mask.type(torch.bool)
                mask = mask.reshape(mask.shape[0],1)
                accumulated_coords_batch_select = torch.masked_select(accumulated_coords_batch,mask)
                accumulated_coords_batch_select = accumulated_coords_batch_select.reshape(-1,3)
                zero_batch_feature = torch.zeros_like(accumulated_coords_batch_select)
                accumulated_coords_select_list.append(accumulated_coords_batch_select)
                zero_batch_feature_list.append(zero_batch_feature)
            accumulated_coords_select_coords, _ = ME.utils.sparse_collate(accumulated_coords_select_list, zero_batch_feature_list)
            current_feat_new = ME.SparseTensor(
                coordinates=accumulated_coords_select_coords,
                features=accumulated_feat.features_at_coordinates(accumulated_coords_select_coords.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager # new shorcut
            )

            branch1 = self.conv_d1[index](current_feat_new)
            branch3 = self.conv_d3[index](current_feat_new)
            branch  = self.global_avg_pool_and_cat(branch1, branch3, current_feat_new)
            branch = self.conv_convert[index](branch)
            current_feat_new = branch + current_feat # new shorcut
            current_feat_new = self.relu(current_feat_new)
            current_feat = ME.SparseTensor(
                coordinates=current_feat.coordinates,
                features=current_feat_new.features_at_coordinates(current_feat.coordinates.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager
            )
        return accumulated_feat, accumulated_t, current_feat
    
    def forward(self, x):
        if self.accumulated_feats is None:
            accumulated_feats = x
            for i in range(len(x)):
                if i in self.vmp_layer:
                    branch1 = self.conv_d1[i](x[i])
                    branch3 = self.conv_d3[i](x[i])
                    branch  = self.global_avg_pool_and_cat(branch1, branch3, x[i])
                    branch = self.conv_convert[i](branch)
                    x[i] = branch + x[i]
                    x[i] = self.relu(x[i])
            self.accumulated_feats = accumulated_feats
            if self.queue > 0:
                self.accumulated_ts = [ME.SparseTensor(
                    coordinates=x[i].coordinates,
                    features=(torch.ones(x[i].coordinates.shape[0],1)*self.time_st).to(x[i].device),
                    tensor_stride=x[i].tensor_stride,
                    coordinate_manager=x[i].coordinate_manager
                ) for i in range(len(x))]
                self.time_st += 1
            return x
        else:
            if self.queue > 0:
                tuple_feats = [self.accumulate(self.accumulated_feats[i], self.accumulated_ts[i], x[i], i)
                     for i in range(len(x))]
                self.accumulated_ts = [tuple_feats[i][1] for i in range(len(x))]
                self.time_st += 1
            else:
                tuple_feats = [self.accumulate(self.accumulated_feats[i], None, x[i], i) for i in range(len(x))]
            self.accumulated_feats = [tuple_feats[i][0] for i in range(len(x))]
            return [tuple_feats[i][2] for i in range(len(x))]

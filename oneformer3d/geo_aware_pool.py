from turtle import forward
import torch
import torch.nn as nn
import pdb, time
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean, scatter


@MODELS.register_module()
class GeoAwarePooling(BaseModule):
    """Pool point features to super points.
    """
    def __init__(self, channel_proj):
        super().__init__()
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, channel_proj),
            nn.LayerNorm(channel_proj)
        )
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * channel_proj, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, 1, bias=False),
            nn.Sigmoid()
        )
    
    def scatter_norm(self, points, idx):
        ''' Normalize positions of same-segment in a unit sphere of diameter 1
        Code is copied from SPT
        '''
        min_segment = scatter(points, idx, dim=0, reduce='min')
        max_segment = scatter(points, idx, dim=0, reduce='max')
        diameter_segment = (max_segment - min_segment).max(dim=1).values
        center_segment = scatter(points, idx, dim=0, reduce='mean')
        center = center_segment[idx]
        diameter = diameter_segment[idx]
        points = (points - center) / (diameter.view(-1, 1) + 1e-2)
        return points, diameter_segment.view(-1, 1)

    def forward(self, x, sp_idx, all_xyz, with_xyz=False):
        all_xyz_ = torch.cat(all_xyz)
        all_xyz, _ = self.scatter_norm(all_xyz_, sp_idx)
        all_xyz = self.pts_proj1(all_xyz)
        all_xyz_segment = scatter(all_xyz, sp_idx, dim=0, reduce='max')
        all_xyz = torch.cat([all_xyz, all_xyz_segment[sp_idx]], dim=-1)
        all_xyz_w = self.pts_proj2(all_xyz) * 2
        if with_xyz:
            x = torch.cat([x * all_xyz_w, all_xyz_], dim=-1)
            x = scatter_mean(x, sp_idx, dim=0)
            x[:, :-3] = x[:, :-3] + all_xyz_segment
        else:
            x = scatter_mean(x * all_xyz_w, sp_idx, dim=0) + all_xyz_segment
        return x, all_xyz_w

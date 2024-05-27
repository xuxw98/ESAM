import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class MergeHead(BaseModule):
    def __init__(self, in_channels, out_channels, d_hidden=512, hidden=1, norm='batch'):
        super(MergeHead, self).__init__()
        assert norm in ['batch', 'layer']
        if hidden == 1:
            if norm == 'batch':
                self.net = nn.Sequential(
                    nn.Linear(in_channels, d_hidden),
                    nn.BatchNorm1d(d_hidden),
                    nn.ReLU(),
                    nn.Linear(d_hidden, out_channels),
                    nn.BatchNorm1d(out_channels),
                )
            elif norm == 'layer':
                self.net = nn.Sequential(
                    nn.Linear(in_channels, d_hidden),
                    nn.LayerNorm(d_hidden),
                    nn.ReLU(),
                    nn.Linear(d_hidden, out_channels),
                    nn.LayerNorm(out_channels),
                )
        elif hidden == 0:
            self.net = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
            )  
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            res = F.normalize(self.net(x), p=2, dim=-1)
            return res
        results = []
        for data in x:
            res = F.normalize(self.net(data), p=2, dim=-1)
            results.append(res)
        return results

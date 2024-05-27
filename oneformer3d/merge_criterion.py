from multiprocessing import reduction
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb, time

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ScanNetMergeCriterion_Fast:
    """
        Merge criterion for ScanNet.
    """
    def __init__(self, tmp=True, p2s=True, tmp_weight=2.0, p2s_weight=2.0):
        self.tmp = tmp
        self.p2s = p2s
        self.tmp_weight = tmp_weight
        self.p2s_weight = p2s_weight
        self.criterion = nn.CrossEntropyLoss()
    
    def tmp_loss_fast(self, feats_t0, ins_mask_t0, feats_t1, ins_mask_t1, tau=0.07):
        n_instances = len(ins_mask_t0)
        if n_instances == 0:
            return torch.tensor(0.0, device=feats_t0.device)
        centers_t0, centers_t1 = [], []
        for inst in range(n_instances):
            centers_t0.append(feats_t0[ins_mask_t0[inst]].mean(0) if
                 len(ins_mask_t0[inst]) != 0 else torch.zeros_like(feats_t0[0]))
            centers_t1.append(feats_t1[ins_mask_t1[inst]].mean(0) if
                 len(ins_mask_t1[inst]) != 0 else torch.zeros_like(feats_t1[0]))
        centers_t0 = torch.stack(centers_t0, dim=0)
        centers_t1 = torch.stack(centers_t1, dim=0)
        logits = torch.mm(centers_t0, centers_t1.transpose(1, 0))
        target = torch.arange(centers_t0.shape[0], device=centers_t0.device).long()
        out = torch.div(logits, tau).contiguous()
        valid_mask = (out.diag() != 0)
        out, target = out[valid_mask, :], target[valid_mask]
        out = torch.where(out == 0, torch.full_like(out, -torch.inf), out)
        loss = self.criterion(out, target) if len(target) > 0 else torch.tensor(0.0, device=feats_t0.device)
        return loss
    
    def p2s_loss_fast(self, feats, ins_mask, tau=0.02):
        n_instances = len(ins_mask)
        if n_instances == 0:
            return torch.tensor(0.0, device=feats.device)
        losses, centers, inst_count, inst_feats = [], [], [], []
        for inst in range(n_instances):
            inst_count.append(len(ins_mask[inst]))
            inst_feats.append(feats[ins_mask[inst]])
            centers.append(inst_feats[-1].max(0)[0] if
                 inst_count[-1] != 0 else torch.zeros_like(feats[0]))
        centers = torch.stack(centers, dim=0)
        max_count = max(inst_count)
        if max_count == 0:
            return torch.tensor(0.0, device=feats.device)
        for i in range(max_count):
            new_array = []
            for inst in range(n_instances):
                if inst_count[inst] - 1 >= i:
                    new_array.append(inst_feats[inst][i])
                elif inst_count[inst] == 0:
                    new_array.append(torch.zeros_like(feats[0]))
                else:
                    new_array.append(inst_feats[inst][-1])
            new_array = torch.stack(new_array, dim=0)
            logits = torch.mm(centers, new_array.transpose(1, 0))
            target = torch.arange(centers.shape[0], device=centers.device).long()
            out = torch.div(logits, tau).contiguous()
            valid_mask = (out.diag() != 0)
            out, target = out[valid_mask, :], target[valid_mask]
            out = torch.where(out == 0, torch.full_like(out, -torch.inf), out)
            loss = self.criterion(out, target) if len(target) > 0 else torch.tensor(0.0, device=feats.device)
            losses.append(loss)
        return sum(losses) / len(losses)
    
    def __call__(self, merge_feat, ins_masks):
        tmp_losses = []
        p2s_losses = []
        for merge_feat, ins_mask in zip(merge_feat, ins_masks):
            n_frames = len(ins_mask)
            if self.tmp:
                tmpLoss = 0.0
                for frame_i in range(n_frames-1):
                    tmpLoss += self.tmp_loss_fast(merge_feat[frame_i], ins_mask[frame_i],
                        merge_feat[frame_i+1], ins_mask[frame_i+1])
                for frame_i in range(1,n_frames):
                    tmpLoss += self.tmp_loss_fast(merge_feat[frame_i], ins_mask[frame_i],
                        merge_feat[frame_i-1], ins_mask[frame_i-1])
                tmp_losses.append(tmpLoss * self.tmp_weight)
            if self.p2s:
                p2sLoss = 0.0
                for frame_i in range(n_frames):
                    p2sLoss += self.p2s_loss_fast(merge_feat[frame_i], ins_mask[frame_i])
                p2s_losses.append(p2sLoss * self.p2s_weight)

        loss_dict = {}
        if self.tmp:
            loss_dict.update(dict(merge_loss_tmp=torch.mean(torch.stack(tmp_losses))))
        if self.p2s:
            loss_dict.update(dict(merge_loss_p2s=torch.mean(torch.stack(p2s_losses))))
        return loss_dict


def tmp_Loss(feats_t, ins_mask_t, feats_t1, ins_mask_t1):
    n_instances = len(ins_mask_t)
    if n_instances < 1:
        return torch.tensor(0.0).cuda()
    loss = torch.tensor(0.0).cuda()
    tau = 0.07
    for inst in range(n_instances):
        if len(ins_mask_t[inst]) == 0 or len(ins_mask_t1[inst]) == 0:
            continue
        center_t = feats_t[ins_mask_t[inst]].mean(0)
        center_t1 = feats_t1[ins_mask_t1[inst]].mean(0)
        e_same = torch.exp((center_t * center_t1).sum(0) / tau) 
        e_all = 0
        for i in range(n_instances):
            if len(ins_mask_t1[i]) == 0:
                continue
            center_i = feats_t1[ins_mask_t1[i]].mean(0)
            e_all += torch.exp((center_t * center_i).sum(0) / tau)
        loss += -torch.log(e_same / e_all)
    return loss / n_instances

def p2s_Loss(feats, ins_mask):
    n_instances = len(ins_mask)
    if n_instances < 1:
        return torch.tensor(0.0).cuda()
    loss = torch.tensor(0.0).cuda()
    tau = 0.02
    for inst in range(n_instances):
        if len(ins_mask[inst]) == 0:
            continue
        center = feats[ins_mask[inst]].max(0)[0]
        for i in range(len(feats[ins_mask[inst]])):
            e_same = torch.exp((center * feats[ins_mask[inst]][i]).sum(0) / tau)
            e_all = e_same.clone()
            for j in range(n_instances):
                if j == inst or len(ins_mask[j]) == 0:
                    continue
                rand_idx = np.random.randint(0, len(ins_mask[j])) # min(i, len(ins_mask[j]) - 1)
                feat_j = feats[ins_mask[j]][rand_idx]
                e_all += torch.exp((center * feat_j).sum(0) / tau)
            loss += -torch.log(e_same / e_all) / len(feats[ins_mask[inst]])
    return loss / n_instances


@MODELS.register_module()
class ScanNetMergeCriterion_Seal:
    """
        Merge criterion for ScanNet.
        Using losses introduced in the paper Seal.
    """
    def __init__(self):
        pass
    
    def __call__(self, merge_feat, ins_masks):
        tmp_losses = []
        p2s_losses = []
        losses = []
        for merge_feat, ins_mask in zip(merge_feat, ins_masks):
            n_instances = len(ins_mask[0])
            n_frames = len(ins_mask)
            tmpLoss = 0.0
            p2sLoss = 0.0
            for frame_i in range(n_frames-1):
                tmpLoss += tmp_Loss(merge_feat[frame_i], ins_mask[frame_i], merge_feat[frame_i+1], ins_mask[frame_i+1])
            for frame_i in range(1,n_frames):
                tmpLoss += tmp_Loss(merge_feat[frame_i], ins_mask[frame_i], merge_feat[frame_i-1], ins_mask[frame_i-1])
            for frame_i in range(n_frames):
                p2sLoss += p2s_Loss(merge_feat[frame_i], ins_mask[frame_i])
            tmp_losses.append(tmpLoss)
            p2s_losses.append(p2sLoss)
            loss = tmpLoss + p2sLoss
            losses.append(loss)

        if len(tmp_losses) == 0:
            tmp_loss= torch.tensor(0.0).cuda()
        else:
            tmp_loss = torch.mean(torch.stack(tmp_losses)).cuda()
        if len(p2s_losses) == 0:
            p2s_loss = torch.tensor(0.0).cuda()
        else:
            p2s_loss = torch.mean(torch.stack(p2s_losses)).cuda()
        if len(losses) == 0:
            loss = torch.tensor(0.0).cuda()
        else:
            loss = torch.mean(torch.stack(losses)).cuda()
        return dict(merge_loss_tmp=tmp_loss, merge_loss_p2s=p2s_loss, merge_loss=loss)
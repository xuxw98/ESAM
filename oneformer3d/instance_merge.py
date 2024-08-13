import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
import pdb
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

# This function is deprecated by OnlineMerge. No update anymore.
def ins_merge_mat(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, inscat_topk_insts):
    """Merge multiview instances according to geometry and query feature
    """
    weights = [0.4,0.4,0.2]
    threshold = 0.75
    frame_num = len(masks)
    points_per_mask = masks[0].shape[1]
    cur_masks, cur_labels, cur_scores, cur_queries, cur_query_feats, cur_sem_preds, cur_xyz = \
        masks[0], labels[0], scores[0], queries[0], query_feats[0], sem_preds[0], xyz_list[0]
    for i in range(1, frame_num):
        next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
            masks[i], labels[i], scores[i], queries[i], query_feats[i], sem_preds[i], xyz_list[i]
        query_feat_scores = (cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
        sem_pred_scores = F.cosine_similarity(cur_sem_preds.unsqueeze(1), next_sem_preds.unsqueeze(0), dim=2)
        xyz_dists = torch.cdist(cur_xyz, next_xyz, p=2)
        xyz_scores = 1 / (xyz_dists + 1e-6)
        
        mix_scores = weights[0] * query_feat_scores + weights[1] * sem_pred_scores + weights[2] * xyz_scores
        mix_scores = torch.where(mix_scores > threshold, mix_scores, torch.zeros_like(mix_scores))
        if mix_scores.shape[0] < mix_scores.shape[1]:
            mix_scores = torch.cat((mix_scores, torch.zeros((mix_scores.shape[1]
                    - mix_scores.shape[0], mix_scores.shape[1])).to(mix_scores.device)), dim=0)
        # Hungarian assign
        row_ind, col_ind = linear_sum_assignment(-mix_scores.cpu())
        row_ind = torch.tensor(row_ind).to(mix_scores.device)
        col_ind = torch.tensor(col_ind).to(mix_scores.device)
        mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
        row_ind = row_ind[mix_scores_mask]
        col_ind = col_ind[mix_scores_mask]

        temp = torch.zeros(cur_masks.shape[0]).bool().to(cur_masks.device)
        temp[row_ind] = True
        temp = temp.unsqueeze(1)
        temp_masks = torch.zeros((cur_masks.shape[0], points_per_mask)).bool().to(cur_masks.device)
        temp_masks[row_ind] = next_masks[col_ind]
        next_masks_ = torch.where(temp, temp_masks,
                                    torch.zeros((cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
        cur_masks = torch.cat((cur_masks, next_masks_), dim=1)
        no_merge_masks = torch.tensor(np.setdiff1d(np.arange(next_masks.shape[0]),
                col_ind.cpu())).to(next_masks.device)
        former_padding = torch.zeros((no_merge_masks.shape[0], points_per_mask * i)).bool().to(next_masks.device)
        new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
        cur_masks = torch.cat((cur_masks, new_masks), dim=0)
        
        cur_scores[row_ind] = (cur_scores[row_ind] * i + next_scores[col_ind]) / (i + 1)
        cur_scores = torch.cat((cur_scores, next_scores[no_merge_masks]), dim=0)
        cur_queries[row_ind] = (cur_queries[row_ind] * i + next_queries[col_ind]) / (i + 1)
        cur_queries = torch.cat((cur_queries, next_queries[no_merge_masks]), dim=0)
        cur_query_feats[row_ind] = (cur_query_feats[row_ind] * i + next_query_feats[col_ind]) / (i + 1)
        cur_query_feats = torch.cat((cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
        cur_sem_preds[row_ind] = (cur_sem_preds[row_ind] * i + next_sem_preds[col_ind]) / (i + 1)
        cur_sem_preds = torch.cat((cur_sem_preds, next_sem_preds[no_merge_masks]), dim=0)
        cur_xyz[row_ind] = (cur_xyz[row_ind] * i + next_xyz[col_ind]) / (i + 1)
        cur_xyz = torch.cat((cur_xyz, next_xyz[no_merge_masks]), dim=0)
    
    if len(cur_scores) > inscat_topk_insts:
        _, kept_ins = cur_scores.topk(inscat_topk_insts)
    else:
        kept_ins = ...
    cur_masks, cur_scores = cur_masks[kept_ins], cur_scores[kept_ins]
    cur_labels = torch.zeros_like(cur_scores).long()
    return cur_masks, cur_labels, cur_scores
       
def ins_cat(masks, labels, scores, inscat_topk_insts):
    """Directly stack multiview instances without mask merging"""
    frame_num = len(masks)
    labels = torch.cat(labels)
    scores = torch.cat(scores)
    if len(scores) > inscat_topk_insts:
        _, kept_ins = scores.topk(inscat_topk_insts)
    else:
        kept_ins = ...
    labels, scores = labels[kept_ins], scores[kept_ins]
    ins_num = [mask.shape[0] for mask in masks]
    frame_indicator = torch.cat([torch.ones(num)*i for i, num in enumerate(ins_num)])
    frame_indicator = frame_indicator.to(scores.device)[kept_ins]
    masks = torch.cat(masks, dim=0)[kept_ins]
    new_mask = masks.new_zeros(size=(masks.shape[0], frame_num*masks.shape[1]))
    for ids in range(len(ins_num)):
        this_frame = (frame_indicator == ids)
        new_mask[this_frame, ids*masks.shape[1]:(ids+1)*masks.shape[1]] = masks[this_frame]
    return new_mask, labels, scores

def ins_merge(points, masks, labels, scores, queries, inscat_topk_insts):
    """Merge multiview instances according to geometry and query feature"""
    frame_num = len(points)
    pts_per_frame = points[0].shape[0]
    cur_instances = [InstanceQuery(mask, label, score, query) for mask, label, score, query \
            in zip(masks[0], labels[0], scores[0], queries[0])]
    cur_points = points[0]
    for i in range(1, frame_num):
        for mask, label, score, query in zip(masks[i], labels[i], scores[i], queries[i]):
            is_merge = False
            for InsQ in cur_instances:
                # merged ins
                if InsQ.compare(cur_points, points[i], mask, label, score, query):
                    InsQ.merge(mask, label, score, query, i)
                    is_merge = True
                    break
            # new ins
            if not is_merge:
                mask = torch.cat([mask.new_zeros(pts_per_frame*i).bool(), mask])
                cur_instances.append(InstanceQuery(mask, label, score, query))
        cur_points = torch.cat([cur_points, points[i]])
        # not merged ins
        for InsQ in cur_instances:
            if len(InsQ.mask) < cur_points.shape[0]:
                InsQ.pad(pts_per_frame)
    merged_mask = torch.stack([InsQ.mask for InsQ in cur_instances], dim=0)
    merged_labels = torch.tensor([InsQ.label for InsQ in cur_instances]).to(merged_mask.device)
    merged_scores = torch.tensor([InsQ.score for InsQ in cur_instances]).to(merged_mask.device)
    if len(merged_scores) > inscat_topk_insts:
        _, kept_ins = merged_scores.topk(inscat_topk_insts)
    else:
        kept_ins = ...
    merged_mask, merged_labels, merged_scores = \
        merged_mask[kept_ins], merged_labels[kept_ins], merged_scores[kept_ins]
    return merged_mask, merged_labels, merged_scores

class GTMerge():
    def __init__(self):
        self.cur_queries = None
        self.fi = 0
        self.merge_counts = None
    
    def clean(self):
        self.cur_queries = None
        self.merge_counts = None
    
    # weighted sum according to count of merge, rather than frame
    def merge(self, queries, cls_preds, query_ins_masks):
        batch_size = len(queries)
        ins_query_list = []
        merge_count_list = []
        # Intra-frame merge: choose one with max score
        for i in range(batch_size):
            n_instances = len(query_ins_masks[i])
            if n_instances == 0:
                return None
            ins_query = []
            merge_count = []
            for j in range(n_instances):
                temp_idx = query_ins_masks[i][j]
                # ins_query.append(queries[i][temp_idx].mean(0) if
                #      len(temp_idx) != 0 else torch.zeros_like(queries[i][0]))
                # merge_count.append(len(temp_idx))
                fg_scores = cls_preds[i][temp_idx].softmax(-1)[:,:-1].sum(-1)
                ins_query.append(queries[i][temp_idx][fg_scores.argmax()] if
                     len(temp_idx) != 0 else torch.zeros_like(queries[i][0]))
                merge_count.append(1 if len(temp_idx) != 0 else 0)
            ins_query_list.append(torch.stack(ins_query, dim=0))
            merge_count_list.append(torch.tensor(merge_count, device=temp_idx.device).unsqueeze(-1))
        if self.cur_queries is None:
            self.cur_queries = ins_query_list
            self.merge_counts = merge_count_list
        else:
            # Inter-frame merge: mean across frame
            for i in range(batch_size):
                # self.cur_queries[i] = (self.cur_queries[i] * self.fi + ins_query_list[i]) / (self.fi + 1)
                self.cur_queries[i] = (self.cur_queries[i] * self.merge_counts[i] + ins_query_list[i]
                     * merge_count_list[i]) / (self.merge_counts[i] + merge_count_list[i] + 1e-6)
                self.merge_counts[i] = self.merge_counts[i] + merge_count_list[i]
        output_queries = []
        for i in range(batch_size):
            output_queries.append(self.cur_queries[i][self.cur_queries[i].sum(-1) != 0])
        self.fi += 1
        return output_queries


class OnlineMerge():
    def __init__(self, inscat_topk_insts, use_bbox=False, merge_type="count"):
        assert merge_type in ['count', 'frame']
        self.merge_type = merge_type
        self.inscat_topk_insts = inscat_topk_insts
        self.use_bbox = use_bbox
        if self.use_bbox:
            self.iou_calculator = AxisAlignedBboxOverlaps3D()
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.fi = 0
        self.merge_counts = None
    
    def clean(self):
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.merge_counts = None
    
    def merge(self, masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes):
        points_per_mask = masks.shape[1]
        # masks, labels, scores, queries, query_feats, sem_preds, xyz_list = \
        #     self.intra_frame_merge(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, q)
        if self.cur_masks is None:
            self.cur_masks = masks
            self.cur_labels = labels
            self.cur_scores = scores
            self.cur_queries = queries
            self.cur_query_feats = query_feats
            self.cur_sem_preds = sem_preds
            self.cur_xyz = self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            self.merge_counts = torch.zeros_like(scores).long()
        else:
            self.fi += 1
            next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
                masks, labels, scores, queries, query_feats, sem_preds, \
                self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            query_feat_scores = (self.cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
            sem_pred_scores = F.cosine_similarity(self.cur_sem_preds.unsqueeze(1), next_sem_preds.unsqueeze(0), dim=2)
            if self.use_bbox:
                xyz_scores = self.iou_calculator(self.cur_xyz, next_xyz, is_aligned=False)
            else:
                xyz_dists = torch.cdist(self.cur_xyz, next_xyz, p=2)
                xyz_scores = 1 / (xyz_dists + 1e-6)
                        
            mix_scores = query_feat_scores * xyz_scores
            inst_label_scores = torch.where(self.cur_labels.unsqueeze(1) == next_labels.unsqueeze(0), torch.ones((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device), torch.zeros((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device))
            
            mix_scores = torch.where(mix_scores > 0, mix_scores, torch.zeros_like(mix_scores))
            mix_scores = mix_scores * inst_label_scores
            if mix_scores.shape[0] < mix_scores.shape[1]:
                mix_scores = torch.cat((mix_scores, torch.zeros((mix_scores.shape[1]
                     - mix_scores.shape[0], mix_scores.shape[1])).to(mix_scores.device)), dim=0)
            # Hungarian assign
            row_ind, col_ind = linear_sum_assignment(-mix_scores.cpu())
            row_ind = torch.tensor(row_ind).to(mix_scores.device)
            col_ind = torch.tensor(col_ind).to(mix_scores.device)
            mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
            row_ind = row_ind[mix_scores_mask]
            col_ind = col_ind[mix_scores_mask]

            temp = torch.zeros(self.cur_masks.shape[0]).bool().to(self.cur_masks.device)
            temp[row_ind] = True
            temp = temp.unsqueeze(1)
            temp_masks = torch.zeros((self.cur_masks.shape[0], points_per_mask)).bool().to(self.cur_masks.device)
            temp_masks[row_ind] = next_masks[col_ind]
            next_masks_ = torch.where(temp, temp_masks,
                                     torch.zeros((self.cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
            self.cur_masks = torch.cat((self.cur_masks, next_masks_), dim=1)
            no_merge_masks = torch.ones(next_masks.shape[0]).bool().to(next_masks.device)
            no_merge_masks[col_ind] = False
            former_padding = torch.zeros((no_merge_masks.nonzero().shape[0], points_per_mask * self.fi)).bool().to(next_masks.device)
            new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
            self.cur_masks = torch.cat((self.cur_masks, new_masks), dim=0)

            self.merge_counts[row_ind] += 1
            if len(no_merge_masks) > 0:
                self.merge_counts = torch.cat([self.merge_counts,
                     torch.zeros(no_merge_masks.shape[0]).long().to(self.merge_counts.device)], dim=0)
            
            if self.merge_type == 'count':
                count = self.merge_counts[row_ind]
            else: count = self.fi
            
            self.cur_scores[row_ind] = (self.cur_scores[row_ind] * count + next_scores[col_ind]) / (count + 1)
            self.cur_scores = torch.cat((self.cur_scores, next_scores[no_merge_masks]), dim=0)
            if self.merge_type == 'count':
                count = count.unsqueeze(-1)
            self.cur_labels = torch.cat((self.cur_labels, next_labels[no_merge_masks]), dim=0)
            self.cur_queries[row_ind] = (self.cur_queries[row_ind] * count + next_queries[col_ind]) / (count + 1)
            self.cur_queries = torch.cat((self.cur_queries, next_queries[no_merge_masks]), dim=0)
            self.cur_query_feats[row_ind] = (self.cur_query_feats[row_ind] * count + next_query_feats[col_ind]) / (count + 1)
            self.cur_query_feats = torch.cat((self.cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
            self.cur_sem_preds[row_ind] = (self.cur_sem_preds[row_ind] * count + next_sem_preds[col_ind]) / (count + 1)
            self.cur_sem_preds = torch.cat((self.cur_sem_preds, next_sem_preds[no_merge_masks]), dim=0)
            self.cur_xyz[row_ind] = (self.cur_xyz[row_ind] * count + next_xyz[col_ind]) / (count + 1)
            self.cur_xyz = torch.cat((self.cur_xyz, next_xyz[no_merge_masks]), dim=0)
            
        if len(self.cur_scores) > self.inscat_topk_insts:
            _, kept_ins = self.cur_scores.topk(self.inscat_topk_insts)
        else:
            kept_ins = ...
        cur_masks, cur_scores = self.cur_masks[kept_ins], self.cur_scores[kept_ins]
        cur_labels = self.cur_labels[kept_ins]
        cur_queries = self.cur_queries[kept_ins]
        cur_bboxes = self.cur_xyz[kept_ins] if self.use_bbox else None
        # cur_labels = torch.zeros_like(self.cur_scores).long()
        return cur_masks, cur_labels, cur_scores, cur_queries, cur_bboxes
    
    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


class InstanceQuery():
    def __init__(self, mask, label, score, query):
        self.mask = mask
        self.label = label
        self.score = score
        self.query = query
        self.merge_count = 1
    
    def pad(self, pts_num):
        self.mask = torch.cat([self.mask, self.mask.new_zeros(pts_num).bool()])
    
    def compare(self, cur_points, points, mask, label, score, query, pts_thr=0.05, thr=0.1):
        if cur_points.shape[0] != len(self.mask):
            return False
        if self.label != label:
            return False
        cur_xyz = cur_points[self.mask, :3].unsqueeze(1) # Mx3
        if cur_xyz.shape[0] > 10000:
            sample_idx = torch.randperm(cur_xyz.shape[0])[:10000]
            cur_xyz = cur_xyz[sample_idx]
        xyz = points[mask, :3].unsqueeze(0) # Nx3
        if xyz.shape[0] > 10000:
            sample_idx = torch.randperm(xyz.shape[0])[:10000]
            xyz = xyz[sample_idx]
        dist_mat = cur_xyz - xyz # MxNx3
        dist_mat = (dist_mat ** 2).sum(-1).sqrt() # MxN
        min_dist1 = dist_mat.min(-1).values # M
        min_dist2 = dist_mat.min(0).values # N
        ratio1 = (min_dist1 < pts_thr).sum() / len(min_dist1)
        ratio2 = (min_dist2 < pts_thr).sum() / len(min_dist2)
        if max(ratio1, ratio2) > thr:
            return True
        else:
            return False
    
    def merge(self, mask, label, score, query, frame_i):
        self.mask = torch.cat([self.mask, mask])
        self.score = (self.score * frame_i + score) / (frame_i + 1)
        self.query = (self.query * frame_i + query) / (frame_i + 1)
        self.merge_count += 1

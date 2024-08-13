from mmdet3d.registry import MODELS
from mmengine.structures import InstanceData
import torch


@MODELS.register_module()
class ScanNetUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)
    
    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData with
                    `sp_masks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes

        for i in range(len(pred['masks'])):
            sem_gt = InstanceData()
            if insts[i].get('query_masks') is not None:
                sem_gt.sp_masks = insts[i].query_masks[-n - 1:, :]
            else:
                sem_gt.sp_masks = insts[i].sp_masks[-n - 1:, :]
            sem_gts.append(sem_gt)

            inst_gt = InstanceData()
            inst_gt.sp_masks = insts[i].sp_masks[:-n - 1, :]
            inst_gt.labels_3d = insts[i].labels_3d[:-n - 1]
            if insts[i].get('query_masks') is not None:
                inst_gt.query_masks = insts[i].query_masks[:-n - 1, :]
            inst_gts.append(inst_gt)
        
        loss = self.inst_criterion(pred, inst_gts)
        loss.update(self.sem_criterion(pred, sem_gts))
        return loss


@MODELS.register_module()
class ScanNetMixedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)
    
    def __call__(self, pred, insts, insts_pts, sp_xyz_list, mask_pred_mode):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData with
                    `sp_masks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes

        pred['centers'] = sp_xyz_list
        for i in range(len(pred['aux_outputs'])):
            pred['aux_outputs'][i]['centers'] = sp_xyz_list

        for i in range(len(pred['masks'])):
            sem_gt = InstanceData()
            if insts[i].get('query_masks') is not None:
                sem_gt.sp_masks = insts[i].query_masks[-n - 1:, :]
            else:
                sem_gt.sp_masks = insts[i].sp_masks[-n - 1:, :]
            sem_gts.append(sem_gt)

            # point-level gt_mask
            inst_gt = InstanceData()
            inst_gt.p_masks = insts_pts[i].p_masks
            inst_gt.sp_masks = insts[i].sp_masks[:-n - 1, :]
            if pred['cls_preds'][i].shape[1] == 2:
                # category agnostic
                inst_gt.labels_3d = torch.zeros_like(insts[i].labels_3d[:-n - 1])
            else:
                inst_gt.labels_3d = insts[i].labels_3d[:-n - 1]
            if 'bboxes_3d' in insts[i].keys():
                inst_gt.bboxes_3d = insts[i].bboxes_3d[:-n - 1, :]
            if insts[i].get('query_masks') is not None:
                inst_gt.query_masks = insts[i].query_masks[:-n - 1, :]
            inst_gts.append(inst_gt)
        
        loss = self.inst_criterion(pred, inst_gts, mask_pred_mode)
        loss.update(self.sem_criterion(pred, sem_gts))
        return loss


@MODELS.register_module()
class S3DISUnifiedCriterion:
    # Is it the same as ScanNetUnifiedCriterion?
    pass

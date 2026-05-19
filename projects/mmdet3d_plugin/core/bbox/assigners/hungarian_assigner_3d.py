import torch

import types
# from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from mmengine.structures import InstanceData
from mmdet.models.task_modules import BBOX_ASSIGNERS

from mmdet.models.task_modules import build_match_cost
from mmdet.models.layers import inverse_sigmoid

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range

    # def assign(self,
    #            bbox_pred,
    #            cls_pred,
    #            gt_bboxes, 
    #            gt_labels,
    #            gt_bboxes_ignore=None,
    #            eps=1e-7):
    #     """Computes one-to-one matching based on the weighted costs.
    #     This method assign each query prediction to a ground truth or
    #     background. The `assigned_gt_inds` with -1 means don't care,
    #     0 means negative sample, and positive number is the index (1-based)
    #     of assigned gt.
    #     The assignment is done in the following steps, the order matters.
    #     1. assign every prediction to -1
    #     2. compute the weighted costs
    #     3. do Hungarian matching on CPU based on the costs
    #     4. assign all to 0 (background) first, then for each matched pair
    #        between predictions and gts, treat this prediction as foreground
    #        and assign the corresponding gt index (plus 1) to it.
    #     Args:
    #         bbox_pred (Tensor): Predicted boxes with normalized coordinates
    #             (cx, cy, w, h), which are all in range [0, 1]. Shape
    #             [num_query, 4].
    #         cls_pred (Tensor): Predicted classification logits, shape
    #             [num_query, num_class].
    #         gt_bboxes (Tensor): Ground truth boxes with unnormalized
    #             coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
    #         gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
    #         gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
    #             labelled as `ignored`. Default None.
    #         eps (int | float, optional): A value added to the denominator for
    #             numerical stability. Default 1e-7.
    #     Returns:
    #         :obj:`AssignResult`: The assigned result.
    #     """
    #     assert gt_bboxes_ignore is None, \
    #         'Only case when gt_bboxes_ignore is None is supported.'
    #     num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

    #     # 1. assign -1 by default
    #     assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
    #                                           -1,
    #                                           dtype=torch.long)
    #     assigned_labels = bbox_pred.new_full((num_bboxes, ),
    #                                          -1,
    #                                          dtype=torch.long)
    #     if num_gts == 0 or num_bboxes == 0:
    #         # No ground truth or boxes, return empty assignment
    #         if num_gts == 0:
    #             # No ground truth, assign all to background
    #             assigned_gt_inds[:] = 0
    #         return AssignResult(
    #             num_gts, assigned_gt_inds, None, labels=assigned_labels)

    #     # 2. compute the weighted costs
    #     # classification and bboxcost.
    #     cls_cost = self.cls_cost(cls_pred, gt_labels)
    #     # regression L1 cost
       
    #     normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
    
    #     reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
      
    #     # weighted sum of above two costs
    #     # --- MMDetection 3.x / MMEngine Core Compatibility Patch ---
    #     # Convert raw tensor predictions into InstanceData containers 
    #     # to prevent 'Tensor object has no attribute scores' crashes.
        
    #     if isinstance(cls_pred, torch.Tensor):
    #         # Create the data container abstraction expected by MMDet 3.x match costs
    #         pred_instances = InstanceData()
    #         pred_instances.scores = cls_pred
            
    #         if isinstance(bbox_pred, torch.Tensor):
    #             pred_instances.bboxes = bbox_pred
    #     else:
    #         pred_instances = cls_pred

    #     # Evaluate cost using our wrapped instances
    #     cls_cost = self.cls_cost(pred_instances, gt_labels)
        
    #     # 3. do Hungarian matching on CPU using linear_sum_assignment
    #     cost = cost.detach().cpu()
    #     if linear_sum_assignment is None:
    #         raise ImportError('Please run "pip install scipy" '
    #                           'to install scipy first.')
    #     matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    #     matched_row_inds = torch.from_numpy(matched_row_inds).to(
    #         bbox_pred.device)
    #     matched_col_inds = torch.from_numpy(matched_col_inds).to(
    #         bbox_pred.device)

    #     # 4. assign backgrounds and foregrounds
    #     # assign all indices to backgrounds first
    #     assigned_gt_inds[:] = 0
    #     # assign foregrounds based on matching results
    #     assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    #     assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    #     return AssignResult(
    #         num_gts, assigned_gt_inds, None, labels=assigned_labels)


    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes, 
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs."""
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. Assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. MMDetection 3.x / MMEngine Core Compatibility Wrap
        # Wrap raw prediction tensors into an InstanceData container *BEFORE* calling cost submodules
        pred_instances = InstanceData()
        pred_instances.scores = cls_pred
        pred_instances.bboxes = bbox_pred

        gt_instances = InstanceData()
        gt_instances.labels = gt_labels.long() 
        gt_instances.bboxes = gt_bboxes

        # 3. Compute normalized ground truth bboxes
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)

        # 4. Compute individual weighted cost matrices
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        
        # 5. Aggregate all costs into the final cost matrix
        # (This combines classification and bounding box regression costs)
        cost = cls_cost + reg_cost

        # 6. Do Hungarian matching on CPU using linear_sum_assignment
        cost_cpu = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')
            
        if torch.isnan(cost_cpu).any() or torch.isinf(cost_cpu).any():
            # Replace NaNs with a massive cost value so the assigner safely skips them
            cost_cpu = torch.where(torch.isnan(cost_cpu), torch.tensor(1e5, dtype=cost_cpu.dtype), cost_cpu)
            # Replace Infs with a massive finite cost value
            cost_cpu = torch.where(torch.isinf(cost_cpu), torch.tensor(1e5, dtype=cost_cpu.dtype), cost_cpu)
            
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_cpu)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 7. Assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds].long()
        
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
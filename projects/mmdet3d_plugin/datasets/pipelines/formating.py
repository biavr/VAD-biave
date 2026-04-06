
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.dataset import Compose
from mmengine.structures import InstanceData

from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.structures.points import BasePoints
# from mmdet.datasets.builder import PIPELINES
from mmdet.registry import TRANSFORMS
from mmcv.transforms import to_tensor
from mmdet3d.datasets.transforms import Pack3DDetInputs

@TRANSFORMS.register_module()
class CustomDefaultFormatBundle3D(Pack3DDetInputs):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """
    def __init__(self, class_names, with_gt=True, with_label=True, with_ego=True):
        # super(CustomDefaultFormatBundle3D, self).__init__(class_names, with_gt, with_label)
        super(CustomDefaultFormatBundle3D, self).__init__()
        self.with_ego = with_ego


    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        # results['gt_map_masks'] = DC(to_tensor(results['gt_map_masks']), stack=True)
        if self.with_ego:
            if 'ego_his_trajs' in results:
                # results['ego_his_trajs'] = DC(to_tensor(results['ego_his_trajs'][None, ...]), stack=True)
                results['ego_his_trajs'] = to_tensor(results['ego_his_trajs'][None, ...])
            if 'ego_fut_trajs' in results:
                # results['ego_fut_trajs'] = DC(to_tensor(results['ego_fut_trajs'][None, ...]), stack=True)
                results['ego_fut_trajs'] = to_tensor(results['ego_fut_trajs'][None, ...])
            if 'ego_fut_masks' in results:
                # results['ego_fut_masks'] = DC(to_tensor(results['ego_fut_masks'][None, None, ...]), stack=True)
                results['ego_fut_masks'] = to_tensor(results['ego_fut_masks'][None, ...])
            if 'ego_fut_cmd' in results:
                results['ego_fut_cmd'] = to_tensor(results['ego_fut_cmd'][None, ...])
            if 'ego_lcf_feat' in results:
                results['ego_lcf_feat'] = to_tensor(results['ego_lcf_feat'][None, ...])
            if 'gt_attr_labels' in results:
                results['gt_attr_labels'] = to_tensor(results['gt_attr_labels'])
                
        return results
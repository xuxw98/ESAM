from ultralytics import YOLO
from PIL import Image
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import pdb
from ultralytics.yolo.utils import is_git_dir
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.v8.segment import SegmentationPredictor
from ultralytics.yolo.cfg import get_cfg
import sys
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

class MyYOLO(YOLO):
    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = MyPredictor(overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
            if 'project' in overrides or 'name' in overrides:
                self.predictor.save_dir = self.predictor.get_save_dir()
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)


class MyPredictor(SegmentationPredictor):
    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""

        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment)
        return preds

@MODELS.register_module()
class FastSAM_Backbone(BaseModule):
    def __init__(self):
        super(FastSAM_Backbone, self).__init__()
        self.yolo = MyYOLO('/home/ubuntu/xxw/OS3D/FastSAM/FastSAM-x.pt')
        self.yolo.model.model = self.yolo.model.model[:15]

    def init_weights(self):
        for param in self.yolo.model.parameters():
            param.requires_grad = False
        self.yolo.model.eval()

    def forward(self, x):
        x = self.yolo.predict(x, device='cuda', retina_masks=True, imgsz=640)
        return x



'''core utils'''
from mmdet3d.structures.bbox_3d import points_cam2img, rotation_3d_in_axis
from functools import partial
from mmdet3d.structures.points import CameraPoints, LiDARPoints, DepthPoints
from abc import abstractmethod
import warnings
import numpy as np
import os 
import csv

def get_points_type(points_type):
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate.
            The valid value are "CAMERA", "LIDAR", or "DEPTH".

    Returns:
        class: Points type.
    """
    if points_type == 'CAMERA':
        points_cls = CameraPoints
    elif points_type == 'LIDAR':
        points_cls = LiDARPoints
    elif points_type == 'DEPTH':
        points_cls = DepthPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR", or "DEPTH"'
                         f' are supported, got {points_type}')

    return points_cls

def apply_3d_transformation(pcd, coord_type, img_meta, reverse=False):
    """Apply transformation to input point cloud.

    Args:
        pcd (torch.Tensor): The point cloud to be transformed.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.

    Note:
        The elements in img_meta['transformation_3d_flow']:
        "T" stands for translation;
        "S" stands for scale;
        "R" stands for rotation;
        "HF" stands for horizontal flip;
        "VF" stands for vertical flip.

    Returns:
        torch.Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device

    # breakpoint()
    pcd_rotate_mat = (
        #torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
        img_meta['pcd_rotation'].clone().detach().to(device,dtype)
        if 'pcd_rotation' in img_meta else torch.eye(
            3, dtype=dtype, device=device))

    pcd_scale_factor = (
        img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (
        torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
        if 'pcd_trans' in img_meta else torch.zeros(
            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coord_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord

def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)
    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat)
    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset
    # pdb.set_trace()
    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()

def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 5).
        labels (torch.Tensor): Labels with shape (N, ).
        scores (torch.Tensor): Scores with shape (N, ).
        attrs (torch.Tensor, optional): Attributes with shape (N, ).
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

def read_label_mapping(filename,
                       label_from='raw_category',
                       label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


# yolo = MyYOLO('weights/FastSAM-x.pt')
# yolo.model.model = yolo.model.model[:15]
# embedding = yolo.predict('images/0.jpg', device='cuda', retina_masks=True, imgsz=640)[0]
# print(embedding.shape)

#embedding = yolo.model(tensor_img)
#print(embedding.shape)

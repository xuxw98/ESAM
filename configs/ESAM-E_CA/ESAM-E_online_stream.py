_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['oneformer3d'])

num_instance_classes = 1
num_semantic_classes = 200
num_instance_classes_eval = 1
use_bbox = True
voxel_size = 0.02

model = dict(
    type='ScanNet200MixFormer3D_Stream',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    voxel_size=voxel_size,
    num_classes=num_instance_classes_eval,
    query_thr=0.5,
    backbone=dict(
        type='Res16UNet34C',
        in_channels=3,
        out_channels=96,
        config=dict(
            dilations=[1, 1, 1, 1],
            conv1_kernel_size=5,
            bn_momentum=0.02)),
    memory=dict(type='MultilevelMemory', in_channels=[32, 64, 128, 256], queue=-1, vmp_layer=(0,1,2,3)),
    pool=dict(type='GeoAwarePooling', channel_proj=96),
    decoder=dict(
        type='ScanNetMixQueryDecoder',
        num_layers=3,
        share_attn_mlp=False, 
        share_mask_mlp=False,
        temporal_attn=False,
        # the last mp_mode should be "P"
        cross_attn_mode=["", "SP", "SP", "SP"], 
        mask_pred_mode=["SP", "SP", "P", "P"],
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=96,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False,
        bbox_flag=use_bbox),
    merge_head=dict(type='MergeHead', in_channels=256, out_channels=256, norm='layer'),
    merge_criterion=dict(type='ScanNetMergeCriterion_Fast', tmp=True, p2s=False),
    criterion=dict(
        type='ScanNetMixedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.5),
        inst_criterion=dict(
            type='MixedInstanceCriterion',
            matcher=dict(
                type='SparseMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)],
                topk=1),
            bbox_loss=dict(type='AxisAlignedIoULoss'),
            loss_weight=[0.5, 1.0, 1.0, 0.5, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=None,
    test_cfg=dict(
        # TODO: a larger topK may be better
        topk_insts=20,
        inscat_topk_insts=200,
        inst_score_thr=0.3,     # modified from 0.3
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1],
        merge_type='learnable_online'))

color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

# dataset settings
train_pipeline = None
test_pipeline = None

train_dataloader = None
val_dataloader =None
test_dataloader = None

val_evaluator = None
test_evaluator = None

optim_wrapper = None

# learning rate
param_scheduler = None

custom_hooks = None
default_hooks = None

# training schedule for 1x
train_cfg = None
val_cfg = None
test_cfg = None

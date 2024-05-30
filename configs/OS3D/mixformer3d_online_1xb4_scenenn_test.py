_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['oneformer3d'])

num_instance_classes = 18
num_semantic_classes = 20
num_instance_classes_eval = 18
use_bbox = True

model = dict(
    type='ScanNet200MixFormer3D_Online',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    voxel_size=0.02,
    num_classes=num_instance_classes_eval,
    query_thr=0.5,
    weights=[0.33,0.33,0.33],
    thresh=0.55,
    backbone=dict(
        type='Res16UNet34C',
        in_channels=3,
        out_channels=96,
        config=dict(
            dilations=[1, 1, 1, 1],
            conv1_kernel_size=5,
            bn_momentum=0.02)),
    memory=dict(type='MultilevelMemory', in_channels=[32, 64, 128, 256], queue=-1, vmp_layer=(0,1,2,3)),
    # memory=dict(type='MultilevelMemory', in_channels=[32, 64, 128, 256], queue=-1, vmp_layer=(2,3)),
    pool=dict(type='GeoAwarePooling', channel_proj=96),
    decoder=dict(
        type='ScanNetMixQueryDecoder',
        num_layers=3,
        share_attn_mlp=False, 
        share_mask_mlp=False,
        temporal_attn=False, # TODO: to be extended
        # the last mp_mode should be "P"
        cross_attn_mode=["", "SP", "SP", "SP"], 
        mask_pred_mode=["P", "P", "P", "P"],
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
    train_cfg=dict(),
    test_cfg=dict(
        # TODO: a larger topK may be better
        topk_insts=20,
        inscat_topk_insts=100,
        inst_score_thr=0.3,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1],
        merge_type='learnable_online'))

# TODO: complete the dataset
dataset_type = 'ScanNetSegMVDataset_'
data_root = 'data/scenenn-mv/'

# floor and chair are changed
class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
    'bathtub', 'otherfurniture']

color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

# dataset settings
test_pipeline = [
    dict(
        type='LoadAdjacentDataFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        num_frames=-1,
        num_sample=20000,
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True,
        with_rec=True,
        dataset_type = 'scenenn'),
    dict(type='PointSegClassMappingWithRec'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=color_mean,
                color_std=color_std),
            dict(
                type='AddSuperPointAnnotations_Online',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False,
                with_rec=True),
        ]),
    dict(type='Pack3DDetInputs_Online', keys=['points', 'sp_pts_mask'] + ['gt_labels_3d'])
]

val_dataloader = dict(
    persistent_workers=False,
    num_workers=0,
    dataset=dict(
        type=dataset_type,
        ann_file='scenenn_mv_oneformer3d_infos_val.pkl',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))
test_dataloader = val_dataloader

label2cat = {i: name for i, name in enumerate(class_names + ['unlabeled'])}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names + ['unlabeled'])

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
inst_mapping = sem_mapping[2:]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0, 1], 
    thing_class_inds=list(range(2, num_semantic_classes)),
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

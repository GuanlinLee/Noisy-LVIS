_base_ = './mask_rcnn_r50_fpn_rfs_seesaw_loss_mstrain_2x_lvis_v1.py'
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))


dataset_type = 'LVISV1Dataset'
data_root = '/mnt/lustre/glli/code/noisy_long_tail_instance_seg/GOL/datasets/coco/'
p = 0.8
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.001,
        dataset=dict(
            type='LVISV1Dataset',
            ann_file='annotations/lvis_v1_train_SN_%f.json'%p,
            img_prefix='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=False),
                dict(
                    type='Resize',
                    img_scale=[(1333, 640), (1333, 672), (1333, 704),
                               (1333, 736), (1333, 768), (1333, 800)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ],
            data_root=
            '/mnt/lustre/glli/code/noisy_long_tail_instance_seg/GOL/datasets/coco'
        )),
    val=dict(
        type='LVISV1Dataset',
        ann_file=
        '/mnt/lustre/glli/code/noisy_long_tail_instance_seg/GOL/datasets/coco/annotations/lvis_v1_val.json',
        img_prefix=
        '/mnt/lustre/glli/code/noisy_long_tail_instance_seg/GOL/datasets/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LVISV1Dataset',
        ann_file=
        '/mnt/lustre/glli/code/noisy_long_tail_instance_seg/GOL/datasets/coco/annotations/lvis_v1_val.json',
        img_prefix=
        '/mnt/lustre/glli/code/noisy_long_tail_instance_seg/GOL/datasets/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox', 'segm'], interval=24)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
cfg_root = '../lvis/'
work_dir = 'experiments/mask_rcnn_r50_fpn_rfs_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_SN_08/'
auto_resume = False
gpu_ids = range(0, 4)
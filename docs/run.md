## Train and Evaluation

Train and evaluate OS3D on ScanNet200-SV：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mixformer3d_sv_1xb4_scannet200.py --work-dir work_dirs/mixformer3d_sv_1xb4_scannet200/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mixformer3d_sv_1xb4_scannet200.py work_dirs/work_dirs/mixformer3d_sv_1xb4_scannet200/epoch_128.pth --work-dir work_dirs/mixformer3d_sv_1xb4_scannet200/
```

Train and evaluate OS3D on ScanNet200-MV：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mixformer3d_online_1xb4_scannet200.py --work-dir work_dirs/mixformer3d_online_1xb4_scannet200/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mixformer3d_online_1xb4_scannet200.py work_dirs/work_dirs/mixformer3d_online_1xb4_scannet200/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_scannet200/
```

Evaluate OS3D on SceneNN-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mixformer3d_online_1xb4_scenenn_CA.py work_dirs/work_dirs/mixformer3d_online_1xb4_scannet200/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_scenenn_CA/
```

Evaluate OS3D on 3RScan-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mixformer3d_online_1xb4_3rscan_CA.py work_dirs/work_dirs/mixformer3d_online_1xb4_scannet200/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_3rscan_CA/
```

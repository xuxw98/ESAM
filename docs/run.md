## Train and Evaluation

Train and evaluate OS3D on ScanNet200-SV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/OS3D_CA/mixformer3d_sv_1xb4_scannet200_CA.py --work-dir work_dirs/mixformer3d_sv_1xb4_scannet200_CA/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D_CA/mixformer3d_sv_1xb4_scannet200_CA.py work_dirs/mixformer3d_sv_1xb4_scannet200_CA/epoch_128.pth --work-dir work_dirs/mixformer3d_sv_1xb4_scannet200_CA/
```

Train and evaluate OS3D on ScanNet200-MV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/OS3D_CA/mixformer3d_online_1xb4_scannet200_CA.py --work-dir work_dirs/mixformer3d_online_1xb4_scannet200_CA/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D_CA/mixformer3d_online_1xb4_scannet200_CA.py work_dirs/mixformer3d_online_1xb4_scannet200_CA/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_scannet200_CA/
```
Train and evaluate OS3D on ScanNet-SV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/OS3D/mixformer3d_sv_1xb4_scannet.py --work-dir work_dirs/mixformer3d_sv_1xb4_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D/mixformer3d_sv_1xb4_scannet.py work_dirs/mixformer3d_sv_1xb4_scannet/epoch_128.pth --work-dir work_dirs/mixformer3d_sv_1xb4_scannet/
```

Train and evaluate OS3D on ScanNet-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/OS3D/mixformer3d_online_1xb4_scannet.py --work-dir work_dirs/mixformer3d_online_1xb4_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D/mixformer3d_online_1xb4_scannet.py work_dirs/mixformer3d_online_1xb4_scannet/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_scannet/
```


Evaluate OS3D on SceneNN-MV (Class Agnostic):

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D_CA/mixformer3d_online_1xb4_scenenn_CA_test.py work_dirs/mixformer3d_online_1xb4_scannet200_CA/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_scenenn_CA_test/
```

Evaluate OS3D on SceneNN-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D/mixformer3d_online_1xb4_scenenn_test.py work_dirs/mixformer3d_online_1xb4_scannet/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_scenenn_test/
```


Evaluate OS3D on 3RScan-MV (Class Agnostic):

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/OS3D_CA/mixformer3d_online_1xb4_3rscan_CA_test.py work_dirs/mixformer3d_online_1xb4_scannet200_CA/epoch_128.pth --work-dir work_dirs/mixformer3d_online_1xb4_3rscan_CA_test/
```


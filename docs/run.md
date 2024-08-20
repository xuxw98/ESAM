## Train and Evaluation

### Class-agnostic 3D instance segmentation on ScanNet200:

#### ESAM：
Train and evaluate ESAM on ScanNet200-SV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM_CA/ESAM_sv_scannet200_CA.py --work-dir work_dirs/ESAM_sv_scannet200_CA/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM_CA/ESAM_sv_scannet200_CA.py work_dirs/ESAM_sv_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM_sv_scannet200_CA/
```

Train and evaluate ESAM on ScanNet200-MV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM_CA/ESAM_online_scannet200_CA.py --work-dir work_dirs/ESAM_online_scannet200_CA/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM_CA/ESAM_online_scannet200_CA.py work_dirs/ESAM_online_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM_online_scannet200_CA/
```

#### ESAM-E：
Train and evaluate ESAM-E on ScanNet200-SV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM-E_CA/ESAM-E_sv_scannet200_CA.py --work-dir work_dirs/ESAM-E_sv_scannet200_CA/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E_CA/ESAM-E_sv_scannet200_CA.py work_dirs/ESAM-E_sv_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM-E_sv_scannet200_CA/
```

Train and evaluate ESAM-E on ScanNet200-MV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM-E_CA/ESAM-E_online_scannet200_CA.py --work-dir work_dirs/ESAM-E_online_scannet200_CA/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E_CA/ESAM-E_online_scannet200_CA.py work_dirs/ESAM-E_online_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM-E_online_scannet200_CA/
```

### Class-agnostic 3D instance segmentation on SceneNN and 3RScan:

#### ESAM：

Evaluate ESAM on SceneNN-MV (Class Agnostic):

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM_CA/ESAM_online_scenenn_CA_test.py work_dirs/ESAM_online_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM_online_scenenn_CA_test/
```

Evaluate ESAM on 3RScan-MV (Class Agnostic):

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM_CA/ESAM_online_3rscan_CA_test.py work_dirs/ESAM_online_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM_online_3rscan_CA_test/
```

#### ESAM-E：
Evaluate ESAM-E on SceneNN-MV (Class Agnostic):

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E_CA/ESAM-E_online_scenenn_CA_test.py work_dirs/ESAM-E_online_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM-E_online_scenenn_CA_test/
```

Evaluate ESAM-E on 3RScan-MV (Class Agnostic):

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E_CA/ESAM-E_online_3rscan_CA_test.py work_dirs/ESAM-E_online_scannet200_CA/epoch_128.pth --work-dir work_dirs/ESAM-E_online_3rscan_CA_test/
```

### Class-aware 3D instance segmentation on ScanNet:
#### ESAM：
Train and evaluate ESAM on ScanNet-SV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM/ESAM_sv_scannet.py --work-dir work_dirs/ESAM_sv_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM/ESAM_sv_scannet.py work_dirs/ESAM_sv_scannet/epoch_128.pth --work-dir work_dirs/ESAM_sv_scannet/
```

Train and evaluate ESAM on ScanNet-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM/ESAM_online_scannet.py --work-dir work_dirs/ESAM_online_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM/ESAM_online_scannet.py work_dirs/ESAM_online_scannet/epoch_128.pth --work-dir work_dirs/ESAM_online_scannet/
```

#### ESAM-E：
Train and evaluate ESAM-E on ScanNet-SV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM-E/ESAM-E_sv_scannet.py --work-dir work_dirs/ESAM-E_sv_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E/ESAM-E_sv_scannet.py work_dirs/ESAM-E_sv_scannet/epoch_128.pth --work-dir work_dirs/ESAM-E_sv_scannet/
```

Train and evaluate ESAM-E on ScanNet-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM-E/ESAM-E_online_scannet.py --work-dir work_dirs/ESAM-E_online_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E/ESAM-E_online_scannet.py work_dirs/ESAM-E_online_scannet/epoch_128.pth --work-dir work_dirs/ESAM-E_online_scannet/
```

#### ESAM-E_FF:
Train and evaluate ESAM-E_FF on ScanNet-SV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM-E_FF/ESAM-E_FF_sv_scannet.py --work-dir work_dirs/ESAM-E_FF_sv_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E_FF/ESAM-E_FF_sv_scannet.py work_dirs/ESAM-E_FF_sv_scannet/epoch_128.pth --work-dir work_dirs/ESAM-E_FF_sv_scannet/
```

Train and evaluate ESAM-E_FF on ScanNet-MV:

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ESAM-E_FF/ESAM-E_FF_online_scannet.py --work-dir work_dirs/ESAM-E_FF_online_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ESAM-E_FF/ESAM-E_FF_online_scannet.py work_dirs/ESAM-E_FF_online_scannet/epoch_128.pth --work-dir work_dirs/ESAM-E_FF_online_scannet/
```

### Open-Vocabulary 3D instance segmentation:
Our model can propose accurate class-agnostic 3D instance masks, which can be fed to open-vocabulary mask classification model like [OpenMask3D](https://github.com/OpenMask3D/openmask3d) to get open-vocabulary 3D segmentation results.

We follow the codebase of SAI3D to adopt this method, please refer to [SAI3D](https://github.com/yd-yin/SAI3D) for more details. Note that you only need to replace the instance masks with the results of ESAM or ESAM-E.
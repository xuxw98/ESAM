# OS3D:  **Online Segment Anything in 3D Scenes**

## Introduction

This repo contains PyTorch implementation for paper OS3D:  Online Segment Anything in 3D Scenes based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

We propose a VFM-assisted 3D instance segmentation framework namely OS3D, which exploits the power of SAM to online segment anything in 3D scenes with high accuracy and fast speed.

## Installation

This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework `v1.4.0`. Please follow [here](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md) for environment setup.

## Dataset Preparation

#### ScanNet200: 

For ScanNet200，follow [here](./data/scannet200/README.md).

For ScanNet200-SV, [download](https://github.com/ScanNet/ScanNet) '2D' and '3D' folders to the folder 'data/scannet200-sv', then  run: 

```
python load_scannet_sv_data_v2.py
cd ../..
python tools/create_data.py scannet200_sv --root-path ./data/scannet200-sv --out-dir ./data/scannet200-sv --extra-tag scannet200_sv
```

For ScanNet200-MV, link '2D' and '3D' folders to the folder 'data/scannet200-mv', then  run: 

```
python load_scannet_mv_data.py
cd ../..
python tools/create_data.py scannet200_mv --root-path ./data/scannet200-mv --out-dir ./data/scannet200-mv --extra-tag scannet200_mv
```

#### SceneNN:

The processed SceneNN data can be downloaded from the repo of [Online3D](https://cloud.tsinghua.edu.cn/d/641cd2b7a123467d98a6/). Run `cat SceneNN.tar.* > SceneNN.tar` to merge the files. Then unzip 'SceneNN.tar' to get 'SceneNN' folder.

For SceneNN, link 'SceneNN' folder to the folder 'data/scenenn', then run:

```
python batch_load_scenenn_data.py
cd ../..
python tools/create_data.py scenenn --root-path ./data/scenenn --out-dir ./data/scenenn --extra-tag scenenn
```

For SceneNN-MV, link 'SceneNN' folder to the folder 'data/scenenn-mv', then run:

```
python load_scenenn_mv_data.py
cd ../..
python tools/create_data.py scenenn_mv --root-path ./data/scenenn-mv --out-dir ./data/scenenn-mv --extra-tag scenenn_mv
```

#### 3RScan:

Download 3RScan dataset from [here](https://github.com/WaldJohannaU/3RScan?tab=readme-ov-file). You can acquire '3RScan' folder which contain several scenes renamed as ['000', '001', ...].

For 3RScan, link '3RScan' folder to the folder 'data/3RScan', then run:

```
python batch_load_3rscan_data.py
cd ../..
python tools/create_data.py 3rscan --root-path ./data/3RScan --out-dir ./data/3RScan --extra-tag 3rscan
```

For 3RScan-MV, link '3RScan' folder to the folder 'data/3RScan-mv', then run:

```
python load_3rscan_mv_data.py
cd ../..
python tools/create_data.py 3rscan_mv --root-path ./data/3RScan-mv --out-dir ./data/3RScan-mv --extra-tag 3rscan_mv
```

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

## Acknowledgement
We thank a lot for the flexible codebase of [Oneformer3D](https://github.com/oneformer3d/oneformer3d) and [Online3D](https://github.com/xuxw98/Online3D), as well as the valuable datasets provided by [ScanNet](https://github.com/ScanNet/ScanNet), [SceneNN](https://github.com/hkust-vgd/scenenn) and [3RScan](https://github.com/WaldJohannaU/3RScan).
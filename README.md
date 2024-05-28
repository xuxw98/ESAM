# OS3D:  **Online Segment Anything in 3D Scenes**

## Introduction

This repo contains PyTorch implementation for paper OS3D: Online Segment Anything in 3D Scenes based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

We propose a VFM-assisted 3D instance segmentation framework namely OS3D, which exploits the power of SAM to online segment anything in 3D scenes with high accuracy and fast speed.

## Getting Started
For environment setup and dataset preparation, please follow:
* [Installation](./docs/installation.md)
* [Dataset Preparation](./docs/dataset_preparation.md)

For training and evaluation, please follow:
* [Train and Evaluation](./docs/run.md)


## Acknowledgement
We thank a lot for the flexible codebase of [Oneformer3D](https://github.com/oneformer3d/oneformer3d) and [Online3D](https://github.com/xuxw98/Online3D), as well as the valuable datasets provided by [ScanNet](https://github.com/ScanNet/ScanNet), [SceneNN](https://github.com/hkust-vgd/scenenn) and [3RScan](https://github.com/WaldJohannaU/3RScan).
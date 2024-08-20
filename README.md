# ESAM:  **Embodied Segment Anything**

## Introduction

This repo contains PyTorch implementation for paper ESAM: Embodied Segment Anything based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

We propose a VFM-assisted 3D instance segmentation framework namely ESAM, which exploits the power of SAM to online segment anything in 3D scenes with high accuracy and fast speed.

## Getting Started
For environment setup and dataset preparation, please follow:
* [Installation](./docs/installation.md)
* [Dataset Preparation](./docs/dataset_preparation.md)

For training and evaluation, please follow:
* [Train and Evaluation](./docs/run.md)


## Main Results
We provide the checkpoints for quick reproduction of the results reported in the paper.

**Class-agnostic 3D instance segmentation results on ScanNet200 dataset:**

|  Method  |   Type  |     VFM     |  AP  | AP@50 | AP@25 | Speed(ms) | Downloads |
|:--------:|:-------:|:-----------:|:----:|:-----:|:-----:|:---------:|:---------:|
| [SAMPro3D](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D) | Offline |     [SAM](https://github.com/facebookresearch/segment-anything)     | 18.0 |  32.8 |  56.1 |     --    |     --    |
|   [SAI3D](https://github.com/yd-yin/SAI3D)  | Offline | [SemanticSAM](https://github.com/UX-Decoder/Semantic-SAM) | 30.8 |  50.5 |  70.6 |     --    |     --    |
|   [SAM3D](https://github.com/Pointcept/SegmentAnything3D)  |  Online |     SAM     | 20.6 |  35.7 |  55.5 | 1369+1518 |     --    |
|   ESAM   |  Online |     SAM     | 42.2 |  63.7 |  79.6 |  1369+**80**  |   [model](https://cloud.tsinghua.edu.cn/f/426d6eb693ff4b1fa04b/?dl=1)   |
|  ESAM-E  |  Online |   [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)   | **43.4** |  **65.4** |  **80.9** |   **20**+**80**   |   [model](https://cloud.tsinghua.edu.cn/f/7578d7e3d6764f6a93ee/?dl=1)   |

**Dataset transfer results from ScanNet200 to SceneNN and 3RScan:**
<table class="tg"><thead>
  <tr>
    <th class="tg-b2st" rowspan="2">Method</th>
    <th class="tg-b2st" rowspan="2">Type </th>
    <th class="tg-b2st" colspan="3">ScanNet200--&gt;SceneNN</th>
    <th class="tg-b2st" colspan="3">ScanNet200--&gt;3RScan</th>
  </tr>
  <tr>
    <th class="tg-wa1i">AP</th>
    <th class="tg-wa1i">AP@50</th>
    <th class="tg-wa1i">AP@25</th>
    <th class="tg-wa1i">AP</th>
    <th class="tg-wa1i">AP@50</th>
    <th class="tg-wa1i">AP@25</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix">SAMPro3D</td>
    <td class="tg-nrix">Offline</td>
    <td class="tg-nrix">12.6</td>
    <td class="tg-nrix">25.8</td>
    <td class="tg-nrix">53.2</td>
    <td class="tg-nrix">3.9</td>
    <td class="tg-nrix">8.0</td>
    <td class="tg-nrix">21.0</td>
  </tr>
  <tr>
    <td class="tg-nrix">SAI3D</td>
    <td class="tg-nrix">Offline</td>
    <td class="tg-nrix">18.6</td>
    <td class="tg-nrix">34.7</td>
    <td class="tg-nrix">65.7</td>
    <td class="tg-nrix">5.4</td>
    <td class="tg-nrix">11.8</td>
    <td class="tg-nrix">27.4</td>
  </tr>
  <tr>
    <td class="tg-nrix">SAM3D</td>
    <td class="tg-nrix">Online</td>
    <td class="tg-nrix">15.1</td>
    <td class="tg-nrix">30.0</td>
    <td class="tg-nrix">51.8</td>
    <td class="tg-nrix">6.2</td>
    <td class="tg-nrix">13.0</td>
    <td class="tg-nrix">33.9</td>
  </tr>
  <tr>
    <td class="tg-nrix">ESAM</td>
    <td class="tg-nrix">Online</td>
    <td class="tg-nrix"><b>28.8</b></td>
    <td class="tg-nrix"><b>52.2</b></td>
    <td class="tg-nrix">69.3</td>
    <td class="tg-nrix"><b>14.1</b></td>
    <td class="tg-nrix"><b>31.2</b></td>
    <td class="tg-nrix"><b>59.6</b></td>
  </tr>
  <tr>
    <td class="tg-nrix">ESAM-E</td>
    <td class="tg-nrix">Online</td>
    <td class="tg-nrix">28.6</td>
    <td class="tg-nrix">50.4</td>
    <td class="tg-nrix"><b>71.0</b></td>
    <td class="tg-nrix">13.9</td>
    <td class="tg-nrix">29.4</td>
    <td class="tg-nrix">58.8</td>
  </tr>
</tbody></table>

**3D instance segmentation results on ScanNet dataset:**
<table class="tg"><thead>
  <tr>
    <th class="tg-gabo" rowspan="2">Method</th>
    <th class="tg-gabo" rowspan="2">Type</th>
    <th class="tg-gabo" colspan="3">ScanNet</th>
    <th class="tg-gabo" colspan="3">SceneNN</th>
    <th class="tg-gabo" rowspan="2">FPS</th>
    <th class="tg-gabo" rowspan="2">Download</th>
  </tr>
  <tr>
    <th class="tg-uzvj">AP</th>
    <th class="tg-uzvj">AP50</th>
    <th class="tg-uzvj">AP25</th>
    <th class="tg-uzvj">AP</th>
    <th class="tg-uzvj">AP50</th>
    <th class="tg-uzvj">AP25</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8"><a href=https://github.com/SamsungLabs/td3d>TD3D</a></td>
    <td class="tg-9wq8">offline</td>
    <td class="tg-9wq8">46.2</td>
    <td class="tg-9wq8">71.1</td>
    <td class="tg-9wq8">81.3</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://github.com/oneformer3d/oneformer3d>Oneformer3D</a></td>
    <td class="tg-9wq8">offline</td>
    <td class="tg-9wq8">59.3</td>
    <td class="tg-9wq8">78.8</td>
    <td class="tg-9wq8">86.7</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://github.com/THU-luvision/INS-Conv>INS-Conv</a></td>
    <td class="tg-9wq8">online</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">57.4</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://github.com/xuxw98/Online3D>TD3D-MA</a></td>
    <td class="tg-9wq8">online</td>
    <td class="tg-9wq8">39.0</td>
    <td class="tg-9wq8">60.5</td>
    <td class="tg-9wq8">71.3</td>
    <td class="tg-9wq8">26.0</td>
    <td class="tg-9wq8">42.8</td>
    <td class="tg-9wq8">59.2</td>
    <td class="tg-9wq8">3.5</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8">ESAM-E</td>
    <td class="tg-9wq8">online</td>
    <td class="tg-9wq8">41.6</td>
    <td class="tg-9wq8">60.1</td>
    <td class="tg-9wq8">75.6</td>
    <td class="tg-9wq8">27.5</td>
    <td class="tg-9wq8">48.7</td>
    <td class="tg-uzvj"><b>64.6</b></td>
    <td class="tg-uzvj"><b>10</b></td>
    <td class="tg-9wq8"><a href=https://cloud.tsinghua.edu.cn/f/1eeff1152a5f4d4989da/?dl=1>model</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">ESAM-E_FF</td>
    <td class="tg-nrix">online</td>
    <td class="tg-wa1i"><b>42.6</b></td>
    <td class="tg-wa1i"><b>61.9</b></td>
    <td class="tg-wa1i"><b>77.1</b></td>
    <td class="tg-wa1i"><b>33.3</b></td>
    <td class="tg-wa1i"><b>53.6</b></td>
    <td class="tg-nrix">62.5</td>
    <td class="tg-nrix">9.8</td>
    <td class="tg-nrix"><a href=https://cloud.tsinghua.edu.cn/f/4c2dd1559e854f48be76/?dl=1>model</a></td>
  </tr>
</tbody></table>

## Acknowledgement
We thank a lot for the flexible codebase of [Oneformer3D](https://github.com/oneformer3d/oneformer3d) and [Online3D](https://github.com/xuxw98/Online3D), as well as the valuable datasets provided by [ScanNet](https://github.com/ScanNet/ScanNet), [SceneNN](https://github.com/hkust-vgd/scenenn) and [3RScan](https://github.com/WaldJohannaU/3RScan).
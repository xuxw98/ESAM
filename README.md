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


## Main Results
We provide the checkpoints for quick reproduction of the results reported in the paper.

**Class-agnostic 3D instance segmentation results on ScanNet200 dataset:**

|  Method  |   Type  |     VFM     |  AP  | AP@50 | AP@25 | Speed(ms) | Downloads |
|:--------:|:-------:|:-----------:|:----:|:-----:|:-----:|:---------:|:---------:|
| SAMPro3D | Offline |     SAM     | 18.0 |  32.8 |  56.1 |     --    |     --    |
|   SAI3D  | Offline | SemanticSAM | 30.8 |  50.5 |  70.6 |     --    |     --    |
|   SAM3D  |  Online |     SAM     | 20.2 |  35.7 |  55.5 | 1369+1518 |     --    |
|   OS3D   |  Online |     SAM     | 41.1 |  63.1 |  78.7 |  1369+88  |   [model](https://cloud.tsinghua.edu.cn/f/09685fd64e2849d681a1/?dl=1)   |
|  OS3D-E  |  Online |   FastSAM   | 43.4 |  65.4 |  80.9 |   20+88   |   [model](https://cloud.tsinghua.edu.cn/f/7578d7e3d6764f6a93ee/?dl=1)   |

**Dataset transfer results from ScanNet200 to SceneNN and 3RScan:**
<table class="tg"><thead>
  <tr>
    <th class="tg-b2st" rowspan="2">Method</th>
    <th class="tg-b2st" rowspan="2">Type </th>
    <th class="tg-b2st" colspan="3">ScanNet200--&gt;SceneNN</th>
    <th class="tg-b2st" colspan="3">ScanNet200--&gt;3RScan</th>
    <th class="tg-b2st" rowspan="2">FPS</th>
  </tr>
  <tr>
    <th class="tg-wa1i">AP</th>
    <th class="tg-wa1i">AP50</th>
    <th class="tg-wa1i">AP25</th>
    <th class="tg-wa1i">AP</th>
    <th class="tg-wa1i">AP50</th>
    <th class="tg-wa1i">AP25</th>
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
    <td class="tg-nrix">--</td>
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
    <td class="tg-nrix">--</td>
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
    <td class="tg-nrix">--</td>
  </tr>
  <tr>
    <td class="tg-nrix">OS3D</td>
    <td class="tg-nrix">Online</td>
    <td class="tg-nrix">26.6</td>
    <td class="tg-nrix">46.2</td>
    <td class="tg-nrix">63.1</td>
    <td class="tg-nrix">10.3</td>
    <td class="tg-nrix">23.6</td>
    <td class="tg-nrix">50.7</td>
    <td class="tg-nrix">3.5</td>
  </tr>
  <tr>
    <td class="tg-nrix">OS3D-E</td>
    <td class="tg-nrix">Online</td>
    <td class="tg-nrix">23.4</td>
    <td class="tg-nrix">43.0</td>
    <td class="tg-nrix">60.0</td>
    <td class="tg-nrix">10.2</td>
    <td class="tg-nrix">22.4</td>
    <td class="tg-nrix">48.5</td>
    <td class="tg-nrix">10</td>
  </tr>
</tbody></table>

**3D instance segmentation results on ScanNet dataset:**
<table class="tg"><thead>
  <tr>
    <th class="tg-gabo" rowspan="2">Method</th>
    <th class="tg-gabo" rowspan="2">Type</th>
    <th class="tg-gabo" colspan="3">ScanNet</th>
    <th class="tg-gabo" rowspan="2">FPS</th>
    <th class="tg-gabo" rowspan="2">Download</th>
  </tr>
  <tr>
    <th class="tg-uzvj">AP</th>
    <th class="tg-uzvj">AP50</th>
    <th class="tg-uzvj">AP25</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">TD3D</td>
    <td class="tg-9wq8">offline</td>
    <td class="tg-9wq8">46.2</td>
    <td class="tg-9wq8">71.1</td>
    <td class="tg-9wq8">81.3</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Oneformer3D</td>
    <td class="tg-9wq8">offline</td>
    <td class="tg-9wq8">59.3</td>
    <td class="tg-9wq8">78.8</td>
    <td class="tg-9wq8">86.7</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8">INS-Conv</td>
    <td class="tg-9wq8">online</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">57.4</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TD3D-MA</td>
    <td class="tg-9wq8">online</td>
    <td class="tg-9wq8">39.0</td>
    <td class="tg-9wq8">60.5</td>
    <td class="tg-9wq8">71.3</td>
    <td class="tg-9wq8">3.5</td>
    <td class="tg-9wq8">--</td>
  </tr>
  <tr>
    <td class="tg-9wq8">OS3D-E</td>
    <td class="tg-9wq8">online</td>
    <td class="tg-9wq8">41.6</td>
    <td class="tg-9wq8">60.1</td>
    <td class="tg-9wq8">75.6</td>
    <td class="tg-9wq8">10</td>
    <td class="tg-9wq8"><a href='https://cloud.tsinghua.edu.cn/f/1eeff1152a5f4d4989da/?dl=1'>model</td>
  </tr>
</tbody></table>

## Acknowledgement
We thank a lot for the flexible codebase of [Oneformer3D](https://github.com/oneformer3d/oneformer3d) and [Online3D](https://github.com/xuxw98/Online3D), as well as the valuable datasets provided by [ScanNet](https://github.com/ScanNet/ScanNet), [SceneNN](https://github.com/hkust-vgd/scenenn) and [3RScan](https://github.com/WaldJohannaU/3RScan).
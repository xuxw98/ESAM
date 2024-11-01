## Installation
The code is tested with Python == 3.8, Pytorch == 1.13.1, CUDA == 11.6, mmengine == 0.10.3, mmdet3d == 1.4.0, mmcv == 2.0.0, mmdet == 3.2.0 and MinkowskiEngine == 0.5.4. We recommend you to use Anaconda to make sure that all dependencies are installed correctly.

**Step 1**: Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 2**: Create a new conda environment and activate it:
```bash
conda create -n ESAM python=3.8
conda activate ESAM
```
**Step 3**: Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.
```bash
conda install pytorch torchvision -c pytorch
```

**Step 4**: Follow [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md) to install mmcv, mmdet3d and mmdet.

**Step 5**: Install MinkowskiEngine:
```bash
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
```

**Step 6**: Install SAM & FastSAM:
* Please follow [here](https://github.com/facebookresearch/segment-anything/blob/main/README.md) for installation of SAM. Then download the checkpoint for [Vit-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it in the folder 'data'.

* Please follow [here](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/README.md) for installation of FastSAM. Then download the checkpoint for [FastSAM](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing) and put it in the folder 'data'.

**Step 7**: Download backbone checkpoint:

We follow [Oneformer3D](https://github.com/filaPro/oneformer3d) to initialize the backbone from [Mask3D](https://github.com/JonasSchult/Mask3D) checkpoint. It should be [downloaded](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/mask3d_scannet200.pth) and put to `work_dirs/tmp` before training.

**Step 8**: Follow [SAM3D](https://github.com/Pointcept/SegmentAnything3D) to install pointops.

**Step 9**: Get `segmentator` repository:

Please refer to [segmentator](https://github.com/Karbo123/segmentator.git) to get mesh segmentator.
You should clone the repository in the folder 'data', and it will be imported in 'batch_load_scannet_data.py' for generating mesh segmentation results.

**Step 10**: Install other dependencies:
```bash
pip install -r requirements.txt
```




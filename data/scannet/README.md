### Prepare ScanNet Data for Indoor Detection or Segmentation Task

We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this level of directory. If you are performing segmentation tasks and want to upload the results to its official [benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/), please also link or move the 'scans_test' folder to this directory.

2. In this directory, extract point clouds and annotations by running `python batch_load_scannet_data.py`. Add the `--scannet200` flag if you want to get markup for the ScanNet200 dataset.

3. Enter the project root directory, generate training data by running

```bash
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
```
&nbsp; &nbsp; &nbsp; &nbsp; or for ScanNet200:

```bash
mkdir data/scannet200
python tools/create_data.py scannet200 --root-path ./data/scannet --out-dir ./data/scannet200 --extra-tag scannet200
```

The overall process for ScanNet could be achieved through the following script

```bash
python batch_load_scannet_data.py
cd ../..
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
```

Or for ScanNet200:

```bash
python batch_load_scannet_data.py --scannet200
cd ../..
mkdir data/scannet200
python tools/create_data.py scannet200 --root-path ./data/scannet --out-dir ./data/scannet200 --extra-tag scannet200
```

The directory structure after pre-processing should be as below

```
scannet
├── meta_data
├── batch_load_scannet_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── README.md
├── scans
├── scans_test
├── scannet_instance_data
├── points
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── super_points
│   ├── xxxxx.bin
├── seg_info
│   ├── train_label_weight.npy
│   ├── train_resampled_scene_idxs.npy
│   ├── val_label_weight.npy
│   ├── val_resampled_scene_idxs.npy
├── scannet_oneformer3d_infos_train.pkl
├── scannet_oneformer3d_infos_val.pkl
├── scannet_oneformer3d_infos_test.pkl

```

## Dataset Preparation

#### ScanNet200: 

For ScanNet200，follow [here](./data/scannet200/README.md).

For ScanNet200-SV, [download](https://github.com/ScanNet/ScanNet) '2D' and '3D' folders to the folder 'data/scannet200-sv', then  run: 

```bash
python load_scannet_sv_data_v2.py
cd ../..
python tools/create_data.py scannet200_sv --root-path ./data/scannet200-sv --out-dir ./data/scannet200-sv --extra-tag scannet200_sv
```

For ScanNet200-MV, link '2D' and '3D' folders to the folder 'data/scannet200-mv', then  run: 

```bash
python load_scannet_mv_data.py
cd ../..
python tools/create_data.py scannet200_mv --root-path ./data/scannet200-mv --out-dir ./data/scannet200-mv --extra-tag scannet200_mv
```

You can also generate the data for ScanNet200-SV and ScanNet200-MV using FastSAM instead of SAM by running the following commands:

```bash
# ScanNet200-SV
python load_scannet_sv_data_v2_fast.py
cd ../..
python tools/create_data.py scannet200_sv --root-path ./data/scannet200-sv --out-dir ./data/scannet200-sv --extra-tag scannet200_sv
```

```bash
# ScanNet200-MV
python load_scannet_mv_data_fast.py
cd ../..
python tools/create_data.py scannet200_mv --root-path ./data/scannet200-mv --out-dir ./data/scannet200-mv --extra-tag scannet200_mv
```


#### ScanNet:
For ScanNet, please follow [here](./data/scannet/README.md).
For ScanNet-SV, link '2D' and '3D' folders to the folder 'data/scannet-sv', then run:

```bash
python load_scannet_sv_data_v2.py
cd ../..
python tools/create_data.py scannet_sv --root-path ./data/scannet-sv --out-dir ./data/scannet-sv --extra-tag scannet_sv
```
For ScanNet-MV, link '2D' and '3D' folders to the folder 'data/scannet-mv', then run:

```bash 
python load_scannet_mv_data.py
cd ../..
python tools/create_data.py scannet_mv --root-path ./data/scannet-mv --out-dir ./data/scannet-mv --extra-tag scannet_mv
```

You can also generate the data for ScanNet-SV and ScanNet-MV using FastSAM instead of SAM by running the following commands:

```bash
# ScanNet-SV
python load_scannet_sv_data_v2_fast.py
cd ../..
python tools/create_data.py scannet_sv --root-path ./data/scannet-sv --out-dir ./data/scannet-sv --extra-tag scannet_sv
```

```bash
# ScanNet-MV
python load_scannet_mv_data_fast.py
cd ../..
python tools/create_data.py scannet_mv --root-path ./data/scannet-mv --out-dir ./data/scannet-mv --extra-tag scannet_mv
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
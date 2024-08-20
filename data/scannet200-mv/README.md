## Prepare ScanNet200-MV Dataset
**Step1:** Link '2D' and '3D' folders to this directory, namely `ESAM/data/scannet200-mv`.
You should get the following directory structure:
```
scannet200-mv
├── 2D
│   ├── scenexxxx_xx
│   └── ... 
├── 3D
│   ├── scenexxxx_xx   
│   └── ...
├── meta_data
│   ├── scannetv2_train.txt
│   ├── scannetv2_val.txt
│   ├── scannetv2_labels.combined.tsv
│   └── ...
├── load_scannet_data.py
├── load_scannet_mv_data.py
├── load_scannet_mv_data_fast.py
├── README.md
└── scannet_utils.py
```

**Step2:** Run the following commands:
```bash
python load_scannet_mv_data.py
```
or use FastSAM instead of SAM by running:
```bash
python load_scannet_mv_data_fast.py
```

It will take around 1~2 day to finish the data preparation and get several new folders, including `axis_align_matrix`, `instance_mask`, `points`, `semantic_mask`, `super_points`.

**Step3:** Go back to the root directory of ESAM, and generate .pkl file by running:
```bash
python tools/create_data.py scannet200_mv --root-path ./data/scannet200-mv --out-dir ./data/scannet200-mv --extra-tag scannet200_mv
```

**Final folder structure:**
``` 
scannet200-mv
├── 2D
│   ├── scenexxxx_xx
│   └── ... 
├── 3D
│   ├── scenexxxx_xx   
│   └── ...
├── axis_align_matrix
│   ├── scenexxxx_xx.npy
│   └── ...
├── instance_mask
│   ├── scenexxxx_xx
│   │   ├── xx.bin
│   │   └── ...
│   └── ...
├── meta_data
│   ├── scannetv2_train.txt
│   ├── scannetv2_val.txt
│   ├── scannetv2_labels.combined.tsv
│   └── ...
├── points
│   ├── scenexxxx_xx
│   │   ├── xx.bin
│   │   └── ...
│   └── ...
├── semantic_mask
│   ├── scenexxxx_xx
│   │   ├── xx.bin
│   │   └── ...
│   └── ...
├── super_points
│   ├── scenexxxx_xx
│   │   ├── xx.bin
│   │   └── ...
│   └── ...
├── load_scannet_data.py
├── load_scannet_mv_data.py
├── load_scannet_mv_data_fast.py
├── scannet200_mv_oneformer3d_infos_train.pkl
├── scannet200_mv_oneformer3d_infos_val.pkl
├── README.md
└── scannet_utils.py
```

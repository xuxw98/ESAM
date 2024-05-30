## Prepare ScanNet200-SV Dataset
**Step1:** Link '2D' and '3D' folders to this directory, namely `OS3D/data/scannet200-sv`.
You should get the following directory structure:
```
scannet200-sv
├── 2D
│   ├── scenexxxx_xx
│   └── ... 
├── 3D
│   ├── scenexxxx_xx   
│   └── ...
├── meta_data
│   ├── scannetv2_sv_train.txt
│   ├── scannetv2_sv_val.txt
│   ├── scannetv2_labels.combined.tsv
│   └── ...
├── load_scannet_data.py
├── load_scannet_sv_data_v2.py
├── load_scannet_sv_data_v2_fast.py
├── README.md
└── scannet_utils.py
```

**Step2:** Run the following commands:
```bash
python load_scannet_sv_data_v2.py   
```
or use FastSAM instead of SAM by running:
```bash
python load_scannet_sv_data_v2_fast.py
```

It will take around 0.5~1 day to finish the data preparation and get a new folder `scannet_sv_instance_data`.

**Step3:** Go back to the root directory of OS3D, and generate .pkl file by running:
```bash
python tools/create_data.py scannet200_sv --root-path ./data/scannet200-sv --out-dir ./data/scannet200-sv --extra-tag scannet200_sv
```

**Final folder structure:**
``` 
scannet200-sv
├── 2D
│   ├── scenexxxx_xx
│   └── ... 
├── 3D
│   ├── scenexxxx_xx   
│   └── ...
├── instance_mask
│   ├── scenexxxx_xx_xx.bin
│   └── ...
├── meta_data
│   ├── scannetv2_sv_train.txt
│   ├── scannetv2_sv_val.txt
│   ├── scannetv2_labels.combined.tsv
│   └── ...
├── points
│   ├── scenexxxx_xx_xx.bin
│   └── ...
├── pose_centered
│   ├── scenexxxx_xx
│   │   ├── x.npy
│   │   └── ...
│   └── ...
├── scannet_sv_instance_data
│   ├── scenexxxx_xx_xx_axis_align_matrix.npy
│   ├── scenexxxx_xx_xx_ins_label.npy
│   ├── scenexxxx_xx_xx_sem_label.npy
│   ├── scenexxxx_xx_xx_sp_label.npy
│   ├── scenexxxx_xx_xx_vert.npy
│   └── ...
├── semantic_mask
│   ├── scenexxxx_xx_xx.bin
│   └── ...
├── super_points
│   ├── scenexxxx_xx_xx.bin
│   └── ...
├── load_scannet_data.py
├── load_scannet_sv_data_v2.py
├── load_scannet_sv_data_v2_fast.py
├── scannet200_sv_oneformer3d_infos_train.pkl
├── scannet200_sv_oneformer3d_infos_val.pkl
├── README.md
└── scannet_utils.py
```

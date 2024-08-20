## Prepare SceneNN Dataset
**Step1:** The processed SceneNN data can be downloaded from the repo of [Online3D](https://cloud.tsinghua.edu.cn/d/641cd2b7a123467d98a6/). Run `cat SceneNN.tar.* > SceneNN.tar` to merge the files. Then unzip 'SceneNN.tar' to get 'SceneNN' folder.

Link 'SceneNN' folder to this directory, namely  `ESAM/data/scenenn`. You should get the following directory structure:
```
scenenn
├── SceneNN
│   ├── 005
│   │   ├── depth
│   │   ├── image
│   │   ├── ins
│   │   ├── label
│   │   ├── point
│   │   ├── pose
│   │   ├── 005.ply
│   │   ├── 005.xml
│   │   └── timestamp.txt
│   ├── 011
│   └── ...
├── batch_load_scenenn_data.py
├── load_scannet_data.py
├── README.md
├── scannetv2-labels.combined.tsv
└── scannet_utils.py
```

**Step2:** Run the following commands:
```bash
python batch_load_scenenn_data.py
```

Then you will get several new folders, including `scenenn_instance_data`, `downsample_indices`, `mesh_segs`.

**Step3:** Go back to the root directory of ESAM, and generate .pkl file by running:
```bash
python tools/create_data.py scenenn --root-path ./data/scenenn --out-dir ./data/scenenn --extra-tag scenenn
```

**Final folder structure:**
``` 
scenenn
├── SceneNN
│   ├── 005
│   │   ├── depth
│   │   ├── image
│   │   ├── ins
│   │   ├── label
│   │   ├── point
│   │   ├── pose
│   │   ├── 005.ply
│   │   ├── 005.xml
│   │   └── timestamp.txt
│   ├── 011
│   └── ...
├── downsample_indices
│   ├── xxx.npy
│   └── ...
├── instance_mask
│   ├── xxx.bin
│   └── ...
├── mesh_segs
│   ├── xxx.segs.json
│   └── ...
├── points
│   ├── xxx.bin
│   └── ...
├── semantic_mask
│   ├── xxx.bin
│   └── ...
├── super_points
│   ├── xxx.bin
│   └── ...
├── scenenn_instance_data
│   ├── xxx_aligned_bbox.npy
│   ├── xxx_ins_label.npy
│   ├── xxx_sem_label.npy
│   ├── xxx_sp_label.npy
│   ├── xxx_unaligned_bbox.npy
│   ├── xxx_vert.npy
│   └── ...
├── batch_load_scenenn_data.py
├── load_scannet_data.py
├── scenenn_oneformer3d_infos_val.pkl
├── README.md
├── scannetv2-labels.combined.tsv
└── scannet_utils.py
```

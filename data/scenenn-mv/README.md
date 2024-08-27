## Prepare SceneNN-MV Dataset
**Step1:** The processed SceneNN data can be downloaded from the repo of [Online3D](https://cloud.tsinghua.edu.cn/d/641cd2b7a123467d98a6/). Run `cat SceneNN.tar.* > SceneNN.tar` to merge the files. Then unzip 'SceneNN.tar' to get 'SceneNN' folder.

Link 'SceneNN' folder to this directory, namely  `ESAM/data/scenenn-mv`. You should get the following directory structure:
```
scenenn-mv
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
├── load_scannet_data.py
├── load_scenenn_mv_data.py
├── README.md
├── scannetv2-labels.combined.tsv
└── scannet_utils.py
```

**Step2:** Run the following commands:
```bash
python load_scenenn_mv_data.py
```
or use FastSAM instead of SAM by running:
```bash
python load_scenenn_mv_data_fast.py
```


Then you will get several new folders, including `instance_mask`, `points`, `semantic_mask`, `super_points`.

**Step3:** Go back to the root directory of ESAM, and generate .pkl file by running:
```bash
python tools/create_data.py scenenn_mv --root-path ./data/scenenn-mv --out-dir ./data/scenenn-mv --extra-tag scenenn_mv
```

**Final folder structure:**
``` 
scenenn-mv
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
├── instance_mask
│   ├── xxx
│   │   ├── x.bin
│   │   └── ...
│   └── ...
├── points
│   ├── xxx
│   │   ├── x.bin
│   │   └── ...
│   └── ...
├── semantic_mask
│   ├── xxx
│   │   ├── x.bin
│   │   └── ...
│   └── ...
├── super_points
│   ├── xxx
│   │   ├── x.bin
│   │   └── ...
│   └── ...
├── load_scannet_data.py
├── load_scenenn_mv_data.py
├── scenenn_mv_oneformer3d_infos_val.pkl
├── README.md
├── scannetv2-labels.combined.tsv
└── scannet_utils.py
```

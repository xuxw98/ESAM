## Prepare 3RScan-MV Dataset
**Step1:** Download 3RScan dataset from [here](https://github.com/WaldJohannaU/3RScan?tab=readme-ov-file). We test our model on the test split, which contain 46 scenes. Then put downloaded scenes in a folder named '3RScan' and rename the scenes as ['000', '001', ..., '045'].

Link '3RScan' folder to this directory, namely  `ESAM/data/3RScan-mv`. You should get the following directory structure:
```
3RScan-mv
├── 3RScan
│   ├── 000
│   │   ├── sequence
│   │   ├── labels.instances.annotated.v2.ply
│   │   ├── mesh.refined_0.png
│   │   ├── mesh.refined.0.010000.segs.v2.json
│   │   ├── mesh.refined.mtl
│   │   ├── mesh.refined.v2.obj
│   │   └── semseg.v2.json
│   ├── 001
│   └── ...
├── load_3rscan_mv_data.py
├── load_scannet_data.py
├── README.md
├── 3RScan.v2 Semantic Classes - Mapping.tsv
└── scannet_utils.py
```

**Step2:** Run the following commands:
```bash
python load_3rscan_mv_data.py
```
or use FastSAM instead of SAM by running:
```bash
python load_3rscan_mv_data_fast.py
```


Then you will get several new folders, including `instance_mask`, `points`, `semantic_mask`, `super_points`.


**Step3:** Go back to the root directory of ESAM, and generate .pkl file by running:
```bash
python tools/create_data.py 3rscan_mv --root-path ./data/3RScan-mv --out-dir ./data/3RScan-mv --extra-tag 3rscan_mv
```

**Final folder structure:**
``` 
3RScan-mv
├── 3RScan
│   ├── 000
│   │   ├── sequence
│   │   ├── labels.instances.annotated.v2.ply
│   │   ├── mesh.refined_0.png
│   │   ├── mesh.refined.0.010000.segs.v2.json
│   │   ├── mesh.refined.mtl
│   │   ├── mesh.refined.v2.obj
│   │   └── semseg.v2.json
│   ├── 001
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
├── load_3rscan_mv_data.py
├── load_scannet_data.py
├── 3rscan_mv_oneformer3d_infos_val.pkl
├── README.md
├── 3RScan.v2 Semantic Classes - Mapping.tsv
└── scannet_utils.py
```

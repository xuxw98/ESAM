## Prepare 3RScan Dataset
**Step1:** Download 3RScan dataset from [here](https://github.com/WaldJohannaU/3RScan?tab=readme-ov-file). We test our model on the test split, which contain 46 scenes. Then put downloaded scenes in a folder named '3RScan' and rename the scenes as ['000', '001', ..., '045'].

<!-- We also provide processed data for 3RScan dataset. You can download the processed dataset from [here], and unzip it to get '3RScan' folder. -->

Link '3RScan' folder to this directory, namely  `ESAM/data/3RScan`. You should get the following directory structure:
```
3RScan
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
├── batch_load_3rscan_data.py
├── load_scannet_data.py
├── README.md
├── 3RScan.v2 Semantic Classes - Mapping.tsv
└── scannet_utils.py
```

**Step2:** Run the following commands:
```bash
python batch_load_3rscan_data.py
```

Then you will get a new folder named `3rscan_instance_data`.

**Step3:** Go back to the root directory of ESAM, and generate .pkl file by running:
```bash
python tools/create_data.py 3rscan --root-path ./data/3RScan --out-dir ./data/3RScan --extra-tag 3rscan
```

**Final folder structure:**
``` 
3RScan
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
│   ├── xxx.bin
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
├── 3rscan_instance_data
│   ├── xxx_aligned_bbox.npy
│   ├── xxx_ins_label.npy
│   ├── xxx_sem_label.npy
│   ├── xxx_sp_label.npy
│   ├── xxx_unaligned_bbox.npy
│   ├── xxx_vert.npy
│   └── ...
├── batch_load_3rscan_data.py
├── load_scannet_data.py
├── 3rscan_oneformer3d_infos_val.pkl
├── README.md
├── 3RScan.v2 Semantic Classes - Mapping.tsv
└── scannet_utils.py
```

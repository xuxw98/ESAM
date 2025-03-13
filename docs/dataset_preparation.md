## Dataset Preparation

For each dataset, you should **first process the reconstructed dataset**, which is the dataset without the `-SV` or `-MV` suffix. This dataset is used for model evaluation.

If you want to train the entire model, you need to **further process the SV and MV data**; if you only want to perform evaluation, you only need to **process the MV data**.

In addition to Tsinghua Cloud, we also upload the checkpoints and processed data to HuggingFace. Click [here](https://huggingface.co/XXXCARREY/EmbodiedSAM/tree/main) for more details.

#### ScanNet200: 

For ScanNet200, please follow [here](../data/scannet200/README.md).

For ScanNet200-SV, please follow [here](../data/scannet200-sv/README.md).

For ScanNet200-MV, please follow [here](../data/scannet200-mv/README.md).

#### ScanNet:
For ScanNet, please follow [here](../data/scannet/README.md).

For ScanNet-SV, please follow [here](../data/scannet-sv/README.md).

For ScanNet-MV, please follow [here](../data/scannet-mv/README.md).

#### SceneNN:
For SceneNN, please follow [here](../data/scenenn/README.md).


For SceneNN-MV, please follow [here](../data/scenenn-mv/README.md).

#### 3RScan:

For 3RScan, please follow [here](../data/3RScan/README.md).


For 3RScan-MV, please follow [here](../data/3RScan-mv/README.md).

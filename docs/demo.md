## Visualization Demo

You can run the visualization demo by running the following command:

```
CUDA_VISIBLE_DEVICES=0 python vis_demo/online_demo.py --scene_idx <scene_idx> --config <config_file> --checkpoint <checkpoint_file>
```

For `ScanNet` or `ScanNet200`, the `<scene_idx>` should be in the format of `scenexxxx_xx`, like `scene0000_00`. For `SceneNN` or `3RScan`, the `<scene_idx>` should be in the format of `xxx`, like `000`.

It will process the specified scene and visualize the results. The visualization includes the input RGB sequence and the segmentation results in the form of a 3D point cloud colored by the predicted instance labels.
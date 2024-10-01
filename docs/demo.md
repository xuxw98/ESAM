## Visualization Demo

You can run the visualization demo by running the following command:

```
CUDA_VISIBLE_DEVICES=0 python vis_demo/online_demo.py --scene_idx <scene_idx> --config <config_file> --checkpoint <checkpoint_file>
```

It will process the specified scene and visualize the results. The visualization includes the input RGB sequence and the segmentation results in the form of a 3D point cloud colored by the predicted instance labels.
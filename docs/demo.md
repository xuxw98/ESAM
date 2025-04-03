## Visualization Demo of Datasets

You can run the visualization demo by running the following command:

```
CUDA_VISIBLE_DEVICES=0 python vis_demo/online_demo.py --scene_idx <scene_idx> --config <config_file> --checkpoint <checkpoint_file>
```

For `ScanNet` or `ScanNet200`, the `<scene_idx>` should be in the format of `scenexxxx_xx`, like `scene0000_00`. For `SceneNN` or `3RScan`, the `<scene_idx>` should be in the format of `xxx`, like `000`.

It will process the specified scene and visualize the results. The visualization includes the input RGB sequence and the segmentation results in the form of a 3D point cloud colored by the predicted instance labels.

## Visualization Demo of Custom Data and Video Stream
#### Try your own data from files: 
You can run the visualization demo of your own data by running the following command:
```
# Check all the arguments of the script
CUDA_VISIBLE_DEVICES=0 python vis_demo/stream_demo.py -h

# Try your own data with default arguments
CUDA_VISIBLE_DEVICES=0 python vis_demo/stream_demo.py --data_root <data_root>
```

The custom data should be orgainized as below:
```
data_root
├── color
│   ├── 0.jpg
│   └── ... 
├── depth
│   ├── 0.png 
│   └── ...
├── pose
│   ├── 0.txt
│   └── ...
└── intrinsic.txt
```

#### Try your own data from video stream:
You can implement this functionality by calling the `run_single_frame` function from the `StreamDemo` class in `vis_demo/stream_demo.py`. First, initialize a `StreamDemo` object, then pass the current video frame's RGBD data along with the camera intrinsics and extrinsics as inputs to `run_single_frame`. This will return segmentation results for all processed frames.

```python
# Initialize the StreamDemo object
stream_demo = StreamDemo(args)
# Input the RGBD data and camera intrinsics/extrinsics
all_points, all_points_color, pred_ins_masks = stream_demo.run_single_frame(rgb, depth, pose, intrinsic)
```
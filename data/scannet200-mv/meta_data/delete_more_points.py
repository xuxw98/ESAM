import os
import sys


scene_names = os.listdir("./points")
for scene_name in scene_names:
    ins_names = os.listdir("./instance_mask/"+scene_name)
    pts_names = os.listdir("./points/"+scene_name)
    for pts_name in pts_names:
        if pts_name not in ins_names:
            os.remove("./points/"+scene_name+"/"+pts_name)
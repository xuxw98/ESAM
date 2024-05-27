import os
import pdb

train_scenes = open('meta_data/scannetv2_train.txt', 'r').readlines()
val_scenes = open('meta_data/scannetv2_val.txt', 'r').readlines()
train_scenes = [ts[:-1] for ts in train_scenes]
val_scenes = [vs[:-1] for vs in val_scenes]

train_sv_scenes = []
val_sv_scenes = []
files = os.listdir('scannet_sv_instance_data/')
for file in files:
    if not file.endswith('_vert.npy'):
        continue
    scene_idx = file[:12]
    if scene_idx in train_scenes:
        train_sv_scenes.append(file[:-9]+'\n')
    elif scene_idx in val_scenes:
        val_sv_scenes.append(file[:-9]+'\n')

with open('meta_data/scannetv2_sv_train.txt', 'w') as f:
    f.writelines(train_sv_scenes)


with open('meta_data/scannetv2_sv_val.txt', 'w') as f:
    f.writelines(val_sv_scenes)

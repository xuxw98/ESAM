# V2: 2d_point+3d_ins+knn-->2d_ins-->2d_sem

import enum
import cv2
import shutil
import numpy as np
import math
from scipy import stats
import os
from plyfile import PlyData,PlyElement
from scipy import stats
from sklearn.cluster import KMeans
from segment_anything import build_sam, SamAutomaticMaskGenerator
import pdb
import torch
import pointops
import scannet_utils
from tqdm import tqdm
def export(mesh_file, xml_file, label_map_file):
    label_map = scannet_utils.read_label_mapping(label_map_file, label_from='nyu40class', label_to='nyu40id')
    # breakpoint()
    mesh, label_txt, ins_label, aligned_bboxes, object_id_to_label = scannet_utils.read_mesh_vertices_rgb(mesh_file, xml_file)
    if mesh.shape[0] > 1000000:
        choice = np.random.choice(mesh.shape[0], 1000000, replace=False)
        mesh = mesh[choice]
        label_txt = label_txt[choice]
        ins_label = ins_label[choice]
    sem_label = []
    for i in range(label_txt.shape[0]):
        if label_txt[i] == 'unknown':
            sem_label.append(0)
            ins_label[i] = 0
        elif label_txt[i] == 'otherprop':
            sem_label.append(40)
        else:
            try:
                sem_label.append(label_map[label_txt[i]])
            except:
                sem_label.append(0)
                ins_label[i] = 0
    sem_label = np.array(sem_label)
    instance_bboxes = extract_bbox(mesh, ins_label)
    return mesh, ins_label, sem_label, instance_bboxes, label_map, object_id_to_label

def extract_bbox(mesh, ins_label):
    num_instances = len(np.unique(ins_label)) - 1
    instance_bboxes = np.zeros((num_instances, 6))
    for obj_id in range(1, num_instances + 1):
        obj_pc = mesh[ins_label == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        xyz_min = np.min(obj_pc, axis=0)
        xyz_max = np.max(obj_pc, axis=0)
        bbox = np.concatenate([(xyz_min + xyz_max) / 2.0, xyz_max - xyz_min])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox
    return instance_bboxes

def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def convert_from_uvd(u, v, depth, intr, pose):
    # u is width index, v is height index
    depth_scale = 1000
    z = depth / depth_scale

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    
    uv = np.concatenate([u,v,padding], axis=0)
    xyz = (np.linalg.inv(intr[:3,:3]) @ uv) * np.expand_dims(z,axis=0)
    xyz = np.concatenate([xyz,padding], axis=0)
    xyz = pose @ xyz
    xyz[:3,:] /= xyz[3,:] 
    return xyz[:3, :].T

def export_one_scan(path_prefix, scan_name):    
    mesh_file = os.path.join(path_prefix, scan_name, scan_name + '.ply')
    xml_file = os.path.join(path_prefix, scan_name, scan_name + '.xml')
    aligned_mesh_vertices, instance_labels, semantic_labels, bboxes, label_map, object_id_to_label = \
        export(mesh_file, xml_file, './scannetv2-labels.combined.tsv')
    bbox_instance_labels = np.arange(1,bboxes.shape[0]+1)

    return aligned_mesh_vertices, instance_labels, semantic_labels, label_map, object_id_to_label, bboxes, bbox_instance_labels


def select_points_in_bbox(xyz, ins, bboxes, bbox_instance_labels):
    # Add a small margin to box size
    delta = 0.05
    for i in range(bboxes.shape[0]):
        instance_target = bbox_instance_labels[i]
        x_max = bboxes[i,0] + bboxes[i,3]/2 + delta
        x_min = bboxes[i,0] - bboxes[i,3]/2 - delta
        y_max = bboxes[i,1] + bboxes[i,4]/2 + delta
        y_min = bboxes[i,1] - bboxes[i,4]/2 - delta
        z_max = bboxes[i,2] + bboxes[i,5]/2 + delta
        z_min = bboxes[i,2] - bboxes[i,5]/2 - delta
        max_range = np.array([x_max, y_max, z_max])
        min_range = np.array([x_min, y_min, z_min])
        margin_positive = xyz-min_range
        margin_negative = xyz-max_range
        in_criterion = margin_positive * margin_negative
        zero = np.zeros(in_criterion.shape)
        one = np.ones(in_criterion.shape)
        in_criterion = np.where(in_criterion<=0,one,zero)
        mask_inbox = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2]
        mask_inbox = mask_inbox.astype(np.bool_)
        mask_ins = np.in1d(ins, instance_target)
        ins[mask_ins*(~mask_inbox)] = 0
    unique_ins = np.unique(ins)
    # including floor and wall as objects (we do not distinguish fg and bg here)
    object_num = len(unique_ins) - int(0 in unique_ins)
    return ins, object_num



def process_cur_scan(cur_scan, mask_generator):
    scan_name_index = cur_scan["scan_name_index"]
    scan_name = cur_scan["scan_name"]
    path_prefix = cur_scan["path_prefix"]
    scan_num = cur_scan["scan_num"]
    print(scan_name)

    scan_path = os.path.join(path_prefix,scan_name)

    unify_dim = (640, 480)
    unify_intrinsic = adjust_intrinsic(make_intrinsic(544.47329,544.47329,320,240), [640,480], unify_dim)

    # Sort string. 0 20 40 60 80 100 120 ...
    POSE_list = sorted(os.listdir(os.path.join(scan_path, 'pose')), key=lambda s: int(s[:-4]))
    rgb_map_list = sorted(os.listdir(os.path.join(scan_path, 'image')), key=lambda s: int(s[5:-4]))[:len(POSE_list)]
    depth_map_list = sorted(os.listdir(os.path.join(scan_path, 'depth')), key=lambda s: int(s[5:-4]))[:len(POSE_list)]
    
    poses = [np.load(os.path.join(scan_path, 'pose', path)) for path in POSE_list]

    os.makedirs("points/%s" % scan_name, exist_ok=True)
    os.makedirs("super_points/%s" % scan_name, exist_ok=True)
    os.makedirs("semantic_mask/%s" % scan_name, exist_ok=True)
    os.makedirs("instance_mask/%s" % scan_name, exist_ok=True)

    aligned_mesh_vertices, instance_labels, semantic_labels, label_map, object_id_to_label, \
        aligned_bboxes, bbox_instance_labels = export_one_scan(path_prefix, scan_name)

    for frame_i, (rgb_map_name, \
        depth_map_name, \
        pose) \
        in enumerate(zip(rgb_map_list, depth_map_list, poses)):
        assert frame_i * 40 + 1 == int(rgb_map_name[5:-4])
        # set interval=40
        # if frame_i % 2 != 0:
        # if frame_i % 2 == 0:
            # continue

        depth_map = cv2.imread(os.path.join(scan_path, 'depth', depth_map_name), -1)
        color_map = cv2.imread(os.path.join(scan_path, 'image', rgb_map_name))
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

        # SAM-->super point
        masks = mask_generator.generate(color_map)
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        group_ids = np.full((color_map.shape[0], color_map.shape[1]), -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0
        for i in range(num_masks):
            mask_now = masks[i]["segmentation"]
            group_ids[mask_now] = group_counter
            group_counter += 1

        # convert depth map to point cloud
        height, width = depth_map.shape    
        w_ind = np.arange(width)
        h_ind = np.arange(height)

        ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
        ww_ind = ww_ind.reshape(-1)
        hh_ind = hh_ind.reshape(-1)
        depth_map = depth_map.reshape(-1)
        group_ids = group_ids.reshape(-1)
        color_map = color_map.reshape(-1, 3)

        valid = np.where(depth_map > 0.1)[0]
        ww_ind = ww_ind[valid]
        hh_ind = hh_ind[valid]
        depth_map = depth_map[valid]
        group_ids = group_ids[valid]
        rgb = color_map[valid]

        # For MV: downsample to 20000
        unaligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, pose)
        unaligned_xyz = np.concatenate([unaligned_xyz, rgb], axis=-1)
        xyz_all = np.concatenate([unaligned_xyz, group_ids.reshape(-1,1)], axis=-1)
        xyz_all = random_sampling(xyz_all, 20000)
        unaligned_xyz, group_ids = xyz_all[:, :6], xyz_all[:, 6]
        aligned_xyz = unaligned_xyz[:,:3]

        # Get instance label (ins) from 3D annotation by KNN
        target_coord = torch.tensor(aligned_xyz).cuda().contiguous().float()
        target_offset = torch.tensor(target_coord.shape[0]).cuda().float()
        source_coord = torch.tensor(aligned_mesh_vertices[:,:3]).cuda().contiguous().float()
        source_offset = torch.tensor(source_coord.shape[0]).cuda().float()
        indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.cpu().numpy()
        ins = instance_labels[indices.reshape(-1)].astype(np.uint32)
        # mask_dis = dis.reshape(-1).cpu().numpy() > 0.05
        # ins[mask_dis] = 0
        # further denoise
        ins, object_num = select_points_in_bbox(aligned_xyz, ins, aligned_bboxes, bbox_instance_labels)
        print('object_num: ',object_num)

        # Get sem from ins
        sem = np.zeros_like(ins, dtype=np.uint32)
        for ins_ids in np.unique(ins):
            if ins_ids != 0:
                sem[ins == ins_ids] = label_map[object_id_to_label[ins_ids]]
        
        # Get superpoints
        # TODO: set other_ins_num as 10-->8
        points_without_seg = unaligned_xyz[group_ids == -1]
        other_ins = KMeans(n_clusters=8, n_init=10).fit(points_without_seg).labels_ + group_ids.max() + 1
        group_ids[group_ids == -1] = other_ins
        unique_ids = np.unique(group_ids)
        if group_ids.max() != len(unique_ids) - 1:
            new_group_ids = np.zeros_like(group_ids)
            for i, ids in enumerate(unique_ids):
                new_group_ids[group_ids == ids] = i
            group_ids = new_group_ids
        
        # Format output, no need for boxes, only ins/sem mask is OK
        group_ids.astype(np.int64).tofile(
            os.path.join("super_points/%s" % scan_name, "%s.bin" % (40*frame_i+1)))
        unaligned_xyz.astype(np.float32).tofile(
            os.path.join("points/%s" % scan_name, "%s.bin" % (40*frame_i+1)))
        sem.astype(np.int64).tofile(
            os.path.join("semantic_mask/%s" % scan_name, "%s.bin" % (40*frame_i+1)))
        ins.astype(np.int64).tofile(
            os.path.join("instance_mask/%s" % scan_name, "%s.bin" % (40*frame_i+1)))


def make_split(mask_generator, path_prefix, scan_name_list):
    os.makedirs("points", exist_ok=True)
    os.makedirs("super_points", exist_ok=True)
    os.makedirs("semantic_mask", exist_ok=True)
    os.makedirs("instance_mask", exist_ok=True)
    os.makedirs("axis_align_matrix", exist_ok=True)

    for scan_name_index, scan_name in enumerate(tqdm(scan_name_list)):
        cur_parameter = {}
        cur_parameter["scan_name_index"] = scan_name_index
        cur_parameter["scan_name"] = scan_name
        cur_parameter["path_prefix"] = path_prefix
        cur_parameter["scan_num"] = len(scan_name_list)

        process_cur_scan(cur_parameter, mask_generator)


def main():
    PATH_PREFIX = "./SceneNN"
    scene_name_list = sorted(os.listdir(PATH_PREFIX))
    mask_generator = SamAutomaticMaskGenerator(build_sam(
        checkpoint="../sam_vit_h_4b8939.pth").to(device="cuda"))
    
    make_split(mask_generator, PATH_PREFIX, scene_name_list)


if __name__ == "__main__":
    main()
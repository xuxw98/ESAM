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
from load_scannet_data import export
from tqdm import tqdm


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

def export_one_scan(scan_name):
    mesh_file = os.path.join('3RScan', scan_name, 'labels.instances.annotated.v2.ply')
    agg_file = os.path.join('3RScan', scan_name, 'semseg.v2.json')
    seg_file = os.path.join('3RScan', scan_name, 'mesh.refined.0.010000.segs.v2.json')
    meta_file = os.path.join('3RScan', scan_name, 'sequence', '_info.txt')
    aligned_mesh_vertices, instance_labels, bboxes, label_map, object_id_to_label = \
        export(mesh_file, agg_file, seg_file, meta_file, './3RScan.v2 Semantic Classes - Mapping.tsv', scannet200=False)
        
    # change label for class-agnostic setting
    for key in object_id_to_label.keys():
        if label_map[object_id_to_label[key]] != 1 and label_map[object_id_to_label[key]] != 2:
            object_id_to_label[key] = 'chair'
            
    # bbox_instance_labels = np.arange(1,bboxes.shape[0]+1)
    bbox_instance_labels = np.array(list(object_id_to_label.keys()))
    return aligned_mesh_vertices, instance_labels, label_map, object_id_to_label, bboxes, bbox_instance_labels


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

def read_info(info_path):
    with open(info_path, 'r') as file:
        file_content = file.read()
    key_value_pairs = file_content.split('\n')[:-1]
    data_dict = {}
    for pair in key_value_pairs:
        key, value = pair.split(' = ')
        data_dict[key.strip()] = value.strip()
        
    colorIntrinsic = data_dict['m_calibrationColorIntrinsic']
    colorIntrinsic = colorIntrinsic.split(' ')
    colorIntrinsic = [float(x) for x in colorIntrinsic]
    data_dict['m_calibrationColorIntrinsic'] = np.array(colorIntrinsic).reshape((4, 4))
    depthIntrinsic = data_dict['m_calibrationDepthIntrinsic']
    depthIntrinsic = depthIntrinsic.split(' ')
    depthIntrinsic = [float(x) for x in depthIntrinsic]
    data_dict['m_calibrationDepthIntrinsic'] = np.array(depthIntrinsic).reshape((4, 4))
    return data_dict

def process_cur_scan(cur_scan, mask_generator):
    scan_name_index = cur_scan["scan_name_index"]
    scan_name = cur_scan["scan_name"]
    path_prefix = cur_scan["path_prefix"]
    scan_num = cur_scan["scan_num"]
    print(scan_name)

    scan_path = os.path.join(path_prefix,scan_name)

    # axis_align_matrix_path = os.path.join(AXIS_ALIGN_MATRIX_PATH, "%s"%(scan_name),"%s.txt"%(scan_name))
    # lines = open(axis_align_matrix_path).readlines()
    # for line in lines:
    #     if 'axisAlignment' in line:
    #         axis_align_matrix = [float(x) \
    #             for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    #         break
    # axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

    info_path = os.path.join(scan_path, 'sequence','_info.txt')
    info = read_info(info_path)
    
    unify_dim = (224, 172)
    unify_intrinsic = adjust_intrinsic(info['m_calibrationDepthIntrinsic'], unify_dim, unify_dim)
        
    # Sort string. 0 20 40 60 80 100 120 ...
    all_files_list = os.listdir(scan_path+'/sequence')
    depth_map_list = []
    POSE_txt_list = []
    rgb_map_list = []

    for file_name in all_files_list:
        if file_name.endswith('.depth.pgm'):
            depth_map_list.append(file_name)
        elif file_name.endswith('.pose.txt'):
            POSE_txt_list.append(file_name)
        elif file_name.endswith('.color.jpg'):
            rgb_map_list.append(file_name)
    POSE_txt_list = sorted(POSE_txt_list, key=lambda x: int(x[6:-9]))
    rgb_map_list = sorted(rgb_map_list, key=lambda x: int(x[6:-10]))
    depth_map_list = sorted(depth_map_list, key=lambda x: int(x[6:-10]))
    # POSE_txt_list = sorted(os.listdir(os.path.join(scan_path, 'pose')), key=lambda s: int(s[:-4]))
    # rgb_map_list = sorted(os.listdir(os.path.join(scan_path, 'color')), key=lambda s: int(s[:-4]))
    # depth_map_list = sorted(os.listdir(os.path.join(scan_path, 'depth')), key=lambda s: int(s[:-4]))

    poses = [load_matrix_from_txt(os.path.join(scan_path, 'sequence', path)) for path in POSE_txt_list]
    aligned_poses = poses.copy()

    os.makedirs("points/%s" % scan_name, exist_ok=True)
    os.makedirs("super_points/%s" % scan_name, exist_ok=True)
    os.makedirs("semantic_mask/%s" % scan_name, exist_ok=True)
    os.makedirs("instance_mask/%s" % scan_name, exist_ok=True)

    aligned_mesh_vertices, instance_labels, label_map, object_id_to_label, \
        aligned_bboxes, bbox_instance_labels = export_one_scan(scan_name)

    for frame_i, (rgb_map_name, \
        depth_map_name, \
        pose, \
        aligned_pose) \
        in enumerate(zip(rgb_map_list, depth_map_list, poses, aligned_poses)):
        # assert frame_i * 20 == int(rgb_map_name[:-4])
        # set interval=5
        if frame_i % 5 != 0:
        # if frame_i % 2 == 0:
            continue

        depth_map = cv2.imread(os.path.join(scan_path, 'sequence', depth_map_name), -1)
        color_map = cv2.imread(os.path.join(scan_path, 'sequence', rgb_map_name))
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        color_map = cv2.resize(color_map, depth_map.shape[::-1])
        color_map = cv2.rotate(color_map, cv2.ROTATE_90_CLOCKWISE)
        # SAM-->super point
        masks = mask_generator.generate(color_map)
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        color_map = cv2.rotate(color_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        group_ids = np.full((color_map.shape[0], color_map.shape[1]), -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0
        for i in range(num_masks):
            mask_now = masks[i]["segmentation"]
            # 将mask_now逆时针旋转90
            mask_now = cv2.rotate(mask_now.astype(int), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)
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
        aligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, aligned_pose)
        if np.isnan(aligned_xyz).any():
            continue
        unaligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, pose)
        unaligned_xyz = np.concatenate([unaligned_xyz, rgb], axis=-1)
        xyz_all = np.concatenate([unaligned_xyz, aligned_xyz, group_ids.reshape(-1,1)], axis=-1)
        xyz_all = random_sampling(xyz_all, 20000)
        unaligned_xyz, aligned_xyz, group_ids = xyz_all[:, :6], xyz_all[:, 6:9], xyz_all[:, 9]

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

        # Get sem from ins
        sem = np.zeros_like(ins, dtype=np.uint32)
        for ins_ids in np.unique(ins):
            if ins_ids != 0:
                sem[ins == ins_ids] = label_map[object_id_to_label[ins_ids]]
        
        # Get superpoints
        # TODO: set other_ins_num as 10-->8
        points_without_seg = unaligned_xyz[group_ids == -1]
        if len(points_without_seg) < 8:
            other_ins = np.zeros(len(points_without_seg), dtype=np.int64) + group_ids.max() + 1
        else:
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
            os.path.join("super_points/%s" % scan_name, "%s.bin" % (frame_i)))
        unaligned_xyz.astype(np.float32).tofile(
            os.path.join("points/%s" % scan_name, "%s.bin" % (frame_i)))
        sem.astype(np.int64).tofile(
            os.path.join("semantic_mask/%s" % scan_name, "%s.bin" % (frame_i)))
        ins.astype(np.int64).tofile(
            os.path.join("instance_mask/%s" % scan_name, "%s.bin" % (frame_i)))


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
    PATH_PREFIX = "./3RScan"
    scene_name_list = sorted(os.listdir(PATH_PREFIX))

    mask_generator = SamAutomaticMaskGenerator(build_sam(
        checkpoint="../sam_vit_h_4b8939.pth").to(device="cuda"))
    
    make_split(mask_generator, PATH_PREFIX, scene_name_list)


if __name__ == "__main__":
    main()
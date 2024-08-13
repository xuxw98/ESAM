# V2: 2d_point+3d_ins+knn-->2d_ins-->2d_sem

import enum
import cv2
import shutil
import numpy as np
import math
from scipy import stats
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from segment_anything import build_sam, SamAutomaticMaskGenerator
import pdb
import torch
import pointops
from load_scannet_data import export
from fastsam import FastSAM
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
    mesh_file = os.path.join('3D', scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join('3D', scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join('3D', scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join('3D', scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
    aligned_mesh_vertices, instance_labels, bboxes, label_map, object_id_to_label = \
        export(mesh_file, agg_file, seg_file, meta_file, 'meta_data/scannetv2-labels.combined.tsv', scannet200=True)
    
    bbox_instance_labels = np.arange(1,bboxes.shape[0]+1)

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

def format_result(result):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

def process_cur_scan(cur_scan, mask_generator):
    scan_name_index = cur_scan["scan_name_index"]
    scan_name = cur_scan["scan_name"]
    path_dict = cur_scan["path_dict"]
    scan_num = cur_scan["scan_num"]
    print(scan_name)

    DATA_PATH = path_dict["DATA_PATH"]
    INS_DATA_PATH = path_dict["INS_DATA_PATH"]
    TARGET_DIR = path_dict["TARGET_DIR"]
    AXIS_ALIGN_MATRIX_PATH = path_dict["AXIS_ALIGN_MATRIX_PATH"]

    scan_name = scan_name.strip("\n")
    scan_path = os.path.join(DATA_PATH,scan_name)
    ins_data_path = os.path.join(INS_DATA_PATH,scan_name)
    path_dict["scan_path"] = scan_path

    axis_align_matrix_path = os.path.join(AXIS_ALIGN_MATRIX_PATH, "%s"%(scan_name),"%s.txt"%(scan_name))
    lines = open(axis_align_matrix_path).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

    unify_dim = (640, 480)
    unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
    
    # Sort string. 0 20 40 60 80 100 120 ...
    POSE_txt_list = sorted(os.listdir(os.path.join(scan_path, 'pose')), key=lambda s: int(s[:-4]))
    rgb_map_list = sorted(os.listdir(os.path.join(scan_path, 'color')), key=lambda s: int(s[:-4]))
    depth_map_list = sorted(os.listdir(os.path.join(scan_path, 'depth')), key=lambda s: int(s[:-4]))

    poses = [load_matrix_from_txt(os.path.join(scan_path, 'pose', path)) for path in POSE_txt_list]
    aligned_poses = [np.dot(axis_align_matrix, pose) for pose in poses]

    aligned_mesh_vertices, instance_labels, label_map, object_id_to_label, \
        aligned_bboxes, bbox_instance_labels = export_one_scan(scan_name)

    for frame_i, (rgb_map_name, \
        depth_map_name, \
        pose, \
        aligned_pose) \
        in enumerate(zip(rgb_map_list, depth_map_list, poses, aligned_poses)):
        assert frame_i * 20 == int(rgb_map_name[:-4])
        # TODO: keep the same number as 25k by use 5 (interval=100)
        # or reduce storage by use 10 (interval=200)
        if frame_i % 10 != 0:
        # if frame_i % 5 != 0:
            continue

        depth_map = cv2.imread(os.path.join(scan_path, 'depth', depth_map_name), -1)
        color_map = cv2.imread(os.path.join(scan_path, 'color', rgb_map_name))
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

        img_path = os.path.join(scan_path, 'color', rgb_map_name)
        
        # # SAM-->super point
        # masks = mask_generator.generate(color_map)
        everything_result = mask_generator(img_path, device='cuda', retina_masks=True, imgsz=640, conf=0.1, iou=0.9,)
        try:
            masks = format_result(everything_result[0])
        except:
            everything_result = mask_generator(img_path, device='cuda', retina_masks=True, imgsz=640, conf=0.1, iou=0.7,)
            masks = format_result(everything_result[0])
    
        # SAM-->super point
        # masks = mask_generator.generate(color_map)
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

        # For SV: zero-centered, downsample to 20000
        aligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, aligned_pose)
        if np.isnan(aligned_xyz).any():
            continue
        unaligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, pose)
        xyz_offset = np.mean(unaligned_xyz, axis=0)
        unaligned_xyz -= xyz_offset
        pose_centered = pose.copy()
        pose_centered[:3, 3] -= xyz_offset
        if not os.path.exists('./pose_centered/'+scan_name):
            os.makedirs('./pose_centered/'+scan_name)
        np.save('./pose_centered/'+scan_name+'/'+ rgb_map_name.replace('.jpg', '.npy'), pose_centered)

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
        # if object_num <= 2:
            # continue

        # Get sem from ins
        sem = np.zeros_like(ins, dtype=np.uint32)
        for ins_ids in np.unique(ins):
            if ins_ids != 0:
                sem[ins == ins_ids] = label_map[object_id_to_label[ins_ids]]
        
        # Get superpoints
        # TODO: set other_ins_num as 10
        points_without_seg = unaligned_xyz[group_ids == -1]
        if len(points_without_seg) < 20:
            other_ins = np.zeros(len(points_without_seg), dtype=np.int64) + group_ids.max() + 1
        else:
            other_ins = KMeans(n_clusters=20, n_init=10).fit(points_without_seg).labels_ + group_ids.max() + 1
        # other_ins = KMeans(n_clusters=10, n_init=10).fit(points_without_seg).labels_ + group_ids.max() + 1
        group_ids[group_ids == -1] = other_ins
        unique_ids = np.unique(group_ids)
        if group_ids.max() != len(unique_ids) - 1:
            new_group_ids = np.zeros_like(group_ids)
            for i, ids in enumerate(unique_ids):
                new_group_ids[group_ids == ids] = i
            group_ids = new_group_ids

        # Format output, no need for boxes, only ins/sem mask is OK
        np.save(os.path.join(TARGET_DIR, scan_name + "_%s_sp_label.npy" % (20*frame_i)), group_ids)
        np.save(os.path.join(TARGET_DIR, scan_name + "_%s_vert.npy" % (20*frame_i)), unaligned_xyz)
        np.save(os.path.join(TARGET_DIR, scan_name + "_%s_sem_label.npy" % (20*frame_i)), sem)
        np.save(os.path.join(TARGET_DIR, scan_name + "_%s_ins_label.npy" % (20*frame_i)), ins)
        np.save(os.path.join(TARGET_DIR, scan_name + "_%s_axis_align_matrix.npy" % (20*frame_i)), axis_align_matrix)


def make_split(mask_generator, path_dict, split="train"):
    TARGET_DIR = path_dict["TARGET_DIR_PREFIX"]
    path_dict["TARGET_DIR"] = TARGET_DIR
    os.makedirs(TARGET_DIR, exist_ok=True)
    f = open("meta_data/scannetv2_%s.txt"%(split))
    scan_name_list = sorted(f.readlines())

    for scan_name_index, scan_name in enumerate(tqdm(scan_name_list)):
        cur_parameter = {}
        cur_parameter["scan_name_index"] = scan_name_index
        cur_parameter["scan_name"] = scan_name
        # if scan_name != 'scene0568_02\n':
        #     continue
        cur_parameter["path_dict"] = path_dict
        cur_parameter["scan_num"] = len(scan_name_list)
        process_cur_scan(cur_parameter, mask_generator)


def main():
    DATA_PATH = "./2D"
    TARGET_DIR_PREFIX = "./scannet_sv_instance_data"
    INS_DATA_PATH = "./2D" # Replace it with the path to 2D
    AXIS_ALIGN_MATRIX_PATH = "./3D" # Replace it with the path to axis_align_matrix path

    path_dict = {"DATA_PATH": DATA_PATH,
                "TARGET_DIR_PREFIX": TARGET_DIR_PREFIX,
                "INS_DATA_PATH": INS_DATA_PATH,
                "AXIS_ALIGN_MATRIX_PATH": AXIS_ALIGN_MATRIX_PATH       
                }

    splits = ["train", "val"]

    mask_generator = FastSAM('../FastSAM-x.pt')
    
    for cur_split in splits:
        make_split(mask_generator, path_dict, cur_split)


if __name__ == "__main__":
    main()
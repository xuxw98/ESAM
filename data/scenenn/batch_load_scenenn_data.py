# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""
import argparse
import datetime
import os
from os import path as osp

import torch
import segmentator
import open3d as o3d
import numpy as np
import scannet_utils
import json

DONOTCARE_CLASS_IDS = np.array([])

SCANNET_OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

def export(mesh_file, xml_file, label_map_file):
    label_map = scannet_utils.read_label_mapping(label_map_file, label_from='nyu40class', label_to='nyu40id')
    # breakpoint()
    mesh, label_txt, ins_label, aligned_bboxes, object_id_to_label = scannet_utils.read_mesh_vertices_rgb(mesh_file, xml_file)
    sem_label = []
    for i in range(label_txt.shape[0]):
        if label_txt[i] == 'unknown':
            sem_label.append(0)
            ins_label[i] = 0
            label_txt[i] = 'otherprop'
        elif label_txt[i] == 'otherprop':
            sem_label.append(40)
        else:
            try:
                sem_label.append(label_map[label_txt[i]])
            except:
                sem_label.append(0)
                ins_label[i] = 0
    sem_label = np.array(sem_label)
    instance_bboxes = extract_bbox(mesh, ins_label, label_map, object_id_to_label)
    return mesh, ins_label, sem_label, instance_bboxes, label_map, object_id_to_label

def extract_bbox(mesh, ins_label, label_map, object_id_to_label):
    num_instances = len(np.unique(ins_label)) - 1
    instance_bboxes = np.zeros((num_instances, 7))
    object_id_to_label_ = list(object_id_to_label.values())
    for obj_id in range(1, num_instances + 1):
        sem_label = label_map[object_id_to_label_[obj_id-1]]
        obj_pc = mesh[ins_label == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        xyz_min = np.min(obj_pc, axis=0)
        xyz_max = np.max(obj_pc, axis=0)
        bbox = np.concatenate([(xyz_min + xyz_max) / 2.0, xyz_max - xyz_min, [sem_label]])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox
    return instance_bboxes

def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False,
                    scannet200=False):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '.ply')
    xml_file = osp.join(scannet_dir, scan_name, scan_name + '.xml')
    # includes axisAlignment info for the train set scans.
    mesh_vertices, instance_labels, semantic_labels, aligned_bboxes, label_map, _ = export(mesh_file, xml_file, label_map_file)

    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')
        OBJ_CLASS_IDS = SCANNET_OBJ_CLASS_IDS

        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        print(f'Num of care instances: {aligned_bboxes.shape[0]}')
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoints = segmentator.segment_mesh(vertices, faces).numpy()
        
    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            superpoints = superpoints[choices]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
            if not os.path.exists('downsample_indices'):
                os.makedirs('downsample_indices')
            np.save(f'downsample_indices/{scan_name}.npy', choices)
            
    segs_dict = {}
    segs_dict['scene_Id'] = scan_name
    segs_dict['segIndices'] = superpoints.tolist()
    # 存为json文件
    if not os.path.exists('mesh_segs'):
        os.makedirs('mesh_segs')
    with open(f'mesh_segs/{scan_name}.segs.json', 'w') as f:
        json.dump(segs_dict, f)
    
    np.save(f'{output_filename_prefix}_sp_label.npy', superpoints)
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)

    unaligned_bboxes = aligned_bboxes.copy()
    if not test_mode:
        assert superpoints.shape == semantic_labels.shape
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy',
                unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)


def batch_export(max_num_point,
                 output_folder,
                 label_map_file,
                 scannet_dir):
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = sorted(os.listdir(scannet_dir))
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        # try:
        export_one_scan(scan_name, output_filename_prefix, max_num_point,
                        label_map_file, scannet_dir, False)
        # except Exception:
        #     print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=500000,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./scenenn_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--val_scenenn_dir', default='./SceneNN', help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='./scannetv2-labels.combined.tsv',
        help='The path of label map file.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.label_map_file,
        args.val_scenenn_dir)


if __name__ == '__main__':
    main()

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
from load_scannet_data import export

DONOTCARE_CLASS_IDS = np.array([])

SCANNET_OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False,
                    scannet200=True):
    mesh_file = os.path.join('3RScan', scan_name, 'labels.instances.annotated.v2.ply')
    agg_file = os.path.join('3RScan', scan_name, 'semseg.v2.json')
    seg_file = os.path.join('3RScan', scan_name, 'mesh.refined.0.010000.segs.v2.json')
    meta_file = os.path.join('3RScan', scan_name, 'sequence', '_info.txt')

    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic = export(mesh_file, agg_file, seg_file, meta_file, label_map_file, None, test_mode, scannet200)
    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')
        OBJ_CLASS_IDS = SCANNET_OBJ_CLASS_IDS

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

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
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
                superpoints = superpoints[choices]
                
    np.save(f'{output_filename_prefix}_sp_label.npy', superpoints)
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)

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
                 scannet_dir,
                 test_mode=False,
                 scannet200=False):
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
        export_one_scan(scan_name, output_filename_prefix, max_num_point,
                        label_map_file, scannet_dir, test_mode, scannet200)
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./3rscan_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--val_scannet_dir', default='./3RScan', help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='./3RScan.v2 Semantic Classes - Mapping.tsv',
        help='The path of label map file.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.label_map_file,
        args.val_scannet_dir,
        test_mode=False,
        scannet200=True)


if __name__ == '__main__':
    main()

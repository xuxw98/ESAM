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

SCANNET200_OBJ_CLASS_IDS = np.array([2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                                    72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
                                    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
                                    488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191])



def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False,
                    scannet200=False):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '.aggregation.json')
    seg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '_vh_clean_2.0.010000.segs.json')
    # includes axisAlignment info for the train set scans.
    meta_file = osp.join(scannet_dir, scan_name, f'{scan_name}.txt')
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode, scannet200)

    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')
        if scannet200:
            OBJ_CLASS_IDS = SCANNET200_OBJ_CLASS_IDS
        else:
            OBJ_CLASS_IDS = SCANNET_OBJ_CLASS_IDS

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoints = segmentator.segment_mesh(vertices, faces).numpy()
    
    np.save(f'{output_filename_prefix}_sp_label.npy', superpoints)
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)

    if not test_mode:
        assert superpoints.shape == semantic_labels.shape
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy',
                unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy',
                axis_align_matrix)


def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 label_map_file,
                 scannet_dir,
                 test_mode=False,
                 scannet200=False):
    if test_mode and not os.path.exists(scannet_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            label_map_file, scannet_dir, test_mode, scannet200)
        except Exception:
            print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./scannet_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--train_scannet_dir', default='scans', help='scannet data directory.')
    parser.add_argument(
        '--test_scannet_dir',
        default='scans_test',
        help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='meta_data/scannetv2-labels.combined.tsv',
        help='The path of label map file.')
    parser.add_argument(
        '--train_scan_names_file',
        default='meta_data/scannet_train.txt',
        help='The path of the file that stores the scan names.')
    parser.add_argument(
        '--test_scan_names_file',
        default='meta_data/scannetv2_test.txt',
        help='The path of the file that stores the scan names.')
    parser.add_argument(
        '--scannet200',
        action='store_true',
        help='Use it for scannet200 mapping')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.train_scannet_dir,
        test_mode=False,
        scannet200=args.scannet200)
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.test_scannet_dir,
        test_mode=True,
        scannet200=args.scannet200)


if __name__ == '__main__':
    main()

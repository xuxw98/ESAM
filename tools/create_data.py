# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from indoor_converter import create_indoor_info_file
from update_infos_to_v2 import update_pkl_infos


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    # info_test_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_test.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)
    # update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_test_path)

def scenenn_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scenenn dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)

def ThreeRScan_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scenenn dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset in ('scannet', 'scannet_sv', 'scannet_mv', 'scannet200', 'scannet200_sv', 'scannet200_mv'):
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset in ('scenenn','scenenn_mv'):
        scenenn_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset in ('3rscan', '3rscan_mv'):
        ThreeRScan_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')

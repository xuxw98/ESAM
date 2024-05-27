# Modified from mmdetection3d/tools/dataset_converters/indoor_converter.py
# We just support ScanNet 200.
import os

import mmengine

from scannet_data_utils import ScanNetData, ScanNetSVData, ScanNetMVData
from scenenn_data_utils import SceneNNData, SceneNNMVData
from trscan_data_utils import TRScanData, TRScanMVData


def create_indoor_info_file(data_path,
                            pkl_prefix='sunrgbd',
                            save_path=None,
                            use_v1=False,
                            workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        use_v1 (bool, optional): Whether to use v1. Default: False.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['scannet', 'scannet_sv', 'scannet_mv', 'scannet200', 'scannet200_sv', 'scannet200_mv',
                          'scenenn', 'scenenn_mv', '3rscan', '3rscan_mv'], \
        f'unsupported indoor dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    train_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_train.pkl')
    val_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_val.pkl')
    # test_filename = os.path.join(
    #     save_path, f'{pkl_prefix}_oneformer3d_infos_test.pkl')
    if pkl_prefix == 'scannet':
        # ScanNet has a train-val-test split
        train_dataset = ScanNetData(root_path=data_path, split='train')
        val_dataset = ScanNetData(root_path=data_path, split='val')
        # test_dataset = ScanNetData(root_path=data_path, split='test')
    elif pkl_prefix == 'scannet_sv':
        train_dataset = ScanNetSVData(root_path=data_path, split='train', save_path=save_path)
        val_dataset = ScanNetSVData(root_path=data_path, split='val', save_path=save_path)    
    elif pkl_prefix == 'scannet_mv':
        train_dataset = ScanNetMVData(root_path=data_path, split='train', save_path=save_path)
        val_dataset = ScanNetMVData(root_path=data_path, split='val', save_path=save_path)
    elif pkl_prefix == 'scannet200':  # ScanNet200
        # ScanNet has a train-val-test split
        train_dataset = ScanNetData(root_path=data_path, split='train',
                                    scannet200=True, save_path=save_path)
        val_dataset = ScanNetData(root_path=data_path, split='val',
                                    scannet200=True, save_path=save_path)
        # test_dataset = ScanNetData(root_path=data_path, split='test',
        #                             scannet200=True, save_path=save_path)
    elif pkl_prefix == 'scannet200_sv':   # ScanNet200-SV
        train_dataset = ScanNetSVData(root_path=data_path, split='train',
                                    scannet200=True, save_path=save_path)
        val_dataset = ScanNetSVData(root_path=data_path, split='val',
                                    scannet200=True, save_path=save_path)
    elif pkl_prefix == 'scannet200_mv':   # ScanNet200-MV
        train_dataset = ScanNetMVData(root_path=data_path, split='train',
                                    scannet200=True, save_path=save_path)
        val_dataset = ScanNetMVData(root_path=data_path, split='val',
                                    scannet200=True, save_path=save_path)
    elif pkl_prefix == 'scenenn':
        val_dataset = SceneNNData(root_path=data_path, split='val')
    elif pkl_prefix == 'scenenn_mv':
        val_dataset = SceneNNMVData(root_path=data_path, split='val')
    elif pkl_prefix == '3rscan':
        val_dataset = TRScanData(root_path=data_path, split='val')
    elif pkl_prefix == '3rscan_mv':
        val_dataset = TRScanMVData(root_path=data_path, split='val')
    else:
        raise NotImplementedError("No dataset: %s" % pkl_prefix)

    if 'scannet' in pkl_prefix:
        infos_train = train_dataset.get_infos(
            num_workers=workers, has_label=True)
        mmengine.dump(infos_train, train_filename, 'pkl')
        print(f'{pkl_prefix} info train file is saved to {train_filename}')

    infos_val = val_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmengine.dump(infos_val, val_filename, 'pkl')
    print(f'{pkl_prefix} info val file is saved to {val_filename}')

    # infos_test = test_dataset.get_infos(
    #     num_workers=workers, has_label=False)
    # mmengine.dump(infos_test, test_filename, 'pkl')
    # print(f'{pkl_prefix} info test file is saved to {test_filename}')

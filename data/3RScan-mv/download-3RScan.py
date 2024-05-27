#!/usr/bin/env python
# Downloads 3RScan public data release

# The data is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.
# Some useful info: Each scan is identified by a unique ID listed here:
# http://campar.in.tum.de/public_datasets/3RScan/scans.txt or https://github.com/WaldJohannaU/3RScan/tree/master/splits
#
# Script usage:
# - To download the entire 3RScan release: download.py -o [directory in which to download]
# - To download a specific scan (e.g., 19eda6f4-55aa-29a0-8893-8eac3a4d8193): download.py -o [directory in which to download] --id 19eda6f4-55aa-29a0-8893-8eac3a4d8193
# - To download the tfrecords: download.py -o [directory in which to download] --type=tfrecords
# - The corresponding metadata file is here: http://campar.in.tum.de/public_datasets/3RScan/3RScan.json
# - 3D semantic scene graphs for 3RScan are available for download on our project page https://3dssg.github.io

import sys
import argparse
import os

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib
import tempfile
import re

BASE_URL = 'http://campar.in.tum.de/public_datasets/3RScan/'
DATA_URL = BASE_URL + 'Dataset/'
TOS_URL = 'http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf'
TEST_FILETYPES = ['mesh.refined.v2.obj', 'mesh.refined.mtl', 'mesh.refined_0.png', 'sequence.zip']
# We only provide semantic annotations for the train and validation scans as well as the for the
# reference scans in the test set.
FILETYPES = TEST_FILETYPES + ['labels.instances.annotated.v2.ply', 'mesh.refined.0.010000.segs.v2.json', 'semseg.v2.json']

RELEASE = 'release_scans.txt'
HIDDEN_RELEASE = 'test_rescans.txt'

RELEASE_SIZE = '~94GB'
id_reg = re.compile(r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}")

def get_scans(scan_file):
    scan_lines = urllib.urlopen(scan_file)
    scans = []
    for scan_line in scan_lines:
        scan_line = scan_line.decode('utf8').rstrip('\n')
        match = id_reg.search(scan_line)
        if match:
            scan_id = match.group()
            scans.append(scan_id)
    return scans

def download_release(release_scans, out_dir, file_types):
    print('Downloading 3RScan release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types)
    print('Downloaded 3RScan release.')

def download_file(url, out_file):
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.urlretrieve(url, out_file_tmp) 
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types):
    print('Downloading 3RScan scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        url = DATA_URL + '/' + scan_id + '/' + ft
        out_file = out_dir + '/' + ft
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)

def download_tfrecord(url, out_dir, file):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, file)
    download_file(url + '/' + file, out_file)


def main():
    parser = argparse.ArgumentParser(description='Downloads 3RScan public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--type', help='specific file type to download')
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the 3RScan terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')

    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)
    file_types = FILETYPES;
    file_types_test = TEST_FILETYPES;

    if args.type:  # download file type
        file_type = args.type
        if file_type == 'tfrecords':
            download_tfrecord(BASE_URL, args.out_dir, 'val-scans.tfrecords')
            download_tfrecord(BASE_URL, args.out_dir, 'train-scans.tfrecords')
            return
        elif file_type not in FILETYPES:
            print('ERROR: Invalid file type: ' + file_type)
            return
        file_types = [file_type]
        if file_type not in TEST_FILETYPES:
            file_types_test = []
        else:
            file_types_test = [file_type]
    if args.id:  # download single scan
        scan_id = args.id
        if scan_id not in release_scans and scan_id not in test_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            out_dir = os.path.join(args.out_dir, scan_id)

            if scan_id in release_scans:
                download_scan(scan_id, out_dir, file_types)
            elif scan_id in test_scans:
                download_scan(scan_id, out_dir, file_types_test)
    else: # download entire release
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire 3RScan release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all 3RScan scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        download_release(release_scans, args.out_dir, file_types)
        download_release(test_scans, args.out_dir, file_types_test)

if __name__ == "__main__": main()
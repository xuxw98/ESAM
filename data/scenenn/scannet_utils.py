# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts '''
import os
import sys
import json
import csv
import pdb
import xml.dom.minidom
import xml.etree.ElementTree as ET
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
import pdb
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename,xml_filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
        label_id_all = plydata['vertex'].data['label']
        with open(xml_filename, 'r', encoding='utf-8') as file:
            xml_data = file.read()
        root = ET.fromstring(xml_data)
        label_id_to_nyu_class = {}
        label_id_to_ins_label = {}
        object_id_to_label = {}
        aabbox_list = []
        for i, label in enumerate(root.findall('label')):
            label_id = label.get('id')
            nyu_class = label.get('nyu_class')
            aabbox = np.array(label.get('aabbox').split(' '),dtype=float)
            aabbox_list.append(aabbox)
            if nyu_class == 'prop' or nyu_class == 'unknown':
                nyu_class = 'otherprop'
            if nyu_class == 'fridge':
                nyu_class = 'refridgerator'
            if nyu_class == 'furniture':
                nyu_class = 'otherfurniture'
            if nyu_class == 'structure':
                nyu_class = 'otherstructure'
            label_id_to_nyu_class[label_id] = nyu_class
            label_id_to_ins_label[label_id] = i+1
            if len(nyu_class) > 0:
                object_id_to_label[i+1] = nyu_class
        aabbox_list = np.array(aabbox_list)
        nyu_class_ret = []
        ins_label_ret = []
        for label_id in label_id_all:
            try:
                nyu_class_ret.append(label_id_to_nyu_class[str(label_id)])
                ins_label_ret.append(label_id_to_ins_label[str(label_id)])
            except:
                nyu_class_ret.append('unknown')
                ins_label_ret.append(0)
        nyu_class_ret = np.array(nyu_class_ret)
        ins_label_ret = np.array(ins_label_ret)
    return vertices, nyu_class_ret, ins_label_ret, aabbox_list, object_id_to_label,


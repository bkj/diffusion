#!/usr/bin/env python

#!/usr/bin/env python

"""
    mat2npy.py
"""

import os
import h5py
import joblib
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, required=True, 
        choices=['oxford5k', 'oxford105k', 'paris6k', 'paris106k'])
    
    parser.add_argument('--feature_type', type=str, required=True,
        choices=['resnet', 'siamac'])
    
    parser.add_argument('--mat_dir', type=str, required=True)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    input_file       = '{}_{}.mat'.format(args.dataset_name, args.feature_type)
    glob_output_file = '{}_{}_glob.npy'.format(args.dataset_name, args.feature_type)
    query_dir        = os.path.join(args.mat_dir, 'query')
    gallery_dir      = os.path.join(args.mat_dir, 'gallery')
    
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir)
    
    with h5py.File(os.path.join(args.mat_dir, input_file), 'r') as f:
        
        glob_q = np.array([f[x[0]][:] for x in f['/glob/Q']])
        glob_q = np.squeeze(glob_q, axis=1)
        np.save(os.path.join(query_dir, glob_output_file), glob_q)
        
        
        glob_g = np.array([f[x[0]][:] for x in f['/glob/V']])
        glob_g = np.squeeze(glob_g, axis=1)
        np.save(os.path.join(gallery_dir, glob_output_file), glob_g)

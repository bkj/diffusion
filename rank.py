#!/usr/bin/env python

"""
    rank.py
"""

# !! Why are the first and second queries identical

import os
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from dataset import Dataset
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing
from evaluate import compute_map_and_print

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',       type=str, default='./cache')
    parser.add_argument('--dataset_name',    type=str, default='oxford5k')
    parser.add_argument('--query_path',      type=str, default='data/query/oxford5k_resnet_glob.npy')
    parser.add_argument('--gallery_path',    type=str, default='data/gallery/oxford5k_resnet_glob.npy')
    parser.add_argument('--gnd_path',        type=str, default='data/gnd_oxford5k.pkl')
    parser.add_argument('--n_trunc',         type=int, default=1000)
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args

# --
# IO

args = parse_args()
os.makedirs(args.cache_dir, exist_ok=True)

dataset = Dataset(args.query_path, args.gallery_path)
queries, gallery = dataset.queries, dataset.gallery

gnd_name = os.path.splitext(os.path.basename(args.gnd_path))[0]
gnd      = pickle.load(open(args.gnd_path, 'rb'))['gnd']

# --
# Search

n_query   = len(queries)

diffusion = Diffusion(features=np.vstack([queries, gallery]), kd=args.kd)
features  = diffusion.get_offline_results(n_trunc=args.n_trunc)
features  = preprocessing.normalize(features, norm="l2", axis=1)

scores    = features[:n_query] @ features[n_query:].T
if hasattr(scores, 'todense'):
    scores = scores.todense()

ranks     = np.argsort(-scores)

compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)

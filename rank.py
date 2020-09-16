#!/usr/bin/env python

"""
    rank.py
"""

import os
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import normalize
from diffusion import TruncatedDiffusion
from evaluate import compute_map_and_print

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',    type=str, default='oxford5k')
    parser.add_argument('--query_path',      type=str, default='data/query/oxford5k_resnet_glob.npy')
    parser.add_argument('--gallery_path',    type=str, default='data/gallery/oxford5k_resnet_glob.npy')
    parser.add_argument('--gnd_path',        type=str, default='data/gnd_oxford5k.pkl')
    parser.add_argument('--n_trunc',         type=int, default=1000) # Diffuse over subgraph of size n_trunc
    parser.add_argument('--kd',              type=int, default=50)   # K in KNN graph
    args = parser.parse_args()
    return args

# --
# IO

args = parse_args()

X_queries = np.load(args.query_path)
X_gallery = np.load(args.gallery_path)
X         = np.vstack([X_queries, X_gallery])

n_query   = X_queries.shape[0]

# --
# Search

features = TruncatedDiffusion(features=X, kd=args.kd).run(n_trunc=args.n_trunc)
features = normalize(features, norm='l2', axis=1) # !! Important (for Oxford5k, at least)

scores = features[:n_query] @ features[n_query:].T
if hasattr(scores, 'todense'):
    scores = scores.todense()

ranks = np.argsort(-scores)

# --
# Evaluate

gnd_name = os.path.splitext(os.path.basename(args.gnd_path))[0]
gnd      = pickle.load(open(args.gnd_path, 'rb'))['gnd']
compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)

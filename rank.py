#!/usr/bin/env python

"""

"""

# !! Why are the first and second queries identical


import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing
from evaluate import compute_map_and_print


def search_old(gamma=3):
    diffusion = Diffusion(gallery, args.cache_dir)
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)

    time0 = time.time()
    print('[search] 1) k-NN search')
    sims, ids = diffusion.knn.search(queries, args.kq)
    sims      = sims ** gamma
    qr_num    = ids.shape[0]

    print('[search] 2) linear combination')
    all_scores = np.empty((qr_num, args.truncation_size), dtype=np.float32)
    all_ranks  = np.empty((qr_num, args.truncation_size), dtype=np.int)
    for i in tqdm(range(qr_num), desc='[search] query'):
        scores = sims[i] @ offline[ids[i]]
        parts  = np.argpartition(-scores, args.truncation_size)[:args.truncation_size]
        ranks  = np.argsort(-scores[parts])
        all_scores[i] = scores[parts][ranks]
        all_ranks[i]  = parts[ranks]
    
    print('[search] search costs {:.2f}s'.format(time.time() - time0))

    # 3) evaluation
    evaluate(all_ranks)

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

# --
# Search

n_query   = len(queries)

diffusion = Diffusion(features=np.vstack([queries, gallery]), cache_dir=args.cache_dir)
offline   = diffusion.get_offline_results(n_trunc=args.n_trunc, kd=args.kd)

features  = preprocessing.normalize(offline, norm="l2", axis=1)
scores    = features[:n_query] @ features[n_query:].T
ranks     = np.argsort(-scores.todense())

# --
# Evaluate

gnd_name = os.path.splitext(os.path.basename(args.gnd_path))[0]

gnd = pickle.load(open(args.gnd_path, 'rb'))['gnd']

compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)

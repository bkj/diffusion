#!/usr/bin/env python

"""
    scratch/active-learning.py
"""

SUPRESS_WARNINGS = True
if SUPRESS_WARNINGS:
    import sys
    def warn(*args, **kwargs): pass
    
    import warnings
    warnings.warn = warn

import sys
sys.path.append('.')

import json
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from sklearn import metrics
from sklearn.preprocessing import normalize

from sklearn_extra.cluster import KMedoids
from diffusion import VanillaDiffusion
from helpers import squeezed_array, permute_data, metric_fns

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name', type=str, default='Adiac')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# --
# IO

df_train = pd.read_csv(f'data/ucr/{args.prob_name}/{args.prob_name}_TRAIN.tsv', header=None, sep='\t')
X_train, y_train = df_train.values[:,1:], df_train.values[:,0]
X_train, y_train = permute_data(X_train, y_train)

df_test = pd.read_csv(f'data/ucr/{args.prob_name}/{args.prob_name}_TEST.tsv', header=None, sep='\t')
X_test, y_test = df_test.values[:,1:], df_test.values[:,0]
X_test, y_test = permute_data(X_test, y_test)

X       = np.vstack([X_test, X_train])
y       = np.hstack([y_test, y_train])
n_test  = X_test.shape[0]
n_train = X_train.shape[0]

print('(n_train, n_test)', (n_train, n_test))

X = normalize(X, 'l2', axis=1)

metric_fn = metric_fns['f1'] if len(set(y_train)) == 2 else metric_fns['f1_macro']

# --

n_trunc = np.inf
kd      = 8
sym_fn  = 'mean'

orig_scores = VanillaDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9).run()
scores      = orig_scores.copy()

euc_dists = squareform(pdist(X, metric='cosine'))

np.fill_diagonal(scores, np.inf)

# >>

k = int(0.3 * n_train)

# KMedoids
kmed     = KMedoids(n_clusters=k, metric='precomputed', max_iter=1000)
train    = kmed.fit(orig_scores.max() - orig_scores).medoid_indices_
pred_idx = orig_scores[:,train].argmax(axis=-1)
print('kmedoid', metric_fn(y, y[train][pred_idx]))

# Heuristic
train    = scores.max(axis=0).argsort()[-k:]
pred_idx = orig_scores[:,train].argmax(axis=-1)
print('heuristic', metric_fn(y, y[train][pred_idx]))

# Random
train    = np.random.choice(X.shape[0], k, replace=False)
pred_idx = orig_scores[:,train].argmax(axis=-1)
print('random', metric_fn(y, y[train][pred_idx]))

# Random (cosine euclidean distance)
train    = np.random.choice(X.shape[0], k, replace=False)
pred_idx = euc_dists[:,train].argmin(axis=-1)
print('random (euc)', metric_fn(y, y[train][pred_idx]))

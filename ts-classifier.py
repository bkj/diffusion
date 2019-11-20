#!/usr/bin/env python

"""
    classifier.py
"""

SUPRESS_WARNINGS = True
if SUPRESS_WARNINGS:
    import sys
    def warn(*args, **kwargs): pass
    
    import warnings
    warnings.warn = warn

import os
os.environ['NUMEXPR_MAX_THREADS'] = '80'

import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from diffusion import VanillaDiffusion
from helpers import squeezed_array, permute_data, metric_fn

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

X      = np.vstack([X_test, X_train])
n_test = X_test.shape[0]

# --
# Baselines

model     = LinearSVC().fit(X_train, y_train)
pred      = model.predict(X_test)
svc_score = metric_fn(y_test, pred)

model     = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
pred      = model.predict(X_test)
knn_score = metric_fn(y_test, pred)

# Why is this worse than exline version?

# --

n_trunc = np.inf
kd      = 8
sym_fn  = 'mean'

# <<
# diffusion_model = TDiffusion(features=X, kd=kd, metric='l2', sym_fn=sym_fn, alpha=0.9)
# d = diffusion_model.run(n_trunc=n_trunc, do_norm=False)
# --
diffusion_model = VanillaDiffusion(features=X, kd=kd, sym_fn=sym_fn)
d = diffusion_model.run()
# <<

scores  = d[:n_test, n_test:]
nscores = normalize(scores, 'l2', axis=1)

# --
# Diffusion w/ subsets of data

# Top-1
top1_idx   = squeezed_array(scores.argmax(axis=-1))
top1_score = metric_fn(y_test, y_train[top1_idx])

# Sum
labels    = np.unique(y_train)
tmp       = [scores[:,y_train == i].sum(axis=-1) for i in labels]
tmp       = np.column_stack([squeezed_array(t) for t in tmp])
sum_score = metric_fn(y_test, labels[tmp.argmax(axis=-1)])

# Norm sum
labels     = np.unique(y_train)
tmp        = [nscores[:,y_train == i].sum(axis=-1) for i in labels]
tmp        = np.column_stack([squeezed_array(t) for t in tmp])
nsum_score = metric_fn(y_test, labels[tmp.argmax(axis=-1)])

print({
    'svc_score'  : svc_score,
    'knn_score'  : knn_score,
    'top1_score' : top1_score,
    'sum_score'  : sum_score,
    'nsum_score' : nsum_score,
})

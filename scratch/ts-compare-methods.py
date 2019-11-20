#!/usr/bin/env python

"""
    scratch/ts-compare-methods.py
    
    Usage:
        echo > compare2.jl
        python scratch/ts-compare-methods.py --prob-name Adiac             >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name FiftyWords        >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name ArrowHead         >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name CinCECGTorso      >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name CricketY          >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name ECG200            >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name FaceFour          >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name Fish              >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name FordA             >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name HandOutlines      >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name Haptics           >> compare2.jl
        python scratch/ts-compare-methods.py --prob-name ItalyPowerDemand  >> compare2.jl
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

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from diffusion import VanillaDiffusion, TruncatedDiffusion
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

X      = np.vstack([X_test, X_train])
n_test = X_test.shape[0]

X = normalize(X, 'l2', axis=1)

metric_fn = metric_fns['f1'] if len(set(y_train)) == 2 else metric_fns['f1_macro']

# --

n_trunc = np.inf
kd      = 8
sym_fn  = 'mean'

tmodel = TruncatedDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9)
t      = tmodel.run(n_trunc=n_trunc)

vmodel = VanillaDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9)
v      = vmodel.run()

# --
# Diffusion w/ subsets of data

def compute_scores(d):
    scores  = d[:n_test, n_test:]
    nscores = normalize(scores, 'l2', axis=1)
    
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
    
    return {
        'top1_score' : top1_score,
        'sum_score'  : sum_score,
        'nsum_score' : nsum_score,
    }

print(json.dumps({
    "prob_name"          : args.prob_name,
    "TruncatedDiffusion" : compute_scores(t),
    "VanillaDiffusion"   : compute_scores(v),
}))

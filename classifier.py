#!/usr/bin/env python

"""
    classifier.py
"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '80'

import sys
import bcolz
import numpy as np

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from diffusion import TDiffusion

def squeezed_array(x):
    return np.asarray(x).squeeze()

# --
# IO

X_train = bcolz.open('../lwll/baselines/features/resnet18/cifar10/train')[:]
y_train = np.load('../lwll/baselines/features/resnet18/cifar10/train/y.npy')

X_test = bcolz.open('../lwll/baselines/features/resnet18/cifar10/test')[:]
y_test = np.load('../lwll/baselines/features/resnet18/cifar10/test/y.npy')

X_train = normalize(X_train, 'l2', axis=1)
X_test  = normalize(X_test, 'l2', axis=1)

X      = np.vstack([X_test, X_train])
n_test = X_test.shape[0]

# --
# Baseline

k     = 10000
model = LinearSVC().fit(X_train[:k], y_train[:k])
pred  = model.predict(X_test)
(y_test == pred).mean()

# 0.62

# --

n_trunc = 1000
kn      = 64

diffusion_model = TDiffusion(features=X)
d = diffusion_model.run(n_trunc=n_trunc, do_norm=False)
d.eliminate_zeros()

scores = d[:n_test, n_test:]
nscores = normalize(scores, 'l2', axis=1)

# --
# Diffusion w/ all data

# Top-1
top1_idx = squeezed_array(scores.argmax(axis=-1))
(y_test == y_train[top1_idx]).mean()

# Sum
tmp = [scores[:,y_train == i].sum(axis=-1) for i in np.unique(y_train)]
tmp = np.column_stack([squeezed_array(t) for t in tmp])
(y_test == tmp.argmax(axis=-1)).mean()

# Norm sum
tmp = [nscores[:,y_train == i].sum(axis=-1) for i in np.unique(y_train)]
tmp = np.column_stack([squeezed_array(t) for t in tmp])
(y_test == tmp.argmax(axis=-1)).mean()

# --
# Diffusion w/ subsets of data

# Top-1
top1_idx = squeezed_array(scores[:,:k].argmax(axis=-1))
(y_test == y_train[top1_idx]).mean()

# Sum
tmp = [scores[:,y_train[:k] == i].sum(axis=-1) for i in np.unique(y_train[:k])]
tmp = np.column_stack([squeezed_array(t) for t in tmp])
(y_test == tmp.argmax(axis=-1)).mean()

# Norm sum
tmp = [nscores[:,y_train[:k] == i].sum(axis=-1) for i in np.unique(y_train[:k])]
tmp = np.column_stack([squeezed_array(t) for t in tmp])
(y_test == tmp.argmax(axis=-1)).mean()

# --
# Results

# 1) On CIFAR w/ resnet18, diffusion classifier _does not_ outperform a LinearSVC
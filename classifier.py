#!/usr/bin/env python

"""
    classifier.py
    
    ?? What does the aff graph look like?  How many connected components?
    ?? How do diffusion parameters effect things?
    
    # 1) On CIFAR10/CIFAR100/eurosat w/ resnet18, 
    #        diffusion classifier _does not_ outperform a LinearSVC
    # 2) On CUB2011 w/ resnet18, diffusion classifier mildly outperforms LinearSVC on small datasets

"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '80'

import sys
import bcolz
import numpy as np

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from diffusion import TDiffusion

def squeezed_array(x):
    return np.asarray(x).squeeze()

def permute_data(X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]

# --
# IO

dataset = 'cifar100'

X_train = bcolz.open(f'../lwll/baselines/features/resnet18/{dataset}/train')[:]
y_train = np.load(f'../lwll/baselines/features/resnet18/{dataset}/train/y.npy')
X_train = normalize(X_train, 'l2', axis=1)
X_train, y_train = permute_data(X_train, y_train)

X_test = bcolz.open(f'../lwll/baselines/features/resnet18/{dataset}/test')[:]
y_test = np.load(f'../lwll/baselines/features/resnet18/{dataset}/test/y.npy')
X_test = normalize(X_test, 'l2', axis=1)
X_test, y_test = permute_data(X_test, y_test)

# <<

X      = np.vstack([X_test, X_train])
n_test = X_test.shape[0]

assert X.shape[1] == 512

# --
# Baseline

k     = 1000
model = LinearSVC().fit(X_train[:k], y_train[:k])
pred  = model.predict(X_test)
svc_acc = (y_test == pred).mean()
print('svc_acc', svc_acc)

# --

n_trunc = 1000
kd      = 32

diffusion_model = TDiffusion(features=X, kd=kd, alpha=0.8)
d = diffusion_model.run(n_trunc=n_trunc, do_norm=False)
d.eliminate_zeros()

scores  = d[:n_test, n_test:]
nscores = normalize(scores, 'l2', axis=1)

# --
# Diffusion w/ subsets of data

# Top-1
top1_idx = squeezed_array(scores[:,:k].argmax(axis=-1))
top1_acc = (y_test == y_train[top1_idx]).mean()

# Sum
labels = np.unique(y_train[:k])
tmp = [scores[:,y_train[:k] == i].sum(axis=-1) for i in labels]
tmp = np.column_stack([squeezed_array(t) for t in tmp])
sum_acc = (y_test == labels[tmp.argmax(axis=-1)]).mean()

# Norm sum
labels   = np.unique(y_train[:k])
tmp      = [nscores[:,y_train[:k] == i].sum(axis=-1) for i in labels]
tmp      = np.column_stack([squeezed_array(t) for t in tmp])
nsum_acc = (y_test == labels[tmp.argmax(axis=-1)]).mean()

print({
    'top1_acc' : top1_acc,
    'sum_acc'  : sum_acc,
    'svc_acc'  : svc_acc,
    'nsum_acc' : nsum_acc,
})

# --
# "active learning" experiment
# Sample a bunch of subsets of labeled data of the same size
# Is the accuracy of the classifier correlated w/ the average max proximity to 
# a labeled data point?  Yes, weakly.

def f(seed):
    rng = np.random.RandomState(seed=seed)
    
    sel      = rng.choice(X_train.shape[0], 1000, replace=False)
    top1_idx = squeezed_array(scores[:,sel].argmax(axis=-1))
    
    acc = (y_test == y_train[sel][top1_idx]).mean()
    z   = squeezed_array(scores[:,sel].max(axis=-1).data)
    return acc, np.mean(z), np.median(z)

import pandas as pd
from tqdm import trange
from joblib import Parallel, delayed
jobs = [delayed(f)(seed=i) for i in trange(5000)]
res  = Parallel(n_jobs=60, backend='multiprocessing', verbose=1)(jobs)

z = pd.DataFrame(res).sort_values(1).reset_index(drop=True)
_ = plt.plot(z[0].rolling(20).mean())
show_plot()

z

# --------------------------------------------------

# # --
# # Diffusion w/ all data

# # Top-1
# top1_idx = squeezed_array(scores.argmax(axis=-1))
# (y_test == y_train[top1_idx]).mean()

# # Sum
# tmp = [scores[:,y_train == i].sum(axis=-1) for i in np.unique(y_train)]
# tmp = np.column_stack([squeezed_array(t) for t in tmp])
# (y_test == tmp.argmax(axis=-1)).mean()

# # Norm sum
# tmp = [nscores[:,y_train == i].sum(axis=-1) for i in np.unique(y_train)]
# tmp = np.column_stack([squeezed_array(t) for t in tmp])
# (y_test == tmp.argmax(axis=-1)).mean()


# !! SVC w/ diffusion distance kernel?
# !! SVC on diffusion scores?

# a = normalize(d[n_test:n_test + k], 'l2', axis=1)
# b = normalize(d[:n_test], 'l2', axis=1)

# z = LinearSVC(C=2).fit(a, y_train[:k])
# (y_test == z.predict(b)).mean()

# --
# Results


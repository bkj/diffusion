#!/usr/bin/env python

"""
    scratch/active-learning-extension .py

    Usage:
        python scratch/active-learning-extension.py --data_dir DATA/DIRECTORY --prob-name TwoPatterns --ts true
        python scratch/active-learning-extension.py --data_dir DATA/DIRECTORY --prob-name cifar10 --ts false

"""


import sys
sys.path.append('.')

import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn_extra.cluster import KMedoids
from sklearn import svm
from scipy.special import entr

from scipy.spatial.distance import squareform, pdist

from diffusion import VanillaDiffusion
from helpers import  permute_data, metric_fns

def weighted_KMedioids(X, y, metric_fn, n_clusters=10, max_iter=1000, n_weights=1, entropy_iters=1, n_entropy=10, weighted_point=True):

    n = X.shape[0]
    kmed = KMedoids(n_clusters=n_clusters, metric='precomputed', max_iter=max_iter)
    train_idx = kmed.fit(X.max() - X).medoid_indices_

    for iter in range(entropy_iters):
        pred_max = np.argsort(scores[:, train_idx], axis=1)[:, ::-1][:, :n_weights]
        true_classes = y[train_idx]
        guess_entropy = np.zeros((n,))

        if weighted_point:
            y_hat = []
            pred_idx = true_classes[pred_max]
            for i in range(n):
                row_weights = X[i, train_idx[pred_max[i,:]]]
                row_class = pred_idx[i, :]
                row_unique = np.unique(row_class)
                guess = np.zeros((row_unique.shape[0],))
                for j in range(row_unique.shape[0]):
                    guess[j] = np.sum(row_weights[np.where(row_class == row_unique[j])[0]])

                guess_entropy[i] = entr(guess).sum()
                y_hat.append(row_unique[guess.argmax()])

        else:
            pred_idx = orig_scores[:, train_idx].argmax(axis=-1)
            y_hat = y[train_idx][pred_idx]
            for i in range(n):
                row_weights = X[i, train_idx[pred_max[i,:]]]
                guess_entropy[i] = entr(row_weights.sum())

        print('KMedioid function, accuracy after iter ', iter, ':', metric_fn(y, y_hat))
        highest_entropy = np.argsort(guess_entropy)[::-1][:n_entropy]
        train_idx = np.append(train_idx, highest_entropy)

    return y_hat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob_name', type=str, default='Adiac')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ts', type=str, default='true')
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

dataset  = args.prob_name
base_dir = args.data_dir
model    = 'resnet18'

if args.ts == 'true':
    df_train = pd.read_csv(f'{base_dir}/data/ucr/{args.prob_name}/{args.prob_name}_TRAIN.tsv', header=None, sep='\t')
    X_train, y_train = df_train.values[:, 1:], df_train.values[:, 0]

    df_test = pd.read_csv(f'{base_dir}/data/ucr/{args.prob_name}/{args.prob_name}_TEST.tsv', header=None, sep='\t')
    X_test, y_test = df_test.values[:, 1:], df_test.values[:, 0]

else:
    X_train = np.load(f'{base_dir}/features/{dataset}/0/train/x.npy')
    y_train = np.load(f'{base_dir}/features/{dataset}/0/train/y.npy')

    X_test = np.load(f'{base_dir}/features/{dataset}/0/test/x.npy')
    y_test = np.load(f'{base_dir}/features/{dataset}/0/test/y.npy')

X_train, y_train = permute_data(X_train, y_train)
X_train          = normalize(X_train, 'l2', axis=1)
X_test, y_test   = permute_data(X_test, y_test)
X_test           = normalize(X_test, 'l2', axis=1)
X                = np.vstack([X_test, X_train])
y                = np.hstack([y_test, y_train])

X = X[0:3000, :]
y = y[0:3000]

metric_fn = metric_fns['f1'] if len(set(y_train)) == 2 else metric_fns['f1_macro']
n_samples = X.shape[0]
n_trunc = np.inf
kd      = 8
sym_fn  = 'mean'

orig_scores = VanillaDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9).run()
scores      = orig_scores.copy()

cos_dists = squareform(pdist(X, metric='cosine'))
np.fill_diagonal(scores, np.inf)

# Testcases
props = .1
k = int(props*n_samples)
n_weights = 5
n_entropy = 10
entropy_iters = 20

print("WEIGHTED POINTS")
print(" ")
y_hat = weighted_KMedioids(X=orig_scores,
                           y=y,
                           metric_fn=metric_fn,
                           n_clusters=k,
                           max_iter=1000,
                           n_weights=n_weights,
                           entropy_iters=entropy_iters,
                           n_entropy=n_entropy,
                           weighted_point=True)

print("UNWEIGHTED POINTS")
print(" ")

y_hat = weighted_KMedioids(X=orig_scores,
                           y=y,
                           metric_fn=metric_fn,
                           n_clusters=k,
                           max_iter=1000,
                           n_weights=n_weights,
                           entropy_iters=entropy_iters,
                           n_entropy=n_entropy,
                           weighted_point=False)


meds = min(n_samples, k + (n_entropy * entropy_iters))

print("equivalent points:", meds)

kmed = KMedoids(n_clusters=meds, metric='precomputed', max_iter=1000)
train = kmed.fit(orig_scores.max() - orig_scores).medoid_indices_
pred_idx = orig_scores[:, train].argmax(axis=-1)
y_guess_kmed = y[train][pred_idx]
print('kmedoid, (diff)', metric_fn(y, y_guess_kmed))

kmed = KMedoids(n_clusters=meds, metric='precomputed', max_iter=1000)
train = kmed.fit(orig_scores.max() - orig_scores).medoid_indices_
pred_idx = orig_scores[:, train].argmax(axis=-1)
y_guess_kmed = y[train][pred_idx]
print('kmedoid, (diff) + n_entropy', metric_fn(y, y_guess_kmed))

#SVC
model = svm.LinearSVC().fit(X_train[:k, :], y_train[:k])
pred = model.predict(X_test)
print('SVC Linear', metric_fn(y_test, pred))

#RBF
model = svm.SVC(gamma='scale', kernel='rbf')
model.fit(X_train[:k, :], y_train[:k])
pred = model.predict(X_test)
print('SVC Kernel', metric_fn(y_test, pred))






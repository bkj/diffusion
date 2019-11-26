#!/usr/bin/env python

"""

"""

import sys
sys.path.append('.')

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn_extra.cluster import KMedoids

from diffusion import VanillaDiffusion
from helpers import permute_data, dtw_distance_matrix

# --
# Helpers

def load_problem(prob_name):
    df_train         = pd.read_csv(f'./data/ucr/{prob_name}/{prob_name}_TRAIN.tsv', header=None, sep='\t')
    X_train, y_train = df_train.values[:, 1:], df_train.values[:, 0]
    X_train          = normalize(X_train, 'l2', axis=1)
    X_train, y_train = permute_data(X_train, y_train)
    
    df_test        = pd.read_csv(f'./data/ucr/{prob_name}/{prob_name}_TEST.tsv', header=None, sep='\t')
    X_test, y_test = df_test.values[:, 1:], df_test.values[:, 0]
    X_test         = normalize(X_test, 'l2', axis=1)
    X_test, y_test = permute_data(X_test, y_test)
    
    return X_train, X_test, y_train, y_test


# --
# CLi

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name', type=str, default='ArrowHead')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)

# --
# IO

X_train, X_test, y_train, y_test = load_problem(args.prob_name)

X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])

n_samples = X.shape[0]
n_train   = X_train.shape[0]
n_test    = X_test.shape[0]

metric_fn = lambda act, pred: metrics.f1_score(act, pred, average='macro')

# --
# Precompute distances

# Cosine distance
cos_dist = squareform(pdist(X, metric='cosine'))

# DTW
dtw_dist = dtw_distance_matrix(X)

# Diffusion
# !! dist is not really a distance -- does that ever cause problems
dif_scores = VanillaDiffusion(features=X, kd=8, sym_fn='mean', alpha=0.9).run()
dif_dist   = dif_scores.max() - dif_scores

# --
# Samplers

def random_sample(dist, k):
    train_sel, _ = train_test_split(np.arange(dist.shape[0]), train_size=k)
    return train_sel


def kmedoid_sample(dist, k):
    kmed      = KMedoids(n_clusters=k, metric='precomputed', max_iter=1000)
    train_sel = kmed.fit(dist).medoid_indices_
    return train_sel


def kmedoid_iterated_unweighted_sample(dist, k, k_init=None, batch_size=1, topk=5):
    if k_init is None:
        k_init = k // 2
    
    train_sel = kmedoid_sample(dist, k=k_init)
    
    for it in range(k_init, k):
        close_labeled_points  = np.sort(dist[:, train_sel], axis=-1)[:,:topk]
        mean_distance         = close_labeled_points.mean(axis=-1)
        largest_mean_distance = np.argsort(mean_distance)[-batch_size:]
        
        train_sel = np.hstack([train_sel, largest_mean_distance])
    
    return train_sel



# --
# Classifiers

def one_nn(dist, y, train_sel):
    D_train = dist[train_sel][:,train_sel]
    D_all   = dist[:,train_sel]
    y_train = y[train_sel]
    
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(D_train, y_train)
    
    return metric_fn(y, model.predict(D_all))

# --
# Run Experiment

dists = {
    "cos" : cos_dist,
    "dtw" : dtw_dist,
    "dif" : dif_dist,
}

samplers = {
    "rand" : random_sample,
    "kmed" : kmedoid_sample,
    "iter" : kmedoid_iterated_unweighted_sample,
}

classifiers = {
    "1nn" : one_nn,
}

res = []
label_budgets = np.linspace(10, n_samples - 10, 10).astype(np.int32)
for k in tqdm(label_budgets):
    tmp = {
        "k" : k
    }
    
    for dist_name, dist in dists.items():
        for sampler_name, sampler in samplers.items():
            train_sel = sampler(dist, k=k)
            for classifier_name, classifier in classifiers.items():
                try:
                    tmp[f'{dist_name}_{sampler_name}_{classifier_name}'] = classifier(dist, y, train_sel)
                except:
                    pass
    
    res.append(tmp)

res = pd.DataFrame(res)
print(res)



"""
    metric_cosine_diffiusion.py

    comparison between distances in cosine and diffusion (fourier?) space.
    Question 1: What is the relation between these two sets of distances for various manifolds produced by CNN features?
    Question 2: What is the relation between these two sets of distances for manifolds produced by other graphs?

"""

import sys
sys.path.append('.')

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist


from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn_extra.cluster import KMedoids
from diffusion import VanillaDiffusion
from helpers import squeezed_array, permute_data, metric_fns

from scipy import stats

from diffusion import VanillaDiffusion, TruncatedDiffusion
from helpers import squeezed_array, permute_data, metric_fns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob_name', type=str, default='Adiac')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

dataset  = args.prob_name
base_dir = args.data_dir
model    = 'resnet18'

#X_train = np.load(f'{base_dir}/features/{dataset}/0/train/x.npy')
#y_train = np.load(f'{base_dir}/features/{dataset}/0/train/y.npy')

df_train         = pd.read_csv(f'{base_dir}/data/ucr/{args.prob_name}/{args.prob_name}_TRAIN.tsv', header=None, sep='\t')
X_train, y_train = df_train.values[:,1:], df_train.values[:,0]
X_train, y_train = permute_data(X_train, y_train)
X_train          = normalize(X_train, 'l2', axis=1)

#X_test = np.load(f'{base_dir}/features/{dataset}/0/test/x.npy')
#y_test = np.load(f'{base_dir}/features/{dataset}/0/test/y.npy').

df_test        = pd.read_csv(f'{base_dir}/data/ucr/{args.prob_name}/{args.prob_name}_TEST.tsv', header=None, sep='\t')
X_test, y_test = df_test.values[:,1:], df_test.values[:,0]
X_test, y_test = permute_data(X_test, y_test)
X_test         = normalize(X_test, 'l2', axis=1)

X       = np.vstack([X_test, X_train])
y       = np.hstack([y_test, y_train])
n_test  = X_test.shape[0]
n_train = X_train.shape[0]

X = normalize(X, 'l2', axis=1)

metric_fn = metric_fns['f1'] if len(set(y_train)) == 2 else metric_fns['f1_macro']

n_trunc = np.inf
kd      = 8
sym_fn  = 'mean'

#tmodel = TruncatedDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9)
#diff_t, vals, cos_distT = tmodel.run(n_trunc=n_trunc)

orig_scores = VanillaDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9).run()
scores      = orig_scores.copy()

cos_dists = squareform(pdist(X, metric='cosine'))
np.fill_diagonal(scores, np.inf)

def spearman_corr(diff, cos, ct):
    corr = []
    for i in range(diff.shape[0]):
        #indices of top ct cos distances
        idx_best = cos[i,:].argsort()[-ct:][::-1]
        u = cos[i,idx_best]
        v = diff[i,idx_best]
        corr.append(stats.spearmanr(u, v))
    corr = np.array(corr)

    return corr

med_acc = []
heur_acc = []
rand_acc = []
med_cos_acc = []
med_cosl_acc = []
props = [.01, .1, .2, .3, .6]
for prop in props:
    k = max(5, int(prop * n_train))
    # KMedoids
    kmed = KMedoids(n_clusters=k, metric='precomputed', max_iter=1000)
    train = kmed.fit(orig_scores.max() - orig_scores).medoid_indices_
    pred_idx = orig_scores[:, train].argmax(axis=-1)
    med_acc.append(metric_fn(y, y[train][pred_idx]))

    # Heuristic
    train = scores.mean(axis=0).argsort()[-k:]
    pred_idx = orig_scores[:, train].argmax(axis=-1)
    heur_acc.append(metric_fn(y, y[train][pred_idx]))

    # Random
    train = np.random.choice(X.shape[0], k, replace=False)
    pred_idx = orig_scores[:, train].argmax(axis=-1)
    rand_acc.append(metric_fn(y, y[train][pred_idx]))

    # Random (cosine cosine distance)
    kmed = KMedoids(n_clusters=k, metric='cosine', max_iter=1000)
    train = kmed.fit(X).medoid_indices_
    pred_idx = cos_dists[:, train].argmin(axis=-1)
    med_cos_acc.append(metric_fn(y, y[train][pred_idx]))

    # Random (cosine coslidean distance)
    train = np.random.choice(X.shape[0], k, replace=False)
    pred_idx = cos_dists[:, train].argmin(axis=-1)
    med_cosl_acc.append(metric_fn(y, y[train][pred_idx]))


# Top-5
wrtCOS = []
wrtDIFF = []
setlist = [5, 10, 20, 50, 100, 1000, 2000]
for i in setlist:
    corr_wrtCOS = spearman_corr(orig_scores, cos_dists, i)
    corr_wrtDIFF = spearman_corr(cos_dists, orig_scores, i)
    wrtCOS.append(np.average(corr_wrtCOS, axis=0)[1])
    wrtDIFF.append(np.average(corr_wrtDIFF, axis=0)[1])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
fig.suptitle('{}'.format(dataset), fontsize=16)
ax1.plot(props, med_acc)
ax1.plot(props, heur_acc)
ax1.plot(props, rand_acc)
ax1.plot(props, med_cos_acc)
ax1.plot(props, med_cosl_acc)
ax1.set(xlabel='k count as prop of train', ylabel='accuracy')
ax1.legend(['medioids', 'heuristic', 'random', 'mediod_cos', 'random_cos'], loc='upper left')

ax2.plot(np.log(setlist), wrtCOS)
ax2.plot(np.log(setlist), wrtDIFF)
ax2.set(xlabel='log(count of max)', ylabel='spearmanr p-value')
ax2.legend(['wrtCOS', 'wrtDIFF'], loc='upper right')

ax3.scatter(cos_dists.flatten(), np.log(orig_scores.flatten()), s= 1)
ax3.set(xlabel='cosine dist', ylabel='log(diffusion score)')

fig.savefig(f'/Users/ezekielbarnett/Documents/canfield/diffusion/results/{dataset}.png',dpi=fig.dpi)

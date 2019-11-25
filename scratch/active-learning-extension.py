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
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from scipy.special import entr
from scipy.spatial.distance import squareform, pdist

from sklearn_extra.cluster import KMedoids

from diffusion import VanillaDiffusion
from helpers import  permute_data, metric_fns




from fastdtw import fastdtw
from joblib import Parallel, delayed

def parmap(fn, x):
    jobs = [delayed(fn)(xx) for xx in x]
    return Parallel(backend='multiprocessing', n_jobs=16)(jobs)

def _fastdtw_metric(a, b):
    return fastdtw(a, b)[0]

def dtw_distance_matrix(X):
    #_dtw_dist_row_all = None
    global _dtw_dist_row_all
    def _dtw_dist_row_all(t):
        return [_fastdtw_metric(t, tt) for tt in X]

    return np.vstack(parmap(_dtw_dist_row_all, list(X)))

def weighted_KMedioids(X, y, metric_fn, n_clusters=10, max_iter=1000, n_weights=1, entropy_iters=1, n_entropy=10, weighted_point=True):

    n = X.shape[0]
    kmed = KMedoids(n_clusters=n_clusters, metric='precomputed', max_iter=max_iter)
    train_idx = kmed.fit(X.max() - X).medoid_indices_

    for iter in range(entropy_iters):
        pred_max = np.argsort(scores[:, train_idx], axis=1)[:, ::-1][:, :n_weights]
        true_classes = y[train_idx]
        guess_entropy = np.zeros((n,))

        #Method for both computing the labels based on weights, plus adding points based on entropy.
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

        #Method for adding points based only on entropy, not considering multiple signals from same label.
        else:
            pred_idx = orig_scores[:, train_idx].argmax(axis=-1)
            y_hat = y[train_idx][pred_idx]
            for i in range(n):
                row_weights = X[i, train_idx[pred_max[i,:]]]
                guess_entropy[i] = entr(row_weights.sum())

        print('KMedioid function, accuracy after iter ', iter, ':', metric_fn(y, y_hat))
        highest_entropy = np.argsort(guess_entropy)[::-1][:n_entropy]
        train_idx = np.append(train_idx, highest_entropy)
        acc = metric_fn(y, y_hat)

    return y_hat, acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob_name', type=str, default='Coffee')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ts', type=str, default='true')
    parser.add_argument('--results', type=str)
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


X_test = normalize(X_test, 'l2', axis=1)
X_train = normalize(X_train, 'l2', axis=1)
p = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[p], y_train[p]
X_test, y_test   = permute_data(X_test, y_test)
X                = np.vstack([X_train, X_test])
y                = np.hstack([y_train, y_test])
X                = X[0:5000, :]
y                = y[0:5000]

n_samples = X.shape[0]
n_test = X_test.shape[0]
n_train = X_train.shape[0]

metric_fn = metric_fns['f1'] if len(set(y_train)) == 2 else metric_fns['f1_macro']
n_trunc = np.inf
kd      = 8
sym_fn  = 'mean'

cos_dists = squareform(pdist(X, metric='cosine'))

orig_scores = VanillaDiffusion(features=X, kd=kd, sym_fn=sym_fn, alpha=0.9).run()
scores      = orig_scores.copy()
dif_dists_train = scores[0:n_train, 0:n_train]
dif_dists_test = scores[n_train::, 0:n_train]


dtw_dists = dtw_distance_matrix(X)
dtw_dists_train = dtw_dists[0:n_train, 0:n_train]
dtw_dists_test = dtw_dists[n_train::, 0:n_train]

cos_dists = squareform(pdist(X, metric='cosine'))

results = []
props =[.1, .2, .3, .4, .5, .6]
for prop in props:

    k = int(prop*n_samples)
    n_weights = 5
    n_entropy = 5
    entropy_iters = 20

    print("WEIGHTED POINTS")
    y_hat, acc_weighted = weighted_KMedioids(X=orig_scores,
                              y=y,
                              metric_fn=metric_fn,
                              n_clusters=k,
                              max_iter=1000,
                              n_weights=n_weights,
                              entropy_iters=entropy_iters,
                              n_entropy=n_entropy,
                              weighted_point=True)


    print("UNWEIGHTED POINTS")
    y_hat, acc_unweighted = weighted_KMedioids(X=orig_scores,
                              y=y,
                              metric_fn=metric_fn,
                              n_clusters=k,
                              max_iter=1000,
                              n_weights=n_weights,
                              entropy_iters=entropy_iters,
                              n_entropy=n_entropy,
                              weighted_point=False)


    meds = min(n_samples, k + (n_entropy * entropy_iters))

    kmed = KMedoids(n_clusters=k, metric='precomputed', max_iter=5000)
    train = kmed.fit(orig_scores.max() - orig_scores).medoid_indices_
    pred_idx = orig_scores[:, train].argmax(axis=-1)
    y_guess_kmed = y[train][pred_idx]
    acc_kmedioid_k = metric_fn(y, y_guess_kmed)
    print('kmedoid, (diff)', acc_kmedioid_k)


    kmed = KMedoids(n_clusters=meds, metric='precomputed', max_iter=1000)
    train = kmed.fit(orig_scores.max() - orig_scores).medoid_indices_
    pred_idx = orig_scores[:, train].argmax(axis=-1)
    y_guess_kmed = y[train][pred_idx]
    acc_kmedioid_kplus = metric_fn(y, y_guess_kmed)
    print('kmedoid, (diff)', acc_kmedioid_kplus)

    #KNN
    parameters = {'n_neighbors': range(2, int(k/3), 3)}

    model = GridSearchCV(KNeighborsClassifier(), parameters, cv=3, verbose=1)
    model.fit(X[train,:], y[train])
    pred = model.predict(X_test)
    acc_knn = metric_fn(y_test, pred)
    print('KNN vanilla', acc_knn)

    model = GridSearchCV(KNeighborsClassifier(metric='precomputed'), parameters, cv=3, verbose=1)
    model.fit(dtw_dists_train[0:k, 0:k], y_train[0:k])
    pred2 = model.predict(dtw_dists_test[:,0:k])
    acc_knn_dtw = metric_fn(y_test, pred2)
    print("KNN dtw", acc_knn_dtw)

    model = GridSearchCV(KNeighborsClassifier(metric='precomputed'), parameters, cv=3, verbose=1)
    model.fit(dif_dists_train[0:k, 0:k], y_train[0:k])
    pred2 = model.predict(dif_dists_test[:,0:k])
    acc_knn_diff = metric_fn(y_test, pred2)
    print("KNN diffusion", acc_knn_diff)


    #SVC (indexing only kmedioids train points may be incorrect?)
    model = svm.LinearSVC().fit(X[train,:], y[train])
    pred = model.predict(X_test)
    acc_svc = metric_fn(y_test, pred)
    print('SVC Linear', acc_svc)

    model2 = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel='rbf').fit(X[train,:], y[train])
    pred2 = model2.predict(X_test)
    acc_svc_rbf = metric_fn(y_test, pred2)
    print('SVC RBF Kernel', acc_svc_rbf)

    model3 = svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='precomputed').fit(dtw_dists_train[0:k,0:k], y_train[0:k])
    pred3 = model3.predict(dtw_dists_test[:,0:k])
    acc_svc_dtw = metric_fn(y_test, pred3)
    print('SVC DTW Kernel', acc_svc_dtw)

    model4 = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel='precomputed').fit(dif_dists_train[0:k,0:k], y_train[0:k])
    pred4 = model4.predict(dif_dists_test[:,0:k])
    acc_svc_diff = metric_fn(y_test, pred4)
    print('SVC DIFF Kernel', acc_svc_diff)








    results.append([prop, acc_weighted, acc_unweighted, acc_kmedioid_k, acc_kmedioid_kplus, acc_knn, acc_knn_diff, acc_knn_dtw, acc_svc, acc_svc_rbf, acc_svc_diff, acc_svc_dtw])

df = pd.DataFrame(results, columns=['prop', 'acc_weighted', 'acc_unweighted', 'acc_kmedioid_k',  'acc_kmediod_kplus',
                                    'acc_knn', 'acc_knn_diff', 'acc_knn_dtw', 'acc_svc', 'acc_svc_rbf', 'acc_svc_diff',
                                    'acc_svc_dtw'])

df.plot(kind='line', x='prop', ylim=[0,1.1], title='accuracy {}'.format(args.prob_name), legend=True)
plt.show()
#plt.savefig(f'/{args.results}/{args.prob_name}.png')


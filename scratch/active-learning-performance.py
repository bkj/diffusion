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
from scipy.spatial.distance import pdist, squareform, cdist

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn_extra.cluster import KMedoids

from diffusion import VanillaDiffusion
from helpers import permute_data, dtw_distance_matrix


# --
# Helpers

def load_problem(prob_name):
    df_train = pd.read_csv(f'./data/ucr/{prob_name}/{prob_name}_TRAIN.tsv', header=None, sep='\t')
    X_train, y_train = df_train.values[:, 1:], df_train.values[:, 0]
    X_train = normalize(X_train, 'l2', axis=1)
    X_train, y_train = permute_data(X_train, y_train)

    df_test = pd.read_csv(f'./data/ucr/{prob_name}/{prob_name}_TEST.tsv', header=None, sep='\t')
    X_test, y_test = df_test.values[:, 1:], df_test.values[:, 0]
    X_test = normalize(X_test, 'l2', axis=1)
    X_test, y_test = permute_data(X_test, y_test)

    return X_train, X_test, y_train, y_test

# --
# Samplers

def random_sample(dist, k):
    train_sel, _ = train_test_split(np.arange(dist.shape[0]), train_size=k)
    return train_sel


def kmedoid_sample(dist, k):
    kmed = KMedoids(n_clusters=k, metric='precomputed', max_iter=1000)
    train_sel = kmed.fit(dist).medoid_indices_
    return train_sel


def kmedoid_iterated_unweighted_sample(dist, k, k_init=None, batch_size=1, topk=5):
    if k_init is None:
        k_init = k // 2

    train_sel = kmedoid_sample(dist, k=k_init)

    for it in range(k_init, k):
        close_labeled_points = np.sort(dist[:, train_sel], axis=-1)[:, :topk]
        mean_distance = close_labeled_points.mean(axis=-1)
        largest_mean_distance = np.argsort(mean_distance)[-batch_size:]
        train_sel = np.hstack([train_sel, largest_mean_distance])

    return train_sel

# --
# Classifiers

def one_nn(dist, y, train_sel, test_sel):
    D_train = dist[train_sel][:, train_sel]
    D_test = dist[test_sel][:, train_sel]
    y_train = y[train_sel]
    y_test = y[test_sel]

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(D_train, y_train)
    return metric_fn(y_test, model.predict(D_test))

def svc(dist, y, train_sel, test_sel):
    D_train = dist[train_sel][:, train_sel]
    D_test = dist[test_sel][:, train_sel]
    y_train = y[train_sel]
    y_test = y[test_sel]

    model = svm.LinearSVC()
    model.fit(D_train, y_train)

    return metric_fn(y_test, model.predict(D_test))


# == bake off dataset names ====================================================

dataset_names = \
(
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "GunPoint",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga"
)


res = []
for dataset in dataset_names:

    X_train, X_test, y_train, y_test = load_problem(dataset)

    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])

    n_samples = X.shape[0]
    n_train = X_train.shape[0]
    idx_test = np.arange(X_train.shape[0], n_samples, 1)

    metric_fn = lambda act, pred: metrics.accuracy_score(act, pred)

    # --
    # Precompute distances
    # Euclidian distance
    euc_dist = squareform(pdist(X, metric='euclidean'))

    # Cosine distance
    cos_dist = squareform(pdist(X, metric='cosine'))

    # DTW
    dtw_dist = dtw_distance_matrix(X)

    # Diffusion
    # !! dist is not really a distance -- does that ever cause problems
    dif_scores = VanillaDiffusion(features=X, kd=8, sym_fn='mean', alpha=0.9).run()
    dif_dist = dif_scores.max() - dif_scores

    # --
    # Run Experiment
    dists = {
        "cos": cos_dist,
        "dtw": dtw_dist,
        "dif": dif_dist,
        "euc": euc_dist,
    }

    samplers = {
        "rand": random_sample,
        "kmed": kmedoid_sample,
        "iter": kmedoid_iterated_unweighted_sample,
    }

    classifiers = {
        "1nn": one_nn,
        "svc": svc,
    }

    tmp = {'dataset': dataset}
    for dist_name, dist in dists.items():
        for sampler_name, sampler in samplers.items():
            train_sel = sampler(dist, k=n_train)
            for classifier_name, classifier in classifiers.items():
                try:
                    tmp[f'{dist_name}_{sampler_name}_{classifier_name}'] = classifier(dist, y, train_sel, idx_test)
                except:
                    pass
    print(dataset, "is done")
    res.append(tmp)


res = pd.DataFrame(res)
res.to_csv(f'./data/results_ucr.csv', index=False)




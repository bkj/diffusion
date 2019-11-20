#!/usr/bin/env python

"""
    helpers.py
"""


import numpy as np
from sklearn import metrics

def squeezed_array(x):
    return np.asarray(x).squeeze()

def permute_data(X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]

def metric_fn(act, pred):
    return metrics.f1_score(act, pred, average='macro')
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

metric_fns = {
    "f1_macro" : lambda act, pred: metrics.f1_score(act, pred, average='macro'),
    "f1"       : lambda act, pred: metrics.f1_score(act, pred, average='binary'),
}
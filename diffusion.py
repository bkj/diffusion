#!/usr/bin/env python

"""
    diffusion.py
"""

import faiss
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

# --
# Helpers

_fn = None
def parmap(f, x, n_jobs):
    jobs = [delayed(f)(xx) for xx in x]
    return Parallel(n_jobs=n_jobs, backend='multiprocessing')(jobs)


class Diffusion(object):
    def __init__(self, features, alpha=0.99, gamma=3, kd=50, n_jobs=60):
        self.features    = features
        self.n_obs       = self.features.shape[0]
        self.feature_dim = self.features.shape[1]
        
        self.gamma  = gamma
        self.alpha  = alpha
        self.kd     = kd
        self.n_jobs = n_jobs
        
        self.knn = faiss.IndexFlatIP(self.feature_dim)
        self.knn.add(features)
    
    def run(self, n_trunc, do_norm):
        global _fn
        
        ball_sims, ball_ids = self.knn.search(self.features, n_trunc)
        
        # Make kd-nearest-neighbors symmetric affinity graph
        neib_sim = ball_sims[:, :self.kd] ** self.gamma
        neib_ids = ball_ids[:, :self.kd]
        aff      = self.get_sym_aff(s=neib_sim, i=neib_ids)
        
        # Make laplacian
        lap = self.get_laplacian(aff=aff)
        
        # Initial signal (query is always most similar to self, so always at index 0)
        signal    = np.zeros(n_trunc)
        signal[0] = 1
        
        # Compute diffusion
        def _fn(i):
            trunc_lap = lap[ball_ids[i]][:, ball_ids[i]]
            scores, _ = linalg.cg(trunc_lap, signal, tol=1e-8, maxiter=1000)
            return scores
        
        vals = parmap(_fn, range(self.n_obs), n_jobs=self.n_jobs)
        
        # Return sparse matrix representation of diffusion
        vals = np.concatenate(vals)
        rows = np.repeat(np.arange(self.n_obs), n_trunc)
        cols = ball_ids.ravel()
        
        out = sparse.csr_matrix((vals, (rows, cols)))
        
        if do_norm:
            out = normalize(out, norm="l2", axis=1)
        
        return out
        
    def get_laplacian(self, aff):
        n  = aff.shape[0]
        
        D = aff @ np.ones(n) + 1e-12
        D = D ** (-0.5)
        D = sparse.diags(D)
        
        return sparse.eye(n) - self.alpha * (D @ aff @ D)
    
    def get_sym_aff(self, s, i):
        row = np.repeat(np.arange(s.shape[0]), s.shape[1])
        col = np.ravel(i)
        val = np.ravel(s)
        aff = sparse.csc_matrix((val, (row, col)))
        
        aff.setdiag(0)           # Remove self-similarity
        aff = aff.minimum(aff.T) # Make symmetric
        aff.eliminate_zeros()
        aff.sort_indices()
        
        return aff

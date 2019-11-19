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
from scipy.sparse.linalg import eigsh

# --
# Helpers

_fn = None
def parmap(f, x, n_jobs):
    jobs = [delayed(f)(xx) for xx in x]
    return Parallel(n_jobs=n_jobs, backend='multiprocessing')(jobs)

sym_fns = {
    "minimum" : lambda x: x.minimum(x.T).tocsr(),
    "maximum" : lambda x: x.maximum(x.T).tocsr(),
    "mean"    : lambda x: ((x + x.T) / 2).tocsr(),
}


class TDiffusion(object):
    def __init__(self, features, alpha=0.99, gamma=3, kd=50, n_jobs=60, metric='cosine', sym_fn='minimum'):
        
        self.features    = np.ascontiguousarray(features.astype(np.float32))
        self.n_obs       = self.features.shape[0]
        self.feature_dim = self.features.shape[1]
        
        self.gamma  = gamma
        self.alpha  = alpha
        self.kd     = kd
        self.n_jobs = n_jobs
        self.metric = metric
        
        if metric == 'cosine':
            assert (np.abs((features ** 2).sum(axis=-1) - 1) < 1e-5).all(), 'Normalize features?'
            self.knn = faiss.IndexFlatIP(self.feature_dim)
        elif metric == 'l2':
            self.knn = faiss.IndexFlatL2(self.feature_dim)
        else:
            raise Exception()
        
        self.knn.add(self.features)
        
        self.aff = None
        
        assert sym_fn in sym_fns
        self.sym_fn = sym_fns[sym_fn]
    
    def run(self, n_trunc, do_norm=True):
        global _fn
        
        n_trunc = min(n_trunc, self.n_obs)
        
        ball_sims, ball_ids = self.knn.search(self.features, n_trunc)
        if self.metric == 'l2':
            ball_sims = np.sqrt(ball_sims)
            ball_sims = 1 - ball_sims / ball_sims.max()
        
        # Compute kd-nearest-neighbors symmetric affinity graph
        neib_sim = ball_sims[:, :self.kd + 1] ** self.gamma
        neib_ids = ball_ids[:, :self.kd + 1]
        aff      = self._get_sym_aff(s=neib_sim, i=neib_ids)
        
        # Compute laplacian
        lap = self._get_laplacian(aff=aff)
        
        # Compute diffusion
        def _fn(i):
            # Initial signal (usually ball_ids[i] == i occurs at index zero, but sometimes not)
            signal = np.zeros(n_trunc)
            signal[ball_ids[i] == i] = 1
            
            trunc_lap = lap[ball_ids[i]][:, ball_ids[i]]
            scores, _ = linalg.cg(trunc_lap, signal, tol=1e-8, maxiter=5000)
            return scores
        
        vals = parmap(_fn, range(self.n_obs), n_jobs=self.n_jobs)
        
        # Return sparse matrix representation of diffusion
        vals = np.concatenate(vals)
        rows = np.repeat(np.arange(self.n_obs), n_trunc)
        cols = ball_ids.ravel()
        
        out = sparse.csr_matrix((vals, (rows, cols)))
        
        if do_norm:
            out = normalize(out, norm="l2", axis=1)
        
        self.aff = aff
        out.eliminate_zeros()
        return out
    
    def _get_sym_aff(self, s, i):
        row = np.repeat(np.arange(s.shape[0]), s.shape[1])
        col = np.ravel(i)
        val = np.ravel(s)
        aff = sparse.csc_matrix((val, (row, col)))
        
        assert aff.min() >= 0
        
        aff.setdiag(0) # Remove self-similarity
        
        aff = self.sym_fn(aff)
        aff.eliminate_zeros()
        aff.sort_indices()
        
        return aff
        
    def _get_laplacian(self, aff):
        n  = aff.shape[0]
        
        D = aff @ np.ones(n) + 1e-6
        D = D ** (-0.5)
        D = sparse.diags(D)
        
        return sparse.eye(n) - self.alpha * (D @ aff @ D)

# --

from scipy.spatial.distance import pdist, squareform

class PlainDiffusion(object):
    def __init__(self, features, alpha=0.9, kd=16, sym_fn='mean'):
        
        self.features    = features
        self.n_obs       = self.features.shape[0]
        self.feature_dim = self.features.shape[1]
        
        # self.gamma  = gamma # !! How does this fit in?
        self.alpha = alpha
        self.kd    = kd
        
        self.aff = None
        
        assert sym_fn in sym_fns
        self.sym_fn = sym_fn
    
    def run(self):
        global _fn
        
        n_nodes = self.features.shape[0]
        
        dist = squareform(pdist(self.features, metric='euclidean'))
        aff  = 1 - dist / dist.max()
        
        assert aff.min() >= 0
        
        # No self-loops
        np.fill_diagonal(aff, -np.inf)
        
        # Binary adjacency matrix
        threshes = np.sort(aff, axis=0)[-self.kd].reshape(1, -1)
        aff[aff < threshes]  = 0
        aff[aff >= threshes] = 1
        
        # Symmetrize
        if self.sym_fn == 'max':
            adj = np.maximum(aff, aff.T)
        elif self.sym_fn == 'min':
            adj = np.minimum(aff, aff.T)
        elif self.sym_fn == 'mean':
            adj = (aff + aff.T) / 2
        else:
            raise Exception
        
        # Normalize adj
        D = adj @ np.ones(n_nodes) + 1e-12
        D = D ** (-0.5)
        D = np.diag(D)
        DAD = D @ adj @ D
        DAD = (DAD + DAD.T) / 2 # Force symmetry
        
        eigval, eigvec = eigsh(DAD, k=n_nodes)
        eigval = eigval.astype(np.float64)
        
        h_eigval = 1 / (1 - self.alpha * eigval)
        
        out = eigvec @ np.diag(h_eigval) @ eigvec.T
        return out

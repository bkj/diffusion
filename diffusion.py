#!/usr/bin/env python

"""
    diffusion.py
"""

import faiss
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from joblib import Parallel, delayed

from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform

# --
# Helpers

_fn = None
def parmap(f, x, n_jobs):
    jobs = [delayed(f)(xx) for xx in x]
    return Parallel(n_jobs=n_jobs, backend='multiprocessing')(jobs)

sparse_sym_fns = {
    "minimum" : lambda x: x.minimum(x.T).tocsr(),
    "maximum" : lambda x: x.maximum(x.T).tocsr(),
    "mean"    : lambda x: ((x + x.T) / 2).tocsr(),
}

dense_sym_fns = {
    "minimum" : lambda x: np.minimum(x, x.T),
    "maximum" : lambda x: np.maximum(x, x.T),
    "mean"    : lambda x: (x + x.T) / 2
}


class TruncatedDiffusion(object):
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
        else:
            raise Exception('!! TruncatedDiffusion: Unrecognized metric')
        
        self.knn.add(self.features)
        
        assert sym_fn in sparse_sym_fns
        self.sym_fn = sparse_sym_fns[sym_fn]
    
    def run(self, n_trunc):
        global _fn
        
        n_trunc = min(n_trunc, self.n_obs)
        
        ball_sims, ball_ids = self.knn.search(self.features, n_trunc)
        
        # Compute kd-nearest-neighbors symmetric affinity graph
        neib_sim = ball_sims[:, :self.kd + 1] ** self.gamma
        neib_ids = ball_ids[:, :self.kd + 1]
        adj      = self._aff2adj(s=neib_sim, i=neib_ids)
        
        # Compute laplacian
        lap = self._adj2lap(adj=adj)
        
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
        
        out.eliminate_zeros()
        return out
    
    def _aff2adj(self, s, i):
        row = np.repeat(np.arange(s.shape[0]), s.shape[1])
        col = np.ravel(i)
        val = np.ravel(s)
        adj = sparse.csc_matrix((val, (row, col)))
        
        assert adj.min() >= 0
        
        adj.setdiag(0)
        
        adj = self.sym_fn(adj)
        adj.eliminate_zeros()
        adj.sort_indices()
        
        return adj
        
    def _adj2lap(self, adj):
        n = adj.shape[0]
        
        D = adj @ np.ones(n) + 1e-12
        D = D ** (-0.5)
        D = sparse.diags(D)
        
        DAD = (D @ adj @ D)
        
        return sparse.eye(n) - self.alpha * DAD

# --

class VanillaDiffusion(object):
    def __init__(self, features, alpha=0.9, kd=16, metric='euclidean', sym_fn='mean'):
        
        self.features    = features
        self.n_obs       = self.features.shape[0]
        self.feature_dim = self.features.shape[1]
        
        # self.gamma  = gamma # !! How does this fit in?
        self.alpha  = alpha
        self.kd     = kd
        self.metric = metric
        
        assert sym_fn in dense_sym_fns
        self.sym_fn = dense_sym_fns[sym_fn]
    
    def run(self):
        global _fn
        
        n_nodes = self.features.shape[0]
        
        dist = squareform(pdist(self.features, metric=self.metric))
        adj  = self._dist2adj(dist)
        
        # Normalize adj
        D   = adj @ np.ones(n_nodes) + 1e-12
        D   = D ** (-0.5)
        D   = np.diag(D)
        
        DAD = D @ adj @ D
        DAD = (DAD + DAD.T) / 2 # Force exact symmetry
        
        eigval, eigvec = eigsh(DAD, k=n_nodes)
        eigval = eigval.astype(np.float64)
        
        h_eigval = 1 / (1 - self.alpha * eigval)
        
        out = eigvec @ np.diag(h_eigval) @ eigvec.T
        return out
    
    def _dist2adj(self, dist):
        adj = 1 - dist / dist.max()
        assert adj.min() >= 0
        
        # No self-loops
        np.fill_diagonal(adj, -np.inf)
        
        # Binarize adjacency matrix
        threshes = np.sort(adj, axis=0)[-self.kd].reshape(1, -1)
        adj[adj < threshes]  = 0
        adj[adj >= threshes] = 1
        
        adj = self.sym_fn(adj)
        
        return adj
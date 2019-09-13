#!/usr/bin/env python

"""
    diffusion.py
"""


import numpy as np
from knn import KNN
from tqdm import tqdm
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from joblib import Parallel, delayed
from scipy.stats import rankdata

trunc_ids  = None
trunc_init = None
lap_alpha  = None
SIM        = None
aff        = None

def ista(s, adj, alpha=0.15, rho=1.0e-5, epsilon=1e-2):
    
    # Compute degree vectors/matrices
    d       = np.asarray(adj.sum(axis=-1)).squeeze() + 1e-10
    d_sqrt  = np.sqrt(d)
    dn_sqrt = 1 / d_sqrt
    
    D       = sparse.diags(d)
    Dn_sqrt = sparse.diags(dn_sqrt)
    
    # Normalized adjacency matrix
    Q = D - ((1 - alpha) / 2) * (D + adj)
    Q = Dn_sqrt @ Q @ Dn_sqrt
    
    # Initialize
    rad  = rho * alpha * d_sqrt
    q    = np.zeros(adj.shape[0], dtype=np.float64)
    
    grad0 = -alpha * dn_sqrt * s
    grad  = grad0
    
    # Run
    thresh = rho * alpha * (1 + epsilon)
    it = 0
    # while np.abs(grad * dn_sqrt).max() > thresh:
    for it in range(100):
        q    = np.maximum(q - grad - rad, 0)
        grad = grad0 + Q @ q
        it += 1
    
    return q * d_sqrt


# from scipy import sparse
# s_mat = sparse.eye(aff.shape[0]).tocsr()
# features = ista_mat(s_mat, aff, do_numba=True, alpha=0.2)

# features.nnz / np.prod(features.shape)

# features  = preprocessing.normalize(features, norm="l2", axis=1)
# scores    = features[:n_query] @ features[n_query:].T
# ranks     = np.argsort(-scores.todense())
# compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)


def get_offline_result(i):
    ids       = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-8, maxiter=1000)
    return scores


def get_offline_result2(i):
    ids  = trunc_ids[i]
    taff = aff[ids][:, ids]
    
    return ista(trunc_init, taff, alpha=0.2)


class Diffusion(object):
    def __init__(self, features, alpha=0.99, gamma=3, kd=50):
        self.features  = features
        self.N         = len(self.features)
        
        self.gamma     = gamma
        self.alpha     = alpha
        self.kd        = kd
        
        self.knn = KNN(self.features, method='cosine')
    
    def get_offline_results(self, n_trunc):
        
        global trunc_ids, trunc_init, lap_alpha, SIM, aff
        
        sims, ids = self.knn.search(self.features, n_trunc)
        trunc_ids = ids
        aff       = self.get_affinity(s=sims[:, :self.kd], i=ids[:, :self.kd])
        lap_alpha = self.get_laplacian(aff=aff)
        
        # <<
        # vals = Parallel(n_jobs=60, backend='multiprocessing', verbose=True)(
        #     delayed(get_offline_result2)(i) for i in range(self.N))
        
        # return np.vstack(vals)
        # --
        trunc_init    = np.zeros(n_trunc)
        trunc_init[0] = 1
        
        vals = Parallel(n_jobs=60, backend='multiprocessing')(
            delayed(get_offline_result2)(i) for i in range(self.N))
        
        vals = np.concatenate(vals)
        
        rows = np.repeat(np.arange(self.N), n_trunc)
        cols = trunc_ids.ravel()
        
        return sparse.csr_matrix((vals, (rows, cols)))
        # >>
        
    def get_laplacian(self, aff):
        n  = aff.shape[0]
        
        D  = aff @ np.ones(n) + 1e-12
        D  = D ** (-0.5)
        D  = sparse.diags(D)
        
        return sparse.eye(n) - self.alpha * (D @ aff @ D)
    
    def get_affinity(self, s, i):
        
        s = s ** self.gamma
        
        row = np.repeat(np.arange(s.shape[0]), s.shape[1])
        col = np.ravel(i)
        val = np.ravel(s)
        
        aff = sparse.csc_matrix((val, (row, col)))
        aff.setdiag(0)
        aff = aff.minimum(aff.T)
        aff.eliminate_zeros()
        aff.sort_indices()
        
        return aff

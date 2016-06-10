# -*- coding: utf-8 -*-
"""
===============================================================================
gt for "graph theory"
===============================================================================
Created on Tue May 10 13:10:06 2016

@author: takanori
"""

import numpy as np
import networkx as nx
import bct
from scipy.spatial.distance import squareform as sqform

def gt_mst(A,max_span=False):
    """ Get max/min spanning tree from weighted matrix.

    Created 05/10/2016
    """
    # convert to nx graph
    A = nx.Graph(A)
    if max_span:
        mst = nx.maximum_spanning_tree(A)
    else:
        mst = nx.minimum_spanning_tree(A)

    mst = nx.to_numpy_matrix(mst)
    return np.asarray(mst)

def gt_mst_design_mat(X,max_span=False):
    """ Return max/min spanning tree given design matrix

    Parameters
    ----------
    X : ndarray of shape=(n_sample,n_features)
        Data/design matrix
    max_span : bool
        return max span tree if True (default=False, returns min-span-tree)

    Returns
    -------
    X_mst : ndarray of shape=(n_sample,n_features)

    Created 05/10/2016
    """
    n,p = X.shape

    X_mst = np.zeros((n,p))
    for i in range(n):
        A = X[i]
        # convert vector to symmetric matrix
        A = sqform(A)

        # compute mst
        mst_mat = gt_mst(A,max_span=max_span)

        # vectorize
        X_mst[i] = sqform(mst_mat)

    return X_mst
#%% === bct stuffs ===
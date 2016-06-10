# -*- coding: utf-8 -*-
"""
===============================================================================
Code forked from ~/python/analysis/nmf/nmf_module.py (05/09/2016)

Here, I can import this module from anywhere
===============================================================================
Created on 9, May 2016

Note: big update on 02/11/2016 - replaced all np.argsort with tw.argsort so
that "ties"-indices are ordered by their occurence.  This helps me do
consistency check with my various knn-type methods.

Note: big update on 02/13/2016 - all NMF methods now default initializes via
'nndsvd', and rho=1e3, tol=5e-5.

Update 02/14/2016 - all NMF methods max_iter now set to 5000
Update 02/14/2016 - changed tol=1e-4 for all nmf methods

@author: takanori
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from numpy import dot, trace

from sklearn.utils.validation import check_is_fitted#, check_random_state
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin

import time

import tak as tw
from scipy.linalg import norm,solve,svd

from tak.core import print_time
#%%==== Main functions ====
def pnmf(X,r=5,max_iter=5000, tol=1e-4, l2_pen=None, W=None,return_cost=False):
    """ Projective NMF

    Projective NMF model of 2010 N. Yang

    Model: :math:`X \\approx W(W^T X)`

    .. math::

        \\min_{W\\geq 0} \\frac{1}{2}\\|X-WH\|^2_F \\text{  such that  } H=W^T X

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space (n_components)
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance
    l2_pen : float (Default = None)
        L2 regularization on basis W
    W : ndarray of shape (n_features, n_components)=(p,r)
        Initial basis-matrix estimate
    return_cost : bool (Default = False)
        Return per-iteration cost and diffW (increases computation)

    Returns
    -------
    W : array of shape (n_features,n_components) = (p,r)
        Non-negative Basis matrix

    Usage
    -----
    >>> # usually X = (n,p) matrix... but pnmf assumes X=(p,n), so apply transpose
    >>> W = pnmf(X.T,r=10)
    >>> W, cost_, diffW_ = pnmf(X.T,r=10,return_cost=True)

    History
    -------
    Update 02/14/2016 - modified "cost" from frobenius to **squared**-
    frobenius norm (for consistency with my other admm scripts)
    """
    p = X.shape[0]

    if W is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

    if return_cost:
        diffW_list=[]
        cost_list=[]

    start = time.time()
    for iter in range(max_iter):
        W_old = W

        XtW = X.T.dot(W)
        WtW = W.T.dot(W)

        # numerator term
        num = X.dot(XtW)
        den = W.dot(XtW.T.dot(XtW)) + X.dot(XtW.dot(WtW)) + 1e-10
#        den = W.dot(XtW.T.dot(XtW)) + 1e-10
        if l2_pen is not None:
            # extra term arising from Frobenius norm regularizer
            den = den + l2_pen * W

        W = W* (num / den)
        W /= norm(W,2) # divide by largest singular value

        diffW = norm(W_old - W, 'fro') / norm(W_old, 'fro')

        if (diffW < tol) and (iter > 100): # allow at least 100 iter for "burn-in"
            break

        if return_cost:
            """keep track of relevant per-iteration values"""
            diffW_list.append(diffW)
            if l2_pen is None:
                cost_list.append(norm(X - W.dot(W.T.dot(X)),'fro')**2)
            else:
                cost_list.append(norm(X - W.dot(W.T.dot(X)),'fro')**2 +
                                 0.5*l2_pen*norm(W,'fro')**2)

        if iter % 500 == 0:
            obj_val = norm(X - W.dot(W.T.dot(X)),'fro')**2
            if l2_pen is not None:
                # add frobenius norm L2 penalty to objective
                obj_val = obj_val + 0.5*l2_pen*norm(W,'fro')**2
            print "iter = {:4}, diff = {:3.2e}, cost = {:6.5e} ({:3.1f} sec)".format(
                iter,diffW, obj_val,time.time()-start)
    #print iter, diffW
    if return_cost:
        return W, np.asarray(cost_list), np.asarray(diffW_list)
    else:
        return W


def gpnmf(X,r, A, lam, max_iter=5000, tol=1e-4, l2_pen=None, W=None,return_cost=False):
    """ Graph-regularized Projective NMF

    Projective NMF model of 2010 N. Yang

    Model: :math:`X \\approx W(W^T X)`

    .. math::

        \\min_{W\\geq 0} \\frac{1}{2}\\|X-WH\|^2_F \\text{  such that  } H=W^T X

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space (n_components)
    A : ndarray of shape [n_samples,n_samples]
        Similarity/adjacency matrix
    lam : float
        regularization parameter on graph penalty
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance
    l2_pen : float (Default = None)
        L2 regularization on basis W
    W : ndarray of shape (n_features, n_components)=(p,r)
        Initial basis-matrix estimate
    return_cost : bool (Default = False)
        Return per-iteration cost and diffW (increases computation)

    Returns
    -------
    W : array of shape (n_features,n_components) = (p,r)
        Non-negative Basis matrix
    """
    p = X.shape[0]

    # degree matrix
    D = np.diag(A.sum(axis=0))

    if W is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

    if return_cost:
        diffW_list=[]
        cost_list=[]

    start = time.time()
    XA = X.dot(A)
    XD = X.dot(D)
    for iter in range(max_iter):
        W_old = W

        XtW = X.T.dot(W)
        WtW = W.T.dot(W)

        # numerator term
        num = 2*X.dot(XtW) + lam*np.dot(XA,XtW)
        den = W.dot(XtW.T.dot(XtW)) + X.dot(XtW.dot(WtW)) + lam*np.dot(XD,XtW) + 1e-10

#        den = W.dot(XtW.T.dot(XtW)) + 1e-10
        if l2_pen is not None:
            # extra term arising from Frobenius norm regularizer
            den = den + l2_pen * W

        W = W* (num / den)
        W /= norm(W,2) # divide by largest singular value

        diffW = norm(W_old - W, 'fro') / norm(W_old, 'fro')

        if (diffW < tol) and (iter > 100): # allow at least 100 iter for "burn-in"
            break

        if return_cost:
            """keep track of relevant per-iteration values"""
            diffW_list.append(diffW)
            if l2_pen is None:
                cost_list.append(norm(X - W.dot(W.T.dot(X)),'fro')**2/2)
            else:
                cost_list.append(norm(X - W.dot(W.T.dot(X)),'fro')**2/2 +
                                 0.5*l2_pen*norm(W,'fro')**2)

        if iter % 500 == 0:
            obj_val = norm(X - W.dot(W.T.dot(X)),'fro')**2
            if l2_pen is not None:
                # add frobenius norm L2 penalty to objective
                obj_val = obj_val + 0.5*l2_pen*norm(W,'fro')**2
            print "iter = {:4}, diff = {:3.2e}, cost = {:6.5e} ({:3.1f} sec)".format(
                iter,diffW, obj_val,time.time()-start)
    #print iter, diffW
    if return_cost:
        return W, np.asarray(cost_list), np.asarray(diffW_list)
    else:
        return W


def nmf_admm(X,r=5,rho=1e3,max_iter=5000,tol=1e-4, disp_freq=500):
    """ NMF solved via ADMM, as proposed in 2010 Y. Zhang from Rice

    [1] 2010 Yin Zhang, "An Alternating direction algorithm for NMF"
    """
    p,n = X.shape

    # main primal variables
    W = np.random.rand(p,r) # X in Y. Zhang
    H = np.random.rand(r,n) # Y in Y. Zhang

    # auxiliary variables introduced via variable splitting
    Wtil = np.zeros(W.shape) # U in Y. Zhang
    Htil = np.zeros(H.shape) # V in Y. Zhang

    # dual variables
    Lam_W = np.zeros(W.shape) # Lambda in Y. Zhang
    Lam_H = np.zeros(H.shape) # Pie    in Y. Zhang

    eye_r = np.eye(r)

    # keep track of frobenius norm loss
    cost_list = []
    cost = norm(W.dot(H) - X,'fro')**2/2
    cost_list.append(cost)

    # keep track of relative prima residual
    res_W = []
    res_H = []

    # also keep track of diffW
    diffW = []

    start=time.time()
    #--- begin admm iterations ---#
    for iter in range(max_iter):
        W_old = W

        #--- primal updates (W,H,Wtil,Htil) ---#
        W = solve( dot(H,H.T) + rho*eye_r,
                   dot(H,X.T) + rho*Wtil.T - Lam_W.T).T

        H = solve( dot(W.T,W) + rho*eye_r,
                   dot(W.T,X) + rho*Htil - Lam_H)

        Wtil = np.maximum(0,W+Lam_W/rho)
        Htil = np.maximum(0,H+Lam_H/rho)

        #--- dual updates (Lam_W, Lam_H) ---
        Lam_W += rho*(W-Wtil)
        Lam_H += rho*(H-Htil)

        #--- compute main objective/cost ---#
        cost = norm(X - dot(W,H), 'fro')**2/2
        cost_list.append(cost)

        #--- compute relative primal residual ---#
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_H.append(norm(H-Htil,'fro')/norm(H,'fro'))
        diffW.append(norm(W-W_old,'fro')/norm(W_old,'fro'))
        if iter % disp_freq == 0:
#            print "iter = {:4}, cost = {:3.3f} ({:3.1f} sec)".format(
#                iter,  cost_list[iter+1],time.time()-start)
            str_ = "iter={:4} cost={:3.3f} ".format(iter,cost_list[iter+1])+\
                   "res_W={:3.2e} res_H={:3.2e} diffW={:3.2e} ({:3.1f} sec)".format(
                   res_W[iter],res_H[iter],diffW[iter],time.time()-start)
            print str_

    # convert residual in pandas dataframe
    res = pd.DataFrame([res_W,res_H,diffW],index=['res_W','res_H','diffW']).T
    return W,H,Wtil,Htil,cost,res


def pnmf_admm(X,r=5,rho=1e3,max_iter=5000,tol=1e-4,W_init='nndsvd',disp_freq=1000,
              silence=False):
    """ Projective NMF using ADMM.

    >>> W,P,H,Wtil,Ptil,cost_list,res = pnmf_admm(X,r,rho=1e2)

    The projective NMF here adopts the following parametrization:

    .. math ::

        \\min_{W,P} \\frac{1}{2}\|X - W(PX)\|^2_F
            \;\; \\text{s.t.}\;\;  W\ge 0, P \ge 0

    This is converted into the following equivalent constrained problem:

    .. math ::

        \\min_{W,P,H} \\frac{1}{2}\|X - WH\|^2_F
            + I_+(\\tilde{W}) + I_+(\\tilde{P})
            \\\\ \\quad \\text{s.t.}\;\;  W\ge 0, P \ge 0, H=PX,
                         W=\\tilde{W}, P=\\tilde{P}

        W,\\tilde{W} \in \mathbb{R}_+^{p\\times r}\\quad
        P,\\tilde{P} \in \mathbb{R}_+^{r\\times p}\\quad
        H \in \mathbb{R}_+^{r\\times n}\\quad

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space (n_components)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd' (default)
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **02/01/2016** - updates based on ``t_0127d2_revisit_pnmf_admm.py``

    - change initialization scheme and order of primal admm updates

    **02/02/2016**

    - added ``silence`` option
    - changed input ``W`` to ``W_init``
    """
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)


    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
#    cost = norm( X - dot(W,P.dot(X)), 'fro')**2/2
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+rho*eye_r, dot(W.T, X) + rho*dot(P,X) - Lam_H)
        W = solve(dot(H,H.T)+rho*eye_r, dot(H, X.T) + rho*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        cost = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
#        cost = norm( X - dot(W,P.dot(X)), 'fro')**2/2
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " res_W={:3.2e}".format(res_W[iter])
            str_ += " res_P={:3.2e}".format(res_P[iter])
            str_ += " res_H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    res = pd.DataFrame([res_W,res_P,res_H,diffcost],
                       index=['res_W','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,Wtil,Ptil,cost_list,res


def projection_spectral(W,projection_type='stiefel'):
    """ Projection onto Stiefel manifold or its convex relaxation.

    Parameters
    ----------
    W : ndarray of shape [p,r], p >= r
        Input matrix
    projection_type : string
        'stiefel' (default) or 'convex'

    Returns
    -------
    The projection of input ``W`` onto the Stiefel manifold or its convex
    relaxation.

    History
    -------
    Created 02/02/2016
    """
    U,S,V = svd(W,full_matrices=False)
    if projection_type == 'stiefel':
        return (U).dot(V)
    elif projection_type == 'convex':
        return (U*np.minimum(1,S)).dot(V)
    else:
        raise ValueError("Projection must be 'stiefel' or 'convex'")


def spnmf_admm(X,r=5,constraint='stiefel',rho=1e3,max_iter=5000,tol=1e-4,
               W_init='nndsvd',disp_freq=1000,silence=False):
    """ Spectral Projective NMF using ADMM.

    Projective NMF with either the orthogonal **Stiefel Manifold contraint**
    or the convex relaxation **

    Usage
    -----
    >>> W,P,H,W1,Ptil,cost_list,res = spnmf_admm(X, r=10, rho=1e2,
    >>> ... constraint='stiefel',max_iter=5000,tol=1e-4,W='nndsvd')

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space (n_components)
    constraint : ``'stiefel'`` or ``'convex'``
        The type of spectral constraint (default: ``'stiefel'``).  This
        determines the type of spectral projection applied.
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None,  rng, or 'nndsvd' (default)
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    History
    --------------
    **02/01/2016**

    - updates based on ``t_0127d2_revisit_pnmf_admm.py``
    - change initialization scheme and order of primal admm updates

    **02/02/2016**

    - Function moved to my nmf_module.  cleaned up codes and docstring.

    **02/03/2016**

    - removed ``W2`` from output for consistency with other PNMFtype methods.
    """
    p,n = X.shape
    #%%=== initialize variables =====
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    W1 = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W1 = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)


    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost = norm(W1.dot(Ptil.dot(X)) - X,'fro')**2/2
#    cost = norm( X - dot(W,P.dot(X)), 'fro')**2/2
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W1 = []
    res_W2 = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+rho*eye_r, dot(W.T, X) + rho*dot(P,X) - Lam_H)
        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*(W1+W2).T - Lam_W1.T - Lam_W2.T).T

        # projections
        W1 = (W+Lam_W1/rho).clip(0)
        W2 = projection_spectral(W+Lam_W2/rho,projection_type=constraint)

        #====================== dual updates =================================#
        Lam_W1 = Lam_W1 + rho*(W - W1)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        cost = norm(W1.dot(Ptil.dot(X)) - X,'fro')**2/2
#        cost = norm( X - dot(W,P.dot(X)), 'fro')**2/2
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W1.append(norm(W-W1,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check =============#
        check_exit =  (diffcost[iter] < tol)
        check_exit &= (  res_W1[iter] < tol)
        check_exit &= (  res_W2[iter] < tol)
        check_exit &= (   res_P[iter] < tol)
        check_exit &= (   res_H[iter] < tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W1={:3.2e}".format(res_W1[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    res = pd.DataFrame([res_W1,res_W2,res_P,res_H,diffcost],
                       index=['res_W1','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,W1,Ptil,cost_list,res


def ls_pnmf_admm(X,y,r=5,gam=1e-1,rho=1e3,max_iter=5000,tol=1e-4,
                 W_init='nndsvd',disp_freq=1000,silence=False,add_intercept=False):
    """ Projective NMF using ADMM.

    >>> W,P,H,Wtil,Ptil,cost_list,res = pnmf_admm(X,r,rho=1e2)

    The projective NMF here adopts the following parametrization:

    .. math ::

        \\min_{W,P} \\frac{1}{2}\|X - W(PX)\|^2_F
            \;\; \\text{s.t.}\;\;  W\ge 0, P \ge 0

    This is converted into the following equivalent constrained problem:

    .. math ::

        \\min_{W,P,H} \\frac{1}{2}\|X - WH\|^2_F
            + I_+(\\tilde{W}) + I_+(\\tilde{P})
            \\\\ \\quad \\text{s.t.}\;\;  W\ge 0, P \ge 0, H=PX,
                         W=\\tilde{W}, P=\\tilde{P}

        W,\\tilde{W} \in \mathbb{R}_+^{p\\times r}\\quad
        P,\\tilde{P} \in \mathbb{R}_+^{r\\times p}\\quad
        H \in \mathbb{R}_+^{r\\times n}\\quad

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int
        Dimension of the embedding space (n_components)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.
    add_intercept : bool (default=False)
        Add intercept term to classifier (added 02/17/2016)

    Update history
    --------------
    **02/01/2016** - updates based on ``t_0127d2_revisit_pnmf_admm.py``

    - change initialization scheme and order of primal admm updates

    **02/02/2016**

    - added ``silence`` option
    - changed input ``W`` to ``W_init``

    **02/17/2016** - added ``add_intercept`` option
    """
    p,n = X.shape
    gam *= 1.# <- ensure float
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    if add_intercept:
        w = np.zeros(r+1) # <- classification vector
    else:
        w = np.zeros(r) # <- classification vector

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)


    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_ls_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
    if add_intercept:
        # used for the least-squares term
        one_n = np.ones(n)
        PX = np.vstack((PX,one_n))
    cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2
    cost = cost_nmf + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_ls_list.append(cost_ls)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        if add_intercept:
            ytil = y - one_n*w[-1]
            H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w[:r],w[:r]),
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + gam*np.outer(w[:r],ytil))
        else:
            H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w,w),
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + gam*np.outer(w,y))

        if add_intercept:
            # H with intercept
            H_ = np.vstack((H,one_n))
            # for now, ignore gam2 regularizer...i never use them
            w = solve( H_.dot(H_.T), H_.dot(y))
        else:
            w = solve( H.dot(H.T) , H.dot(y))

        W = solve(dot(H,H.T)+rho*eye_r, dot(H, X.T) + rho*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
        if add_intercept: # for computing LS loss term
            PX = np.vstack((PX,one_n))
        cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2
        cost = cost_nmf + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_ls_list.append(cost_ls)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "    Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " nmf={:.2e}".format(cost_nmf_list[iter+1])
            str_ += " ls={:.2e}".format(cost_ls_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " res_W={:3.2e}".format(res_W[iter])
            str_ += " res_P={:3.2e}".format(res_P[iter])
            str_ += " res_H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
#    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_ls_list],
                        index=['total','nmf','ls']).T
    res = pd.DataFrame([res_W,res_P,res_H,diffcost],
                       index=['res_W','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res

def ls_pnmf_admm_bal(X,y,r=5,gam=1e-1,rho=1e3,max_iter=5000,tol=1e-4,
                 W_init='nndsvd',disp_freq=1000,silence=False,add_intercept=False):
    """ Same as ``ls_pnmf_admm``, but with class-weighted loss

    >>> W,P,H,Wtil,Ptil,cost_list,res = ls_pnmf_admm_bal(X,r,rho=1e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int
        Dimension of the embedding space (n_components)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.
    add_intercept : bool (default=False)
        Add intercept term to classifier (added 02/17/2016)

    Update history
    --------------
    **02/17/2016** - created
    """
    p,n = X.shape
    rho *= 1.
    gam *= 1.# <- ensure float
    #%% get class balance info
    weight_pos = 1.*(y==-1).sum()/n
    weight_neg = 1 - weight_pos
    weight_vec = np.ones(n)
    weight_vec[y==+1] = weight_pos
    weight_vec[y==-1] = weight_neg
    J = np.diag( weight_vec )
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    if add_intercept:
        w = np.zeros(r+1) # <- classification vector
    else:
        w = np.zeros(r) # <- classification vector

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)


    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_ls_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
    if add_intercept:
        # used for the least-squares term
        one_n = np.ones(n)
        PX = np.vstack((PX,one_n))
    # compute weighted error
    _err = (y-PX.T.dot(w))
    _err = _err.T.dot( J.dot(_err))
    cost_ls  = gam*_err/2
    #cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2
    cost = cost_nmf + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_ls_list.append(cost_ls)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        WtW = dot(W.T,W) + rho*eye_r
        if add_intercept:
            ytil = y - one_n*w[-1]
            _S = dot(W.T, X) + rho*dot(P,X) - Lam_H + gam*np.outer(w[:r],ytil)
            wtw = gam*np.outer(w[:r],w[:r])
            #H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w[:r],w[:r]),_S)
        else:
            _S = dot(W.T, X) + rho*dot(P,X) - Lam_H + gam*np.outer(w,y)
            wtw = gam*np.outer(w,w)
            #H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w,w),_S)

        H[:,y==+1] = solve(WtW + weight_pos*wtw, _S[:,y==+1])
        H[:,y==-1] = solve(WtW + weight_neg*wtw, _S[:,y==-1])

        if add_intercept:
            # H with intercept
            H_ = np.vstack((H,one_n))
            # for now, ignore gam2 regularizer...i never use them
            w = solve( H_.dot(J.dot(H_.T)), H_.dot(J.dot(y)))
        else:
            w = solve( H.dot(J.dot(H.T)) , H.dot(J.dot(y)))

        W = solve(dot(H,H.T)+rho*eye_r, dot(H, X.T) + rho*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
        if add_intercept: # for computing LS loss term
            PX = np.vstack((PX,one_n))
        # compute weighted error
        _err = (y-PX.T.dot(w))
        _err = _err.T.dot( J.dot(_err))
        cost_ls  = gam*_err/2
        #cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2
        cost = cost_nmf + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_ls_list.append(cost_ls)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "    Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " nmf={:.2e}".format(cost_nmf_list[iter+1])
            str_ += " ls={:.2e}".format(cost_ls_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " res_W={:3.2e}".format(res_W[iter])
            str_ += " res_P={:3.2e}".format(res_P[iter])
            str_ += " res_H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
#    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_ls_list],
                        index=['total','nmf','ls']).T
    res = pd.DataFrame([res_W,res_P,res_H,diffcost],
                       index=['res_W','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res


def ls_spnmf_admm(X,y,r=5,gam=1e-1,rho=1e3,max_iter=5000,tol=1e-4,
                  W_init='nndsvd',disp_freq=2000,silence=False,
                  constraint='stiefel',gam2=0,add_intercept=False):
    """ Projective NMF using ADMM.  gam2 = regularizer for classifier

    >>> W,P,H,w,Wtil,Ptil,cost,res = pnmf_admm(X,r,rho=1e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int
        Dimension of the embedding space (n_components)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.
    add_intercept : bool (default=False)
        Add intercept term to classifier (added 02/17/2016)

    Update history
    --------------
    **02/02/2016** - created but dirty code

    **02/03/2016** - Added regularizer    ``gam2``

    **02/17/2016** - added ``add_intercept`` option
    """
    p,n = X.shape
    rho *= 1.
    gam *= 1.# <- ensure float
    gam2 *= 1.# <- ensure float
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    if add_intercept:
        w = np.zeros(r+1) # <- classification vector
    else:
        w = np.zeros(r) # <- classification vector

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)


    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_ls_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
    if add_intercept:
        # used for the least-squares term
        one_n = np.ones(n)
        PX = np.vstack((PX,one_n))
    cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
    cost = cost_nmf + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_ls_list.append(cost_ls)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        if add_intercept:
            ytil = y - one_n*w[-1]
            H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w[:r],w[:r]),
                      dot(W.T, X)+rho*dot(P,X)-Lam_H + gam*np.outer(w[:r],ytil))
        else:
            H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w,w),
                      dot(W.T, X)+rho*dot(P,X)-Lam_H + gam*np.outer(w,y))

        if add_intercept:
            # H with intercept
            H_ = np.vstack((H,one_n))
            # for now, ignore gam2 regularizer...i never use them
            w = solve( H_.dot(H_.T), H_.dot(y))
        else:
            w = solve( H.dot(H.T) + (gam2/gam)*eye_r, H.dot(y))

        W = solve(dot(H,H.T)+2*rho*eye_r,
                  dot(H, X.T) + rho*(Wtil+W2).T - Lam_W.T - Lam_W2.T).T

        Wtil = (W+Lam_W/rho).clip(0)
        W2 = projection_spectral(W+Lam_W2/rho,constraint)

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
        if add_intercept: # for computing LS loss term
            PX = np.vstack((PX,one_n))
        cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
        cost = cost_nmf + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_ls_list.append(cost_ls)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_W2[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "    Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " nmf={:.2e}".format(cost_nmf_list[iter+1])
            str_ += " ls={:.2e}".format(cost_ls_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
#    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_ls_list],
                        index=['total','nmf','ls']).T
    res = pd.DataFrame([res_W,res_W2,res_P,res_H,diffcost],
                       index=['res_W','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res



def ls_spnmf_admm_bal(X,y,r=5,gam=1e-1,rho=1e3,max_iter=5000,tol=1e-4,
                  W_init='nndsvd',disp_freq=2000,silence=False,
                  constraint='stiefel',gam2=0,add_intercept=False):
    """ PSame as ``ls_spnmf_admm``, but with class-weighted loss

    >>> W,P,H,w,Wtil,Ptil,cost,res = ls_spnmf_admm_bal(X,y,r,rho=1e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int
        Dimension of the embedding space (n_components)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.
    add_intercept : bool (default=False)
        Add intercept term to classifier (added 02/17/2016)

    Update history
    --------------
    **02/17/2016** - function created
    """
    p,n = X.shape
    rho *= 1.
    gam *= 1.# <- ensure float
    gam2 *= 1.# <- ensure float
    #%% get class balance info
    weight_pos = 1.*(y==-1).sum()/n
    weight_neg = 1 - weight_pos
    weight_vec = np.ones(n)
    weight_vec[y==+1] = weight_pos
    weight_vec[y==-1] = weight_neg
    J = np.diag( weight_vec )
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    if add_intercept:
        w = np.zeros(r+1) # <- classification vector
    else:
        w = np.zeros(r) # <- classification vector

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)


    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_ls_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
    if add_intercept:
        # used for the least-squares term
        one_n = np.ones(n)
        PX = np.vstack((PX,one_n))
    # compute weighted error
    _err = (y-PX.T.dot(w))
    _err = _err.T.dot( J.dot(_err))
    cost_ls  = gam*_err/2 + gam2*norm(w)**2/2
    #cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
    cost = cost_nmf + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_ls_list.append(cost_ls)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        WtW = dot(W.T,W) + rho*eye_r
        if add_intercept:
            ytil = y - one_n*w[-1]
            _S = dot(W.T, X)+rho*dot(P,X)-Lam_H + gam*np.outer(w[:r],ytil)
            wtw = gam*np.outer(w[:r],w[:r])
            #H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w[:r],w[:r]),
            #          dot(W.T, X)+rho*dot(P,X)-Lam_H + gam*np.outer(w[:r],ytil))
        else:
            _S = dot(W.T, X)+rho*dot(P,X)-Lam_H + gam*np.outer(w,y)
            wtw = gam*np.outer(w,w)
            #H = solve(dot(W.T,W) + rho*eye_r + gam*np.outer(w,w),
            #          dot(W.T, X)+rho*dot(P,X)-Lam_H + gam*np.outer(w,y))
        H[:,y==+1] = solve(WtW + weight_pos*wtw, _S[:,y==+1])
        H[:,y==-1] = solve(WtW + weight_neg*wtw, _S[:,y==-1])


        if add_intercept:
            # H with intercept
            H_ = np.vstack((H,one_n))
            # for now, ignore gam2 regularizer...i never use them
            w = solve( H_.dot(J.dot(H_.T)), H_.dot(J.dot(y)))
        else:
            w = solve( H.dot(J.dot(H.T)) + (gam2/gam)*eye_r, H.dot(J.dot(y)))

        W = solve(dot(H,H.T)+2*rho*eye_r,
                  dot(H, X.T) + rho*(Wtil+W2).T - Lam_W.T - Lam_W2.T).T

        Wtil = (W+Lam_W/rho).clip(0)
        W2 = projection_spectral(W+Lam_W2/rho,constraint)

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
        if add_intercept: # for computing LS loss term
            PX = np.vstack((PX,one_n))
        # compute weighted error
        _err = (y-PX.T.dot(w))
        _err = _err.T.dot( J.dot(_err))
        cost_ls  = gam*_err/2 + gam2*norm(w)**2/2
        #cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
        cost = cost_nmf + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_ls_list.append(cost_ls)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_W2[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "    Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " nmf={:.2e}".format(cost_nmf_list[iter+1])
            str_ += " ls={:.2e}".format(cost_ls_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
#    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_ls_list],
                        index=['total','nmf','ls']).T
    res = pd.DataFrame([res_W,res_W2,res_P,res_H,diffcost],
                       index=['res_W','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res



#%% === graph regularized methods
def gpnmf_admm(X,r,L,tau=1e-1,rho=1e3,max_iter=5000,tol=1e-4,W_init='nndsvd',
               disp_freq=1000,silence=False):
    """ Graph regularized Projective NMF using ADMM.

    >>> W,P,H,Wtil,Ptil,cost_list,res = gpnmf_admm(X,r,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space (n_components)
    L : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix.
    tau : float
        Amount of graph regularization to apply
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None (default), rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **02/03/2016** - function created
    """
    rho *= 1.
    tau *= 1.
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_graph_list = []

    cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
    cost = cost_nmf + cost_graph

    cost_nmf_list.append(cost_nmf)
    cost_graph_list.append(cost_graph)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+(2*rho)*eye_r,
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil - Lam_Htil )
        W = solve(dot(H,H.T)+rho*eye_r, dot(H, X.T) + rho*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        Htil = solve(tau*L + rho*eye_n, rho*H.T+Lam_Htil.T).T

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
        cost = cost_nmf + cost_graph

        cost_nmf_list.append(cost_nmf)
        cost_graph_list.append(cost_graph)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " graph={:3.2e}".format(cost_graph)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " Htil={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list],
                        index=['total','nmf','graph']).T

    res = pd.DataFrame([res_W,res_P,res_H,diffcost],
                       index=['res_W','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,Wtil,Ptil,cost,res


def sgpnmf_admm(X,r,L,tau=1e-1,rho=1e3,max_iter=5000,tol=1e-4,W_init='nndsvd',
               disp_freq=1000,silence=False):
    """ Spectral Graph regularized Spectral Projective NMF using ADMM.

    >>> W,P,H,Wtil,Ptil,cost_list,res = gpnmf_admm(X,r,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space (n_components)
    L : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix.
    tau : float
        Amount of graph regularization to apply
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None (default), rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **02/03/2016** - function created
    """
    rho *= 1.
    tau *= 1.
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)  # <- for spectral projection
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape) # <- for spectral projection
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_graph_list = []

    cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
    cost = cost_nmf + cost_graph

    cost_nmf_list.append(cost_nmf)
    cost_graph_list.append(cost_graph)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+(2*rho)*eye_r,
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil - Lam_Htil )
        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*Wtil.T - Lam_W.T + rho*W2.T - Lam_W2.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        W2 = projection_spectral(W + Lam_W2/rho)

        Htil = solve(tau*L + rho*eye_n, rho*H.T+Lam_Htil.T).T

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
        cost = cost_nmf + cost_graph

        cost_nmf_list.append(cost_nmf)
        cost_graph_list.append(cost_graph)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_W2[iter] < tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " G={:3.2e}".format(cost_graph)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " H2={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list],
                        index=['total','nmf','graph']).T

    res = pd.DataFrame([res_W,res_W2,res_P,res_H,diffcost],
                       index=['res_W','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,Wtil,Ptil,cost,res


def ls_sgpnmf_admm(X,y,r,L,tau=1e-1,gam=1e-1,gam2=0,rho=1e3,max_iter=5000,
                   tol=1e-4,W_init='nndsvd',disp_freq=1000,silence=False,
                   add_intercept=False):
    """ Graph regularized Spectral Projective NMF using ADMM.

    >>> W,P,H,w,Wtil,Ptil,cost,res = ls_sgpnmf_admm(X,y,r,L,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int
        Dimension of the embedding space (n_components)
    L : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix.
    tau : float
        Amount of graph regularization to apply
    gam : float
        Amount of weight on the classification loss
    gam2 : float (default=0)
        Amount of regularization on the classifier (results in ridge regression
        update)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end
    add_intercept : bool (default=False)
        Add intercept term to classifier (added 02/16/2016)

    Update history
    --------------
    **02/04/2016** - function created

    **02/16/2016** - added ``add_intercept`` option
    """
    rho *= 1.
    tau *= 1.
    gam *= 1.
    gam2 *= 1.
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    if add_intercept:
        w = np.zeros(r+1) # <- classification vector
    else:
        w = np.zeros(r) # <- classification vector

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)  # <- for spectral projection
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape) # <- for spectral projection
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_ls_list = []
    cost_nmf_list = []
    cost_graph_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
    if add_intercept:
        # used for the least-squares term
        one_n = np.ones(n)
        PX = np.vstack((PX,one_n))
    cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
    cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
    cost = cost_nmf + cost_graph + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_ls_list.append(cost_ls)
    cost_graph_list.append(cost_graph)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        if add_intercept:
            ytil = y - one_n*w[-1]
            H = solve(dot(W.T,W)+(2*rho)*eye_r + gam*np.outer(w[:r],w[:r]),
                      dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil -
                      Lam_Htil + gam*np.outer(w[:r],ytil))
        else:
            H = solve(dot(W.T,W)+(2*rho)*eye_r + gam*np.outer(w,w),
                      dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil -
                      Lam_Htil + gam*np.outer(w,y))
        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*Wtil.T - Lam_W.T + rho*W2.T - Lam_W2.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        W2 = projection_spectral(W + Lam_W2/rho)

        Htil = solve(tau*L + rho*eye_n, rho*H.T+Lam_Htil.T).T

        if add_intercept:
            # H with intercept
            H_ = np.vstack((H,one_n))
            # for now, ignore gam2 regularizer...i never use them
            w = solve( H_.dot(H_.T), H_.dot(y))
        else:
            w = solve( H.dot(H.T) + (gam2/gam)*eye_r, H.dot(y))

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
        if add_intercept: # for computing LS loss term
            PX = np.vstack((PX,one_n))
        cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
        cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
        cost = cost_nmf + cost_graph + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_ls_list.append(cost_ls)
        cost_graph_list.append(cost_graph)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_W2[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "    Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " G={:3.2e}".format(cost_graph)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " H2={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list,cost_ls_list],
                        index=['total','nmf','graph','ls']).T

    res = pd.DataFrame([res_W,res_W2,res_P,res_H,diffcost],
                       index=['res_W','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res

def ls_sgpnmf_admm_bal(X,y,r,L,tau=1e-1,gam=1e-1,gam2=0,rho=1e3,max_iter=5000,
                   tol=1e-4,W_init='nndsvd',disp_freq=1000,silence=False,
                   add_intercept=False):
    """ Same as ``ls_sgpnmf_admm``, but with class-weighted loss

    >>> W,P,H,w,Wtil,Ptil,cost,res = ls_sgpnmf_admm(X,y,r,L,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int
        Dimension of the embedding space (n_components)
    L : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix.
    tau : float
        Amount of graph regularization to apply
    gam : float
        Amount of weight on the classification loss
    gam2 : float (default=0)
        Amount of regularization on the classifier (results in ridge regression
        update)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end
    add_intercept : bool (default=False)
        Add intercept term to classifier (added 02/16/2016)

    Update history
    --------------
    **02/17/2016** - function created
    """
    rho *= 1.
    tau *= 1.
    gam *= 1.
    gam2 *= 1.
    p,n = X.shape
    #%% get class balance info
    weight_pos = 1.*(y==-1).sum()/n
    weight_neg = 1 - weight_pos
    weight_vec = np.ones(n)
    weight_vec[y==+1] = weight_pos
    weight_vec[y==-1] = weight_neg
    J = np.diag( weight_vec )
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    if add_intercept:
        w = np.zeros(r+1) # <- classification vector
    else:
        w = np.zeros(r) # <- classification vector

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)  # <- for spectral projection
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape) # <- for spectral projection
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_ls_list = []
    cost_nmf_list = []
    cost_graph_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
    if add_intercept:
        # used for the least-squares term
        one_n = np.ones(n)
        PX = np.vstack((PX,one_n))
    # compute weighted error
    _err = (y-PX.T.dot(w))
    _err = _err.T.dot( J.dot(_err))
    cost_ls  = gam*_err/2 + gam2*norm(w)**2/2
    #cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
    cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
    cost = cost_nmf + cost_graph + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_ls_list.append(cost_ls)
    cost_graph_list.append(cost_graph)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        WtW = W.T.dot(W) + (2*rho)*eye_r
        if add_intercept:
            ytil = y - one_n*w[-1]
            _S = dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil -\
                      Lam_Htil + gam*np.outer(w[:r],ytil)
            #H = solve(dot(W.T,W)+(2*rho)*eye_r + gam*np.outer(w[:r],w[:r]),_S)
            wtw = gam*np.outer(w[:r],w[:r])
        else:
            _S = dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil -\
                      Lam_Htil + gam*np.outer(w,y)
            #H = solve(dot(W.T,W)+(2*rho)*eye_r + gam*np.outer(w,w),_S)
            wtw = gam*np.outer(w,w)
        H[:,y==+1] = solve(WtW + weight_pos*wtw, _S[:,y==+1])
        H[:,y==-1] = solve(WtW + weight_neg*wtw, _S[:,y==-1])

        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*Wtil.T - Lam_W.T + rho*W2.T - Lam_W2.T).T
        Wtil = (W+Lam_W/rho).clip(0)


        W2 = projection_spectral(W + Lam_W2/rho)

        Htil = solve(tau*L + rho*eye_n, rho*H.T+Lam_Htil.T).T

        if add_intercept:
            # H with intercept
            H_ = np.vstack((H,one_n))
            # for now, ignore gam2 regularizer...i never use them
            w = solve( H_.dot(J.dot(H_.T)), H_.dot(J.dot(y)))
        else:
            w = solve( H.dot(J.dot(H.T)) + (gam2/gam)*eye_r, H.dot(J.dot(y)))

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(Wtil.dot(PX) - X,'fro')**2/2
        if add_intercept: # for computing LS loss term
            PX = np.vstack((PX,one_n))
        # compute weighted error
        _err = (y-PX.T.dot(w))
        _err = _err.T.dot( J.dot(_err))
        cost_ls  = gam*_err/2 + gam2*norm(w)**2/2
#        cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2 + gam2*norm(w)**2/2
        cost_graph = tau*trace( L.dot( dot(Htil.T, Htil)))
        cost = cost_nmf + cost_graph + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_ls_list.append(cost_ls)
        cost_graph_list.append(cost_graph)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_W2[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "    Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " G={:3.2e}".format(cost_graph)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " H2={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list,cost_ls_list],
                        index=['total','nmf','graph','ls']).T

    res = pd.DataFrame([res_W,res_W2,res_P,res_H,diffcost],
                       index=['res_W','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res


def pnge_admm(X,r1,r2,L1,L2,tau=1e-1,rho=1e3,max_iter=5000,tol=1e-4,
              W_init='nndsvd',disp_freq=1000,silence=False):
    """ Projective non-negative graph embedding using ADMM.

    Note: in my Onenote, I used H_s, H_p...these are H_1 and H_2 in this code
    (basically, replace all (s,p) pairs with (1,2) pair)

    >>> W,P,H,Wtil,Ptil,cost_list,res = gpnmf_admm(X,r,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r1 : int
        Dimension of the intrinsic embedding space (n_components)
    r2 : int
        Dimension of the complementary embedding space (n_components)
    L1 : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix of intrinsic graph
    L2 : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix of penalty graph.
    tau : float
        Amount of graph regularization to apply
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **02/03/2016** - function created
    """
    r = r1+r2
    rho *= 1.
    tau *= 1.
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    idx_i = np.arange(r1)       # <- intrinsic graph indices (top half of H matrices)
    idx_p = np.arange(r1,r1+r2) # <- penalty   graph indices (bot half of H matrices)
#    Htil1 = np.zeros((r1,n)) # <- the top half part of Htil (intrinsic part)
#    Htil2 = np.zeros((r2,n)) # <- the bot half part of Htil (complementary part)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_graph_list = []

    cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    _term1 = trace(L1.dot( dot(Htil[idx_i].T,Htil[idx_i])))
    _term2 = trace(L2.dot( dot(Htil[idx_p].T,Htil[idx_p])))
    cost_graph = (tau/2)*(_term1 + _term2)
    cost = cost_nmf + cost_graph

    cost_nmf_list.append(cost_nmf)
    cost_graph_list.append(cost_graph)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+(2*rho)*eye_r,
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil - Lam_Htil )
        W = solve(dot(H,H.T)+rho*eye_r, dot(H, X.T) + rho*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        # --- Htil updates...broken down to two part: Htil1 and Htil2 ---
        # intrinsic graph update (top half of Htil matrix)
        Htil[idx_i] = solve(tau*L1 + rho*eye_n, rho*H[idx_i].T+Lam_Htil[idx_i].T).T

        # penalty graph update (bottom half of Htil matrix)
        Htil[idx_p] = solve(tau*L2 + rho*eye_n, rho*H[idx_p].T+Lam_Htil[idx_p].T).T

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        _term1 = trace(L1.dot( dot(Htil[idx_i].T,Htil[idx_i])))
        _term2 = trace(L2.dot( dot(Htil[idx_p].T,Htil[idx_p])))
        cost_graph = (tau/2)*(_term1 + _term2)
        cost = cost_nmf + cost_graph

        cost_nmf_list.append(cost_nmf)
        cost_graph_list.append(cost_graph)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " graph={:3.2e}".format(cost_graph)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " Htil={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list],
                        index=['total','nmf','graph']).T

    res = pd.DataFrame([res_W,res_P,res_H,diffcost],
                       index=['res_W','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,Wtil,Ptil,cost,res



def spnge_admm(X,r1,r2,L1,L2,tau=1e-1,rho=1e3,max_iter=5000,tol=1e-4,
               W_init='nndsvd',disp_freq=1000,silence=False):
    """ Projective non-negative graph embedding using ADMM.

    Note: in my Onenote, I used H_s, H_p...these are H_1 and H_2 in this code
    (basically, replace all (s,p) pairs with (1,2) pair)

    >>> W,P,H,Wtil,Ptil,cost_list,res = gpnmf_admm(X,r,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    r1 : int
        Dimension of the intrinsic embedding space (n_components)
    r2 : int
        Dimension of the complementary embedding space (n_components)
    L1 : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix of intrinsic graph
    L2 : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix of penalty graph.
    tau : float
        Amount of graph regularization to apply
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None (default), rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **02/04/2016** - function created
    """
    r = r1+r2
    rho *= 1.
    tau *= 1.
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)   # <- for spectral projection
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    idx_i = np.arange(r1)       # <- intrinsic graph indices (top half of H matrices)
    idx_p = np.arange(r1,r1+r2) # <- penalty   graph indices (bot half of H matrices)
#    Htil1 = np.zeros((r1,n)) # <- the top half part of Htil (intrinsic part)
#    Htil2 = np.zeros((r2,n)) # <- the bot half part of Htil (complementary part)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)   # <- for spectral projection
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_graph_list = []

    cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    _term1 = trace(L1.dot( dot(Htil[idx_i].T,Htil[idx_i])))
    _term2 = trace(L2.dot( dot(Htil[idx_p].T,Htil[idx_p])))
    cost_graph = (tau/2)*(_term1 + _term2)
    cost = cost_nmf + cost_graph

    cost_nmf_list.append(cost_nmf)
    cost_graph_list.append(cost_graph)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+(2*rho)*eye_r,
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + rho*Htil - Lam_Htil )
        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*Wtil.T - Lam_W.T + rho*W2.T - Lam_W2.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        # specral projection on two column blocks separately
        _W = W + Lam_W2/rho
        W2[:,idx_i] = projection_spectral(_W[:,idx_i])
        W2[:,idx_p] = projection_spectral(_W[:,idx_p])

        # --- Htil updates...broken down to two part: Htil1 and Htil2 ---
        # intrinsic graph update (top half of Htil matrix)
        Htil[idx_i] = solve(tau*L1 + rho*eye_n, rho*H[idx_i].T+Lam_Htil[idx_i].T).T

        # penalty graph update (bottom half of Htil matrix)
        Htil[idx_p] = solve(tau*L2 + rho*eye_n, rho*H[idx_p].T+Lam_Htil[idx_p].T).T

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        _term1 = trace(L1.dot( dot(Htil[idx_i].T,Htil[idx_i])))
        _term2 = trace(L2.dot( dot(Htil[idx_p].T,Htil[idx_p])))
        cost_graph = (tau/2)*(_term1 + _term2)
        cost = cost_nmf + cost_graph

        cost_nmf_list.append(cost_nmf)
        cost_graph_list.append(cost_graph)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " graph={:3.2e}".format(cost_graph)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " Htil={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list],
                        index=['total','nmf','graph']).T

    res = pd.DataFrame([res_W,res_W2,res_P,res_H,diffcost],
                       index=['res_W','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,Wtil,Ptil,cost,res


def ls_pnge_admm(X,y,r1,r2,L1,L2,tau=1e-1,gam=1e-2,rho=1e3,max_iter=5000,
                 tol=1e-4,W_init='nndsvd',disp_freq=1000,silence=False):
    """ Projective non-negative graph embedding using ADMM.

    Note: in my Onenote, I used H_s, H_p...these are H_1 and H_2 in this code
    (basically, replace all (s,p) pairs with (1,2) pair)

    >>> W,P,H,Wtil,Ptil,cost_list,res = gpnmf_admm(X,r,rho=5e2)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r1 : int
        Dimension of the intrinsic embedding space (n_components)
    r2 : int
        Dimension of the complementary embedding space (n_components)
    L1 : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix of intrinsic graph
    L2 : ndarray of shape = [n_samples, n_samples]
        Laplacian matrix of penalty graph.
    tau : float
        Amount of graph regularization to apply
    gam : float
        Amount of weight on the classification loss
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None (default), rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **02/03/2016** - function created
    """
    r = r1+r2
    rho *= 1.
    tau *= 1.
    gam *= 1.
    p,n = X.shape
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)
    Htil = np.zeros(H.shape) # <- for graph regularization

    idx_i = np.arange(r1)       # <- intrinsic graph indices (top half of H matrices)
    idx_p = np.arange(r1,r1+r2) # <- penalty   graph indices (bot half of H matrices)
#    Htil1 = np.zeros((r1,n)) # <- the top half part of Htil (intrinsic part)
#    Htil2 = np.zeros((r2,n)) # <- the bot half part of Htil (complementary part)

    # classifier
    w = np.zeros(r1)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    Lam_Htil = np.zeros(H.shape) # <- for graph regularization
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_ls_list = []
    cost_graph_list = []

    cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    cost_ls  = gam*norm(y - H[idx_i].T.dot(w)) **2/2
    _term1 = trace(L1.dot( dot(Htil[idx_i].T,Htil[idx_i])))
    _term2 = trace(L2.dot( dot(Htil[idx_p].T,Htil[idx_p])))
    cost_graph = (tau/2)*(_term1 + _term2)
    cost = cost_nmf + cost_graph + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_graph_list.append(cost_graph)
    cost_ls_list.append(cost_ls)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []
    res_Htil = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)
    eye_n = np.eye(n)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # intrinsic H update
        Wi = W[:,idx_i]
        Wp = W[:,idx_p]
        H[idx_i] = solve(dot(Wi.T,Wi)+(2*rho)*np.eye(r1) + gam*np.outer(w,w),
                  dot(Wi.T, X) + rho*dot(P[idx_i],X) - Lam_H[idx_i] +
                  rho*Htil[idx_i] - Lam_Htil[idx_i] + gam*np.outer(w,y) )
        H[idx_p] = solve(dot(Wp.T,Wp)+(2*rho)*np.eye(r2),
                  dot(Wp.T, X) + rho*dot(P[idx_p],X) - Lam_H[idx_p] +
                  rho*Htil[idx_p] - Lam_Htil[idx_p] )
        w = solve( H[idx_i].dot(H[idx_i].T), H[idx_i].dot(y))

        W = solve(dot(H,H.T)+rho*eye_r, dot(H, X.T) + rho*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho).clip(0)

        # --- Htil updates...broken down to two part: Htil1 and Htil2 ---
        # intrinsic graph update (top half of Htil matrix)
        Htil[idx_i] = solve(tau*L1 + rho*eye_n, rho*H[idx_i].T+Lam_Htil[idx_i].T).T

        # penalty graph update (bottom half of Htil matrix)
        Htil[idx_p] = solve(tau*L2 + rho*eye_n, rho*H[idx_p].T+Lam_Htil[idx_p].T).T

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        Lam_Htil = Lam_Htil + rho*(H-Htil)

        #============ comptute objective values ==============================#
        # main loss function
        cost_nmf = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        cost_ls  = gam*norm(y - H[idx_i].T.dot(w)) **2/2
        _term1 = trace(L1.dot( dot(Htil[idx_i].T,Htil[idx_i])))
        _term2 = trace(L2.dot( dot(Htil[idx_p].T,Htil[idx_p])))
        cost_graph = (tau/2)*(_term1 + _term2)
        cost = cost_nmf + cost_graph + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_graph_list.append(cost_graph)
        cost_ls_list.append(cost_ls)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_Htil.append(norm(H-Htil,'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_Htil[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:3.2e}".format(cost_nmf)
            str_ += " graph={:3.2e}".format(cost_graph)
            str_ += " ls={:3.2e}".format(cost_ls)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " Htil={:3.2e}".format(res_Htil[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert to pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_graph_list,cost_ls_list],
                        index=['total','nmf','graph','ls']).T

    res = pd.DataFrame([res_W,res_P,res_H,diffcost],
                       index=['res_W','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost,res
#%% === class ====
class PNMF_ADMM(BaseEstimator,TransformerMixin):
    def __init__(self,r=10,rho=5e2,W_init=None,max_iter=5000,tol=1e-4,
                 disp_freq=50000,silence=True):
        self.r = r
        self.rho = rho
        self.W_init = W_init
        self.max_iter = max_iter
        self.tol = tol
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        W,P,H,Wtil,Ptil,cost_list,res = pnmf_admm(X.T,r=self.r, rho=self.rho,
            W_init=self.W_init, max_iter=self.max_iter, tol=self.tol,
            disp_freq=self.disp_freq, silence=self.silence)
        self.W_ = Wtil
        self.P_ = Ptil
        self.H_ = H
        self.cost_ = cost_list
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_.T)


class SPNMF_ADMM(BaseEstimator,TransformerMixin):
    def __init__(self,r=10,rho=5e2,constraint='stiefel',W_init=None,
                 max_iter=5000,tol=1e-4,disp_freq=50000,silence=True):
        self.r = r
        self.rho = rho
        self.constraint = constraint
        self.W_init = W_init
        self.max_iter = max_iter
        self.tol = tol
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        W,P,H,W1,Ptil,cost_list,res = spnmf_admm(X.T,r=self.r,
            constraint=self.constraint, rho=self.rho, max_iter=self.max_iter,
            tol=self.tol, W_init=self.W_init, disp_freq=self.disp_freq,
            silence=self.silence)

        self.W_ = W1
        self.P_ = Ptil
        self.H_ = H
        self.cost_ = cost_list
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_.T)


class PNGE_ADMM(BaseEstimator,TransformerMixin):
    def __init__(self,knn_i,knn_p,r1=5,r2=10,rho=5e2,tau=1e-1,W_init=None,
                 max_iter=5000,tol=1e-4,disp_freq=50000,silence=True):
        """
        knn_i = # within-class kNN neighbors
        knn_p = # betwen-class kNN neighbors
        """
        self.r1 = r1
        self.r2 = r2
        self.knn_i = knn_i
        self.knn_p = knn_p
        self.r  = r1+r2
        self.tau = tau
        self.rho = rho
        self.W_init = W_init
        self.max_iter = max_iter
        self.tol = tol
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        # construct intrinsic and penalty graph
        Li = get_knn_within_class( X,y,n_neighbors=self.knn_i)[1]
        Lp = get_knn_between_class(X,y,n_neighbors=self.knn_p)[1]

        W,P,H,Wtil,Ptil,cost,res = pnge_admm(X.T, r1=self.r1,r2=self.r2,
            L1=Li,L2=Lp, tau=self.tau, rho=self.rho, max_iter=self.max_iter,
            tol=self.tol, W_init=self.W_init, disp_freq=self.disp_freq,
            silence=self.silence)

        self.W_ = Wtil
        self.P_ = Ptil
        self.H_ = H
        self.cost_ = cost
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_[:self.r1].T)

class LS_PNGE_ADMM(BaseEstimator,TransformerMixin,ClassifierMixin):
    def __init__(self,knn_i,knn_p,r1=5,r2=10,rho=5e2,tau=1e-1,gam=1e-1,W_init=None,
                 max_iter=5000,tol=1e-4,disp_freq=50000,silence=True):
        """
        knn_i = # within-class kNN neighbors
        knn_p = # betwen-class kNN neighbors
        """
        self.r1 = r1
        self.r2 = r2
        self.knn_i = knn_i
        self.knn_p = knn_p
        self.r  = r1+r2
        self.tau = tau
        self.gam = gam
        self.rho = rho
        self.W_init = W_init
        self.max_iter = max_iter
        self.tol = tol
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        # construct intrinsic and penalty graph
        Li = get_knn_within_class( X,y,n_neighbors=self.knn_i)[1]
        Lp = get_knn_between_class(X,y,n_neighbors=self.knn_p)[1]

        W,P,H,w,Wtil,Ptil,cost,res = ls_pnge_admm(X.T,y,r1=self.r1,r2=self.r2,
            L1=Li,L2=Lp, tau=self.tau, gam=self.gam,
            rho=self.rho, max_iter=self.max_iter,
            tol=self.tol, W_init=self.W_init, disp_freq=self.disp_freq,
            silence=self.silence)

        self.W_ = Wtil
        self.P_ = Ptil
        self.H_ = H
        self.w_ = w
        self.cost_ = cost
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_[:self.r1].T)


    def decision_function(self,X):
        check_is_fitted(self,'P_')

        H = self.transform(X)

        # apply classifier
        score = H.dot(self.w_)
        return score

    def predict(self,X):
        check_is_fitted(self,'P_')

        # project data to low dimensional space
        H = self.transform(X)

        # apply classifier
        score = H.dot(self.w_)
        ypr = np.sign(score)
        return ypr


class LS_SPNMF_ADMM(BaseEstimator,ClassifierMixin,TransformerMixin):
    def __init__(self,gam=1e-1,gam2=0.,r=10,rho=5e2,constraint='stiefel',W_init=None,
                 max_iter=5000,tol=1e-4,disp_freq=50000,silence=True):
        self.r = r
        self.gam = gam
        self.gam2 = gam2
        self.rho = rho
        self.constraint = constraint
        self.W_init = W_init
        self.max_iter = max_iter
        self.tol = tol
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        W,P,H,w,Wtil,Ptil,cost,res = ls_spnmf_admm(X.T, y, r=self.r,
            gam=self.gam, gam2=self.gam2,
            constraint=self.constraint, rho=self.rho, max_iter=self.max_iter,
            tol=self.tol, W_init=self.W_init, disp_freq=self.disp_freq,
            silence=self.silence)

        self.W_ = Wtil
        self.w_ = w
        self.P_ = Ptil
        self.H_ = H
        self.cost_ = cost
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_.T)


    def decision_function(self,X):
        check_is_fitted(self,'P_')
        # project data to low dimensional space
        H = np.dot(X, self.P_.T)

        # apply classifier
        score = H.dot(self.w_)
        return score

    def predict(self,X):
        check_is_fitted(self,'P_')

        # project data to low dimensional space
        H = np.dot(X, self.P_.T)

        # apply classifier
        score = H.dot(self.w_)
        ypr = np.sign(score)
        return ypr


class LS_SGPNMF_ADMM(BaseEstimator,ClassifierMixin,TransformerMixin):
    """Created 02/04/2016"""
    def __init__(self,r=10,tau=1e-1,gam=1e-1,gam2=0.,rho=5e2,max_iter=5000,
                 tol=1e-4,W_init=None,disp_freq=50000,silence=True,
                 n_neighbors=5, graph_type='knn_within_class'):
        """
        graph_type : {'knn' or 'knn_within_class'}
            (X,y) driven graph construction method
        """
        self.r = r
        self.tau = tau
        self.gam = gam
        self.gam2 = gam2
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.W_init = W_init
        self.disp_freq = disp_freq
        self.silence = silence

        # feature driven graph construction method
        self.n_neighbors = n_neighbors
        self.graph_type = graph_type

    def fit(self,X,y):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        # create within-class KNN graph
        if self.graph_type == 'knn_within_class':
            L = get_knn_within_class(X,y,n_neighbors=self.n_neighbors)[1]
        elif self.graph_type == 'knn':
            L = get_knn_graph(X,n_neighbors=self.n_neighbors)[1]
#        # if L is a tuple or list with two integers, create supervised
#        # MFA graph (knn_withinclass and knn_betweenclass graph)
#        if isinstance(self.L,list) or isinstance(self.L,tuple):
#            # within class intrinsic graph
#            Ls = get_knn_within_class(X,y,n_neighbors=self.L[0])[1]
#
#            # between class penalty graph
#            Lp = get_knn_between_class(X,y,n_neighbors=self.L[1])[1]
#
#            # final laplacian matrix of interest
#            L = Ls - Lp
#        elif self.L.shape[1] == len(y):
#            # else (nxn) laplacian matrix is assumed to be given
#            # (add exception statement later)
#            L = self.L


        W,P,H,w,Wtil,Ptil,cost,res = ls_sgpnmf_admm(X.T, y, r=self.r, L=L,
            tau=self.tau, gam=self.gam, gam2=self.gam2, rho=self.rho,
            max_iter=self.max_iter, tol=self.tol, W_init=self.W_init,
            disp_freq=self.disp_freq, silence=self.silence)

        self.W_ = Wtil
        self.w_ = w
        self.P_ = Ptil
        self.H_ = H
        self.cost_ = cost
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_.T)


    def decision_function(self,X):
        check_is_fitted(self,'P_')
        # project data to low dimensional space
        H = np.dot(X, self.P_.T)

        # apply classifier
        score = H.dot(self.w_)
        return score

    def predict(self,X):
        check_is_fitted(self,'P_')

        # project data to low dimensional space
        H = np.dot(X, self.P_.T)

        # apply classifier
        score = H.dot(self.w_)
        ypr = np.sign(score)
        return ypr




#%%==== helper functions =====
def nnd_svd(X, r, get_H=False):
    """ My wrapper to NND-SVD from nimfa module

    NND-SVD = "Non-Negative Double Singular Value Decomposition", proposed by
    Boutsidis2008.  Since my focus is on PNMF, just return matrix W.

    Parameters
    ----------
    X : ndarray of shape [p,n]
        Data matrix with columns as data points
    r : int
        Dimension of the embedding space
    get_H : bool (default = False)
        Return coefficient matrix too (not needed when using projective NMF)

    Returns
    -------
    W : ndarray of shape [p,r]
        Initialization of basis & projection matrix
    H : (optional) ndarray of shape [r,n]
        Initialization of coefficient matrix.  Returned if ``get_H=True``
    """
    from nimfa.methods.seeding.nndsvd import Nndsvd

    if get_H:
        Winit,Hinit = Nndsvd().initialize(X,rank=r,options=dict(flag=0))
        return np.array(Winit), np.array(Hinit)
    else:
        Winit = Nndsvd().initialize(X,rank=r,options=dict(flag=0))[0]
        # convert to ndarray (instead of matrix form)
        return np.array(Winit)



def prox_hinge(x,tau):
    """ Proximal operator of the hinge loss

    Return the prox operator of the hinge loss, given by:

    .. math::

        \\text{prox}(x,\\tau) =
            \\begin{cases}
                t       & \\text{if}\\quad t > 1 \\\\
                1       & \\text{if}\\quad 1-\\tau \\le t \\le 1 \\\\
                t+\\tau & \\text{if}\\quad t < 1-\\tau
            \\end{cases}

    Parameters
    ----------
    x : array_like
        Input values

    tau : float
        Prox-operator value

    Returns
    ------
    y : ndarray
        The prox-operator of the hinge loss

    Plot
    ----
    >>> x = np.linspace(-4,4,501)
    >>> plt.plot(x, prox(x,tau=2))
    >>> plt.grid('on')
    """
    return x * (x > 1) + 1. * ((1-tau)<=x) * (x<=1) + (x+tau)*(x < (1-tau))

#%% --- graph related ---
def get_subject_graph(df, return_as_df=False):
    """ Get "subject-graph", where S_ij = 1 if (i,j) represents the same subject

    Returns the similarity matrix S and Laplacian matrix L of shape [n,n].
    Usecase for TBI, where we have scans from same subjects.

    Usage
    -----
    >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    >>> S,L = get_subject_graph(df)

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.

    History
    -------
    Created 02/03/2016

    See ``dev_0203_created_subject_graph.py``
    """
    n = df.shape[0]

    #from scipy import sparse
    S = np.zeros((n,n))

    for i in range(n):
        S[i,:] = df['Subject_ID'][i] == df['Subject_ID']

    D = np.diag( S.sum(axis=0) )
    L = D - S

    if return_as_df:
        S = pd.DataFrame(S, index=df['Subject_ID'],columns=df['Subject_ID'])
        L = pd.DataFrame(L, index=df['Subject_ID'],columns=df['Subject_ID'])

    return S,L


def get_knn_kfn(X,n_neighbors):
    """ Get k-nearest and k-farthest neighbors (in Euclidean distance)

    Parameters
    ----------
    X : ndarray of shape [n,p]
        Design matrix
    n_neighbors : int
        Number of neighbors

    Return
    ------
    knn : ndarray of shape [n,n_neighbors]
        K-nearest-neighbors
    kfn : ndarray of shape [n,n_neighbors,]
        K-farthest-neighbors

    Snippets
    --------
    I used this as a sanity check for this function (use Spyder variable
    explorer to compare EDM_sorted and EDM_knn, EDM_kfn below)

    >>> knn,kfn = tw_nmf.get_knn_kfn(X,n_neighbors=5)
    >>> EDM = tw.get_EDM(X)
    >>> EDM_sorted = np.sort(EDM, axis=1)
    >>> EDM_knn = np.zeros(knn.shape)
    >>> EDM_kfn = np.zeros(knn.shape)
    >>> for i in range( EDM_knn.shape[0]):
    >>>     EDM_knn[i] = EDM[i, knn[i]]
    >>>     EDM_kfn[i] = EDM[i, kfn[i]]

    **Created 01/24/2016**
    ----------------------
    See ``0124_knn_and_kfarthest_b.py``

    Updates 02/11/2016

    - replaced np.argsort with tw.argsort to handle "ties"
    - add huge diagonal item to EDM
    """
    from sklearn.metrics.pairwise import pairwise_distances
    n = X.shape[0]

    # add huge item to diagonal to ensure "self" will be farthest
    EDM = pairwise_distances(X,metric='euclidean') + 10e10*np.eye(n)

    #idx_rank  = np.argsort(EDM,axis=1)
    idx_rank = tw.argsort(EDM,axis=1)

    # here ok to start from beginning - i avoided self-similarity above
    knn = idx_rank[:,:n_neighbors]

    # -1 shift since the "farthest" guy is the huge diagonal EDM element
    # I added above
    kfn = idx_rank[:,-n_neighbors-1:-1]
    return knn,kfn



def get_knn(X,n_neighbors):
    """ Get k-nearest neighbors (in Euclidean distance)

    This function is a simple wrapper to ``get_knn_kfn``, which i created first.

    Parameters
    ----------
    X : ndarray of shape [n,p]
        Design matrix
    n_neighbors : int
        Number of neighbors

    Return
    ------
    knn : ndarray of shape [n,n_neighbors]
        K-nearest-neighbors for each samples (each row contains indices of kNN)

    Snippets
    --------
    I used this as a sanity check for this function (use Spyder variable
    explorer to compare EDM_sorted and EDM_knn, EDM_kfn below)

    >>> knn,kfn = tw_nmf.get_knn_kfn(X,n_neighbors=5)
    >>> EDM = tw.get_EDM(X)
    >>> EDM_sorted = np.sort(EDM, axis=1)
    >>> EDM_knn = np.zeros(knn.shape)
    >>> for i in range( EDM_knn.shape[0]):
    >>>     EDM_knn[i] = EDM[i, knn[i]]

    **Created 02/11/2016**
    ----------------------
    """
    knn = get_knn_kfn(X,n_neighbors)[0]
    return knn


def get_kfn(X,n_neighbors):
    """ Get k-farthest neighbors (in Euclidean distance)

    This function is a simple wrapper to ``get_knn_kfn``, which i created first.

    Parameters
    ----------
    X : ndarray of shape [n,p]
        Design matrix
    n_neighbors : int
        Number of neighbors

    Return
    ------
    kfn : ndarray of shape [n,n_neighbors]
        K-farthest-neighbors for each samples (each row contains indices of kFN)

    Snippets
    --------
    I used this as a sanity check for this function (use Spyder variable
    explorer to compare EDM_sorted and EDM_knn, EDM_kfn below)

    >>> knn,kfn = tw_nmf.get_knn_kfn(X,n_neighbors=5)
    >>> EDM = tw.get_EDM(X)
    >>> EDM_sorted = np.sort(EDM, axis=1)
    >>> EDM_knn = np.zeros(knn.shape)
    >>> for i in range( EDM_knn.shape[0]):
    >>>     EDM_knn[i] = EDM[i, knn[i]]

    **Created 02/11/2016**
    ----------------------
    """
    kfn = get_knn_kfn(X,n_neighbors)[1]
    return kfn


def make_graph_from_knn(knn,symm_method='OR'):
    """ Given knn index info, create similarity and adjacency graph

    Parameters
    ----------
    knn : ndarray of shape [n,n_neighbors]
        K-nearest-neighbors indices (can also be k-FARTHEST)
    symm_method : {'OR', 'AND'} (default="OR")
        Symmetrization method (since kNN is not a symmetric relation).
        Logical "AND" or "OR"

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.  For now, binary only.

    Usage
    -----
    >>> knn = tw_nmf.get_knn(scores_exec,n_neighbors=3)
    >>> S,L = tw.make_graph_from_knn(knn)

    History
    --------
    Created 02/11/2016
    """
    n = knn.shape[0]
    S = np.zeros((n,n))
    for i in range(n):
        S[i,knn[i]]=1

    # symmetrize
    if symm_method == 'OR':
        S = ((S + S.T)!=0).astype(float)
    elif symm_method == 'AND':
        S = ((S + S.T)==2).astype(float)

    # create laplacian matrix
    D = np.diag( S.sum(axis=0) )
    L = D - S

    return S,L




def get_knn_kfn_graph(X,n_neighbors):
    knn,kfn = get_knn_kfn(X,n_neighbors)
    n = X.shape[0]

    A_knn = np.zeros((n,n))
    A_kfn = np.zeros((n,n))

    for i in range(n):
        A_knn[i,knn[i]]=1
        A_kfn[i,kfn[i]]=1

    # symmetrize
    A_knn = ((A_knn + A_knn.T)!=0).astype(float)
    A_kfn = ((A_kfn + A_kfn.T)!=0).astype(float)
    return A_knn, A_kfn


def get_knn_graph(X,n_neighbors=3, symm_method = 'OR', get_kfn=False):
    """ Given [n,p] data matrix, return kNN graph (or kFN graph)

    Parameters
    ----------
    X : ndarray of shape [n,p]
        Design matrix
    n_neighbors : int
        Number of neighbors
    symm_method : {'OR', 'AND'} (default="OR")
        Symmetrization method (since kNN is not a symmetric relation).
        Logical "AND" or "OR"
    get_kfn : bool (default=False)
        Get k-"farthest"-neighbor graph (**added 02/11/2016**)

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.  For now, binary only.

    History
    -------
    Created 02/03/2016

    02/11/2016 - Added new argument ``get_kfn`` for k-farthest option.
    Also decided to use new function i created, ``make_graph_from_knn`` to
    construct S,L from knn index info (helps code modularity)
    """
    n = X.shape[0]
    S = np.zeros((n,n))

    #--- get neighbor indices ----#
    if get_kfn:
        # note: abusive variable name
        _,nbr_idx = get_knn_kfn(X,n_neighbors)
    else:
        nbr_idx,_ = get_knn_kfn(X,n_neighbors)

    #--- create binary similarity graph ----#
    S,L = make_graph_from_knn(nbr_idx,symm_method)

    return S,L


def edm_checker(Xfeatures,S):
    """ Can be handy for checking is kNN structure makes sense

    Created 02/03/2016.  See ``t_0203_gpnmf_trial1.py``
    """
    from scipy.stats import rankdata

    # EDM vectorized
    edm = tw.dvec(tw.get_EDM(Xfeatures))

    # study df_edm inside spyder's variable explorer.
    # (you'll see the "top rank" distances are usually "is_neighbor")
    df_edm_checker = pd.DataFrame([edm,rankdata(edm),tw.dvec(S)],
                           index=['dist','rank','is_neighbor']).T
    return df_edm_checker


def edm_checker_embedding(X,H,S):
    """ Checker to see if "similar" sample nearby in embedded space

    X : ndarray of shape = [n_samples,n_features]
        Original feature
    H : ndarray of shape = [n_samples,n_reduced_features]
        Features embedded in lower dimensional space
    S : ndarray of shape = [n_samples,n_samples]
        Similarity matrix.  The nonzero locations indicates pairs where
        EDM values should be small in the embedded space.

    Returns
    -------
    df_edm : dataframe with nchoosek(n_samples,2) rows (all possible pairs)
        Contains ranking info of EDM in original and embedded feature space.
        I used this to view in spyder's explorer as a check to see my
        graph regularizer is working.

    History
    -------
    Created 02/03/2016.  See ``t_0203_gpnmf_trial1.py``
    """
    ##%% ensure subjects that are indicated "similar" by S are indeed nearby in H
    from scipy.stats import rankdata

    # EDM in original space
    edm_raw = tw.dvec(tw.get_EDM(X))

    # EDM in new space
    edm = tw.dvec(tw.get_EDM(H))

    # study df_edm inside spyder's variable explorer.
    # (you'll see the "top rank" distances are usually "is_neighbor"
    df_edm = pd.DataFrame([edm_raw,rankdata(edm_raw),edm,rankdata(edm),tw.dvec(S)],
                           index=['dist_X','rank_X','dist_H','rank_H','is_neighbor']).T
    return df_edm


def get_knn_within_class(X,y,n_neighbors=3,symm_method='OR'):
    """ Get within-class k-nearest-neighbor.

    KNN structure with the restriction that the label of the data pair agree.

    X : ndarray of shape [n_samples,n_feaures]
        Data matrix
    y : label vector of shape [n_samples]
        Binary label vector

    History
    -------
    Created 02/03/2016 - see ``dev_0203_get_within_betweenclass_knn.py``

    Update 02/11/2016

    - replaced np.argsort with tw.argsort to handle "ties"
    - decided to use new function i created, ``make_graph_from_knn`` to
      construct S,L from knn index info (helps code modularity)
    - removed variable ``ranking`` inside code...it doesn't seem to do anything
    """
    n = len(y)

    # set diagonal to large value to avoid self similarity (helps the sorting and ranking)
    EDM = tw.get_EDM(X) + 1e10*np.eye(n)

    #idx_rank = np.argsort(EDM,axis=1)
    idx_rank = tw.argsort(EDM,axis=1)

    knn_within = np.zeros((n, n_neighbors),dtype=int)
    for i in range(n):
        kth_nearest = 0
        for j in range(n):
            # candidate subject to compare
            isub = idx_rank[i,j]

            # if candidate subject label agrees, than add to knn
            if y[i] == y[isub]:
                knn_within[i,kth_nearest] = isub

                # now look for the "next" nearest neighbor
                kth_nearest += 1
                if kth_nearest == n_neighbors: break

    # now we're ready to make knn graph matrices
    S,L = make_graph_from_knn(knn_within,symm_method)

    return S,L,knn_within




def get_knn_between_class(X,y,n_neighbors=3,symm_method='OR'):
    """ Get between-class k-nearest-neighbor.

    KNN structure with the restriction that the label of the data pair disagree

    History
    -------
    Created 02/03/2016 - see ``dev_0203_get_within_betweenclass_knn.py``

    Update 02/11/2016

    - replaced np.argsort with tw.argsort to handle "ties"
    - decided to use new function i created, ``make_graph_from_knn`` to
      construct S,L from knn index info (helps code modularity)
    - removed variable ``ranking`` inside code...it doesn't seem to do anything
    """
    n = len(y)

    # set diagonal to large value to avoid self similarity (helps the sorting and ranking)
    EDM = tw.get_EDM(X) + 1e10*np.eye(n)

    #idx_rank = np.argsort(EDM,axis=1)
    idx_rank = tw.argsort(EDM,axis=1)

    knn_between = np.zeros((n, n_neighbors),dtype=int)
    for i in range(n):
        kth_nearest = 0
        for j in range(n):
            # candidate subject to compare
            isub = idx_rank[i,j]

            # if candidate subject label disagrees, than add to knn
            if y[i] != y[isub]:
                knn_between[i,kth_nearest] = isub

                # now look for the "next" nearest neighbor
                kth_nearest += 1
                if kth_nearest == n_neighbors: break

    # now we're ready to make knn graph matrices
    S,L = make_graph_from_knn(knn_between,symm_method)
    return S,L,knn_between


def get_gose_graph_sorted_y(df,y, n_neighbors=3):
    """ Get GOSE/DRS knn-graph.

    For control subjects, all similarity will have zero values.

    Here assume y is sorted into blocks of TBI and HC.  This way it is easy
    to assign similarity matrix, as it'll have block diagonal structure.

    Parameters
    ----------
    df : dataframe
        >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    y : array-like of shape = (n_subjects,)
        Vector of binary labels.  Must be sorted in blocks of +1 and -1
    n_neighbors : int
        Number of neighbors to select in kNN (ie, value of k)

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.  For now, binary only.
        S_{i,j} will be 1 if subjects i and j are both TBI, and the
        Euclidean distance in [GOSE,DRS] are among kNN.
    """
    n = len(y)

    df_tbi = df.query('y_dx == +1')[['GOSE','DRS']]
    S_tbi = get_knn_graph(df_tbi.values,n_neighbors=n_neighbors)[0]

    S = np.zeros((n,n))
    S[np.ix_( df_tbi.index, df_tbi.index)] = S_tbi

    L = np.diag(S.sum(axis=0)) - S
    return S, L

def get_gose_graph(df,y,n_neighbors=3):
    """ Get GOSE/DRS knn-graph, where y can be shuffled in any arbitrary order.

    For control subjects, all similarity will have zero values.

    Here assume y is sorted into blocks of TBI and HC.  This way it is easy
    to assign similarity matrix, as it'll have block diagonal structure.

    Parameters
    ----------
    df : dataframe
        >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    y : array-like of shape = (n_subjects,)
        Vector of binary labels
    n_neighbors : int
        Number of neighbors to select in kNN (ie, value of k)

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.  For now, binary only.
        S_{i,j} will be 1 if subjects i and j are both TBI, and the
        Euclidean distance in [GOSE,DRS] are among kNN.
    """
    #--- first sort TBI/HC subjects ---#
    """numpy argsort will return index-order of "ties" in seemingly arbitrary
       order....i want the "ties" to be returned in the order of the original
       occurence (for example, if y[10],y[15],y[21] are tied, i want the
       idx_sort to give me (10,15,21), not random permutation of it.
       Use rankdata for this.
       http://stackoverflow.com/questions/14671013/
       ranking-of-numpy-array-with-possible-duplicates
    """
    #idx_sort = np.argsort(y) #<== don't do this!
    idx_sort = sp.stats.rankdata(y,'ordinal').argsort()
    rev_sort = np.argsort(idx_sort) # <- to "undo" sorting
    #http://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python

    df_sort = df.ix[idx_sort,:].reset_index(drop=True)
    y_sort  = y[idx_sort]

    #--- now we can use the "sorted" gose/drs graph construction ---#
    S = get_gose_graph_sorted_y(df_sort,y_sort,n_neighbors)[0]

    # undo the sorting
    S = S[np.ix_(rev_sort,rev_sort)]

    # Laplacian graph
    L = np.diag(S.sum(axis=0)) - S
    return S,L



def get_subject_constrained_gose_knn(df,y,n_neighbors=10,sanity_check=False):
    """ Get gose/drs knn graph, with the constraint that if S_ij cannot be
    the same subject

    Parameters
    ----------
    df : dataframe
        >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    y : array-like of shape = (n_subjects,)
        Vector of binary labels
    n_neighbors : int
        Number of neighbors to select in kNN (ie, value of k)
    sanity_check : bool (default=False)
        Return extra "sanity-check" items

        >>> # sanity_check = False
        >>> knn_gose, knn_gose_subj = get_subject_constrained_gose_knn(df,y)
        >>> # sanity_check = True
        >>> knn_gose, knn_gose_subj,df_EDM,df_EDM_sorted,df_EDM_knn = ...

    Returns
    -------
    knn_gose : pd.DataFrame of of shape=(n_tbi,n_neibors)
        knn structure, where each row indicates a subject, and col-values
        contains integer indices of kNN for that subject.
    knn_gose_subj : pd.DataFrame of of shape=(n_tbi,n_neibors)
        Contains string of Subject_ID (good sanity check for myself that
        the same subject is not connected via knn)

    Extra returns if sanity_check is on
    -----------------------------------
    Bit tedious to explain in words....
    Run the snippet code above, and explore them in spyder variable explorer,
    and you'll get what i mean...

    History
    -------
    Created 02/11/2016 - see ``dev_0211_create_constrained_gose_graph.py``
    """
    n = len(y)
    n_tbi = (y==+1).sum()

    df_tbi = df.query('y_dx==1').reset_index()
    id_list = df_tbi['Subject_ID'].tolist()

    # set diagonal to large value to avoid self similarity (helps the sorting and ranking)
    EDM = tw.get_EDM(df_tbi[['GOSE','DRS']].values) + 1e10*np.eye(n_tbi)

    #--- create subject constrained gose knn graph ---#
    knn_gose = np.zeros((n_tbi, n_neighbors),dtype=int)
    knn_gose_subj = np.zeros((n_tbi,n_neighbors),dtype=object)
    idx_rank = tw.argsort(EDM,axis=1)

    for i in range(n_tbi):
        kth_nearest = 0
        id_i = id_list[i]
        for j in range(n):
            # candidate subject to compare
            isub = idx_rank[i,j]

            # if indices (i,j) are NOT the same subject, than add to knn
            if id_i != id_list[isub]:
                knn_gose[i,kth_nearest] = isub
                #knn_gose_subj[i,kth_nearest] = id_list[isub]
                knn_gose_subj[i,kth_nearest] = df_tbi['ID'].tolist()[isub]

                # now look for the "next" nearest neighbor
                kth_nearest += 1
                if kth_nearest == n_neighbors:
                    break

    # convert to dataframe so i can view in spyder variable explorer
    knn_gose      = pd.DataFrame(knn_gose, index=df_tbi['ID'])
    knn_gose_subj = pd.DataFrame(knn_gose_subj, index=df_tbi['ID'])

    if not sanity_check:
        return knn_gose, knn_gose_subj

    #---- for sanity check, get sorted EDM with distance and subject-id -----#
    idx_sort = tw.argsort(EDM,axis=1)
    df_EDM_sorted = pd.DataFrame(index=df_tbi['ID'],columns=range(n_tbi))
    for i in range(n_tbi):
        for j in range(n_tbi):
            _dist = EDM[i, idx_sort[i,j]]
            _id   = df_tbi['ID'].tolist()[idx_sort[i,j]]
            df_EDM_sorted.ix[i,j] = (_id,_dist)

    #--- now get constrained knn info, of distance and subject-id ---#
    #--- make sure no same-subject are not connected              ---#
    df_EDM_knn= pd.DataFrame(index=df_tbi['ID'], columns=range(n_neighbors))
    for i in range(n_tbi):
        for j in range(n_neighbors):
            idx = knn_gose.ix[i,j]
            _dist = EDM[i,idx]
            _id   = df_tbi['ID'].tolist()[idx]
            df_EDM_knn.ix[i,j] = (_id, _dist)

    df_EDM = pd.DataFrame(EDM,index=df_tbi['ID'],columns=df_tbi['ID'])

    # Tranpose makes it easier to view in spyder variable explorer
    df_EDM_sorted = df_EDM_sorted.T
    df_EDM_knn    = df_EDM_knn.T

    return knn_gose, knn_gose_subj,df_EDM,df_EDM_sorted,df_EDM_knn


def get_subject_constrained_gose_graph(df,y,n_neighbors=10):
    """ Get gose/drs knn graph, with the constraint that if S_ij cannot be
    the same subject

    Note that df and y can be permuted in any order, which is important
    when running cross-validation.

    Parameters
    ----------
    df : dataframe
        >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    y : array-like of shape = (n_subjects,)
        Vector of binary labels
    n_neighbors : int
        Number of neighbors to select in kNN (ie, value of k)

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.  For now, binary only.
        S_{i,j} will be 1 if subjects i and j are both TBI, and the
        Euclidean distance in [GOSE,DRS] are among kNN.

    History
    -------
    Created 02/11/2016 - see ``dev_0211_create_constrained_gose_graph.py``
    """
    n = len(y)

    #| note that since i'm using rankdata.argsort(), all these sorting and
    #| revsorting will not do anything  if y are already sorted, which is
    #| nice since i don't have to create and maintain a separate function
    idx_sort = sp.stats.rankdata(y,'ordinal').argsort()
    rev_sort = np.argsort(idx_sort) # <- to "undo" sorting
    #http://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python

    df_sort = df.ix[idx_sort,:].reset_index(drop=True)
    y_sort  = y[idx_sort]

    #--- now we can use the "sorted" gose/drs graph construction ---#
    knn_gose = get_subject_constrained_gose_knn(df_sort,y_sort,n_neighbors)[0]
    S_tbi = make_graph_from_knn(knn_gose.values)[0]
    S = np.zeros((n,n))

    # thanks to the sorting, this S will have block diagonal structure
    S[np.ix_( y_sort==+1,y_sort==+1)] = S_tbi

    #==== undo the sorting ====#
    S = S[np.ix_(rev_sort,rev_sort)]

    # Laplacian graph
    L = np.diag(S.sum(axis=0)) - S
    return S,L


def get_gose_kfn(df,n_neighbors=10):
    """ Get gose/drs kfn graph.

    Parameters
    ----------
    df : dataframe
        >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    y : array-like of shape = (n_subjects,)
        Vector of binary labels
    n_neighbors : int
        Number of neighbors to select in kFN (ie, value of k)

    Returns
    -------
    kfn_gose : pd.DataFrame of of shape=(n_tbi,n_neibors)
        kfn structure, where each row indicates a subject, and col-values
        contains integer indices of kNN for that subject.
    kfn_gose_subj : pd.DataFrame of of shape=(n_tbi,n_neibors)
        Contains string of Subject_ID (good for sanity check with ``df_EDM``)
    df_EDM : pd.DataFrame of of shape=(n_neighbors,n_neighbors)
        Col/Index = subject id

    History
    -------
    Created 02/11/2016 - see ``dev_0211_create_kfn_gose_graph.py``
    """
    df_tbi = df.query('y_dx==1').reset_index()
    n_tbi = df_tbi.shape[0]

    ##### set diagonal to large value to avoid self similarity (helps the sorting and ranking)
    # no, don't do this....i want the last n_neighbor columns to be the "farthest"
    EDM = tw.get_EDM(df_tbi[['GOSE','DRS']].values)# + 1e10*np.eye(n_tbi)

    #--- create subject constrained gose knn graph ---#
    kfn_gose = get_kfn(df_tbi[['GOSE','DRS']].values,n_neighbors)
    kfn_gose = kfn_gose[:,::-1] # reverse sort so "farthest" appears in the first column
    kfn_gose_subj = np.zeros((n_tbi,n_neighbors),dtype=object)
    for i in range(n_tbi):
        for j in range(n_neighbors):
            idx = kfn_gose[i,j]
            kfn_gose_subj[i,j] = df_tbi['ID'].tolist()[idx]

    # convert to dataframe so i can view in spyder variable explorer
    kfn_gose      = pd.DataFrame(kfn_gose, index=df_tbi['ID'])
    kfn_gose_subj = pd.DataFrame(kfn_gose_subj, index=df_tbi['ID']).T

    df_EDM = pd.DataFrame(EDM,index=df_tbi['ID'],columns=df_tbi['ID'])
    EDM = pd.DataFrame(EDM)
    return kfn_gose,kfn_gose_subj,df_EDM, EDM


def get_gose_graph_kfn(df,y,n_neighbors=10):
    """ Get gose/drs kfn graph

    Note that df and y can be permuted in any order, which is important
    when running cross-validation.

    Parameters
    ----------
    df : dataframe
        >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    y : array-like of shape = (n_subjects,)
        Vector of binary labels
    n_neighbors : int
        Number of neighbors to select in kFN (ie, value of k)

    Returns
    -------
    S, L : ndarray of shape = [n,n]
        Similarity and Laplacian matrix, respectively.  For now, binary only.
        S_{i,j} will be 1 if subjects i and j are both TBI, and the
        Euclidean distance in [GOSE,DRS] are among kFN.

    TODO
    -----
    HUGE overlap with ``get_subject_constrained_gose_graph``....the crux is
    sorting tbi subjects into cluster, than inverse sorting.  Try to merge
    two functions together if I find i may need to create another variant
    of this idea.

    History
    -------
    Created 02/11/2016 - see ``dev_0211_create_kfn_gose_graph.py``
    """
    n = len(y)

    #| note that since i'm using rankdata.argsort(), all these sorting and
    #| revsorting will not do anything  if y are already sorted, which is
    #| nice since i don't have to create and maintain a separate function
    idx_sort = sp.stats.rankdata(y,'ordinal').argsort()
    rev_sort = np.argsort(idx_sort) # <- to "undo" sorting
    #http://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python

    df_sort = df.ix[idx_sort,:].reset_index(drop=True)
    y_sort  = y[idx_sort]

    #--- now we can use the "sorted" gose/drs graph construction ---#
    kfn_gose = get_gose_kfn(df_sort,n_neighbors)[0]
    S_tbi = make_graph_from_knn(kfn_gose.values)[0]
    S = np.zeros((n,n))

    # thanks to the sorting, this S will have block diagonal structure
    S[np.ix_( y_sort==+1,y_sort==+1)] = S_tbi

    #==== undo the sorting ====#
    S = S[np.ix_(rev_sort,rev_sort)]

    # Laplacian graph
    L = np.diag(S.sum(axis=0)) - S
    return S,L
#%%----

def normalize_W(W):
    """ Normalize the columns of W to unit norm

    W : ndarray of shape [p,r]
        Typically the basis matrix returned from NMF
    """
    Wnorm = np.zeros(W.shape)
    from scipy.linalg import norm

    for i in range(W.shape[1]):
        Wnorm[:,i] = W[:,i] / norm(W[:,i])

    return Wnorm


def disp_corrW(W):
    """ Display correlation matrix of basis matrix W returned from NMF

    Useful to see how "orthogonal" the learnt basis are.
    """
    Wnorm = normalize_W(W)

    tw.figure('f')
    plt.subplot(131),tw.imtak(W.T.dot(W)), plt.colorbar(orientation='horizontal')
    plt.subplot(132),tw.imtak(Wnorm.T.dot(Wnorm)), plt.colorbar(orientation='horizontal')
    plt.gci().set_clim((0,1))
    plt.subplot(133),tw.imtak(np.corrcoef(W.T)), plt.colorbar(orientation='horizontal')
    plt.gci().set_clim((0,1))
#    plt.colorbar()


#%% **** FAILURES ****
def hinge_pnmf_admm(X, y, r, rho=1e2,lam=1e-1,gam=1e-1,max_iter=5000,tol=1e-4,
                    W_init=None,disp_freq=2000,silence=False):
    """ Supervised Projective NMF with hinge-loss (solved via ADMM)

    >>> W,P,H,w,Wtil,Ptil,wtil,cost,res = pnmf_admm(X,y,r=10,rho=1e2)

    Parameters
    ----------
    X : ndarray of shape =[n_features, n_samples]=[p,n]
        Data matrix with columns as data points
    y : array-like, shape = [n_samples]
        Binary Target vector relative to X (must be +/- 1)
    r : int
        Dimension of the embedding space (n_components)
    lam : float
        Regularization parameter (amount of emphasis on hinge-loss)
    gam : float
        Regularization parameter (classification regularizer)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None (default), rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **Created 02/02/2016**
    """
    p,n = X.shape
    lam *= 1. # ensure float
    gam *= 1. # ensure float
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # classification weight vector
    w = np.zeros(r)

    # aux. variables
    Wtil = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)
    wtil = np.zeros(n)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    d_w   = np.zeros(n)

#    Y = np.diag(y)
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_recon_list = []
    cost_supervised_list = []
#    cost = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2 + \
#           lam*np.maximum(0, 1-y*dot(H.T,w)).sum() + norm(gam)**2/2
#    cost = norm( X - dot(W,P.dot(X)), 'fro')**2/2

    cost_recon = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    cost_recon_list.append(cost_recon)

    cost_supervised = lam*np.maximum(0, 1-y*dot(H.T,w)).sum() + norm(gam)**2/2
    cost_supervised_list.append(cost_supervised)

    cost = cost_recon + cost_supervised
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    # keep track of relative primal residual
    res_W = []
    res_P = []
    res_H = []
    res_w = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)

    rho_W = rho*50000000

    start=time.time()
    for iter in range(max_iter):
#        rho = min(1e6, rho*1.0001)
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...


        # (W,H) updates
        W = solve(dot(H,H.T)+rho_W*eye_r, dot(H, X.T) + rho_W*Wtil.T - Lam_W.T).T
        Wtil = (W+Lam_W/rho_W).clip(0)

        H = solve(dot(W.T,W)+rho*(eye_r+np.outer(w,w)),
                  dot(W.T, X) - Lam_H - np.outer(w, (d_w*y)) +
                  rho*(dot(P,X) + rho*np.outer(w, (wtil*y))) )

        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (w,wtil) updates
        w = solve( H.dot(H.T) + gam*np.eye(r), H.dot( -y*d_w + rho*wtil) ) / rho
        wtil = prox_hinge( y*dot(H.T,w) + d_w/rho, lam/rho )

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho_W*(W - Wtil)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        d_w = d_w - rho*(wtil - y*dot(H.T,w))

        #============ comptute objective values ==============================#
        # main loss function
        cost_recon = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        cost_recon_list.append(cost_recon)

        cost_supervised = lam*np.maximum(0, 1-y*dot(H.T,-w)).sum() + norm(gam)**2/2
        cost_supervised_list.append(cost_supervised)

        cost = cost_recon + cost_supervised
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
#        res_w.append(norm(wtil - y*dot(H.T,w))/norm(wtil))
        res_w.append(0)

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_w[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>-1) and (iter % disp_freq == 0):
#            print rho,
#            rho = min(1e6, rho*1.05)
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " costU={:6.5e}".format(cost_recon_list[iter+1])
            str_ += " costS={:6.5e}".format(cost_supervised_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " w={:3.2e}".format(res_w[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    res = pd.DataFrame([res_W,res_P,res_H,res_w,diffcost],
                       index=['res_W','res_P','res_H','res_w','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost_list,res


def hinge_spnmf_admm(X, y, r, rho=1e2,lam=1e-1,gam=1e-1,max_iter=5000,tol=1e-4,
                    W_init=None,disp_freq=2000,silence=False):
    """ Supervised Projective NMF with hinge-loss (solved via ADMM)

    >>> W,P,H,w,Wtil,Ptil,wtil,cost,res = pnmf_admm(X,y,r=10,rho=1e2)

    Parameters
    ----------
    X : ndarray of shape =[n_features, n_samples]=[p,n]
        Data matrix with columns as data points
    y : array-like, shape = [n_samples]
        Binary Target vector relative to X (must be +/- 1)
    r : int
        Dimension of the embedding space (n_components)
    lam : float
        Regularization parameter (amount of emphasis on hinge-loss)
    gam : float
        Regularization parameter (classification regularizer)
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None (default), rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    Update history
    --------------
    **Created 02/02/2016**
    """
    p,n = X.shape
    lam *= 1. # ensure float
    gam *= 1. # ensure float
    #%%=== initialize variables =====
    # define main primal variable W (basis matrix)
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

#    P = np.zeros((r,p))
    P = np.random.rand(r,p)

    # classification weight vector
    w = np.zeros(r)

    # aux. variables
    Wtil = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)
    wtil = np.zeros(n)
    wtil = np.random.randn(n)

    # dual variables
    Lam_W = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)
    d_w   = np.zeros(n)

#    Y = np.diag(y)
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_recon_list = []
    cost_supervised_list = []
#    cost = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2 + \
#           lam*np.maximum(0, 1-y*dot(H.T,w)).sum() + norm(gam)**2/2
#    cost = norm( X - dot(W,P.dot(X)), 'fro')**2/2

    cost_recon = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
    cost_recon_list.append(cost_recon)

    cost_supervised = lam*np.maximum(0, 1-y*dot(H.T,w)).sum() + norm(gam)**2/2
    cost_supervised_list.append(cost_supervised)

    cost = cost_recon + cost_supervised
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W = []
    res_W2 = []
    res_P = []
    res_H = []
    res_w = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)
    rho_W = rho*5e5
    rho_W2 = rho*5e5
    start=time.time()
    for iter in range(max_iter):
#        rho = min(1e6, rho*1.0001)
#        W_old = W #<- to keep track of diffW
        # (W,H) updates
        W = solve(dot(H,H.T)+(rho_W+rho)*eye_r,
                  dot(H, X.T) + rho_W*Wtil.T - Lam_W.T - Lam_W2.T + rho*W2.T).T
        Wtil = (W+Lam_W/rho_W).clip(0)
        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)


        W2 = projection_spectral(W+Lam_W2/rho)

        H = solve(dot(W.T,W)+rho*(eye_r+np.outer(w,w)),
                  dot(W.T, X) - Lam_H - np.outer(w, (d_w*y)) +
                  rho*(dot(P,X) + rho*np.outer(w, (wtil*y))) )

        # (w,wtil) updates
        w = solve( H.dot(H.T) + gam*np.eye(r), H.dot( -y*d_w + rho*wtil) ) / rho
        wtil = prox_hinge( y*dot(H.T,w) + d_w/rho, lam/rho )

        #====================== dual updates =================================#
        Lam_W = Lam_W + rho_W*(W - Wtil)
        Lam_W2 = Lam_W2 - rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))
        d_w = d_w - rho*(wtil - y*dot(H.T,w))

        #============ comptute objective values ==============================#
        # main loss function
        cost_recon = norm(Wtil.dot(Ptil.dot(X)) - X,'fro')**2/2
        cost_recon_list.append(cost_recon)

        cost_supervised = lam*np.maximum(0, 1-y*dot(H.T,-w)).sum() + norm(gam)**2/2
        cost_supervised_list.append(cost_supervised)

        cost = cost_recon + cost_supervised
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W.append(norm(W-Wtil,'fro')/norm(W,'fro'))
        res_W2.append(norm(W2-Wtil,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))
        res_w.append(norm(wtil - y*dot(H.T,w))/norm(wtil))
#        res_w.append(0)

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check ====#
        check_exit  = (diffcost[iter] < tol)
        check_exit &= (res_W[iter] <  tol)
        check_exit &= (res_P[iter] <  tol)
        check_exit &= (res_H[iter] <  tol)
        check_exit &= (res_w[iter] <  tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>-1) and (iter % disp_freq == 0):
#            print rho,
#            rho = min(1e6, rho*1.05)
#            str_ = "iter={:4} cost={:3.3f}".format(iter,cost_list[iter+1])
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost_list[iter+1])
            str_ += " costU={:6.5e}".format(cost_recon_list[iter+1])
            str_ += " costS={:6.5e}".format(cost_supervised_list[iter+1])
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W={:3.2e}".format(res_W[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
            str_ += " w={:3.2e}".format(res_w[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    res = pd.DataFrame([res_W,res_P,res_H,res_w,diffcost],
                       index=['res_W','res_P','res_H','res_w','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,w,Wtil,Ptil,cost_list,res


#%%=== label constraint method that didn't turnout so useful ====
def lc_spnmf_admm(X,y,r=5,alpha=1e-1,constraint='stiefel',rho=1e3,max_iter=5000,
                  tol=1e-4,W_init='nndsvd',disp_freq=1000,silence=False):
    """ Label constrained Spectral Projective NMF using ADMM.

    **LC** Idea from 2011 Z. Jiang (CVPR) - Learning a discriminative
    dictionary for sparse coding via label constrained k-svd

    Usage
    -----
    >>> W,P,H,Wtil,Ptil,A,cost,res = lc_spnmf_admm(....)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int (must be even for this LC-approach)
        Dimension of the embedding space (n_components)
    alpha : float
        Regularization for discriminative code
    constraint : ``'stiefel'`` or ``'convex'``
        The type of spectral constraint (default: ``'stiefel'``).  This
        determines the type of spectral projection applied.
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd' (default)
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    History
    --------------
    **02/08/2016** - function created (forked from ``spnmf_admm``)
    """
    p,n = X.shape
    alpha *= 1.0
    rho *= 1.0
    if r%2 != 0:
        raise ValueError("r must be an even integer")
    #%%=== initialize variables =====
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    W1 = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W1 = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)

    #--- variables for LC approach ---#
    # linear transformation matrix that parametrizes LC
    A = np.zeros((r,r))

    # get discriminative codes
    Q = get_discriminative_codes(y,r)
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_LC_list = []

    cost_nmf = norm(W1.dot(Ptil.dot(X)) - X,'fro')**2/2
    cost_LC = alpha*norm( Q - A.dot(H), 'fro')**2/2
    cost = cost_nmf + cost_LC

    cost_nmf_list.append(cost_nmf)
    cost_LC_list.append(cost_LC)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W1 = []
    res_W2 = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)



    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+rho*eye_r + alpha*dot(A.T,A),
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + alpha*dot(A.T,Q) )
        A = solve( H.dot(H.T), H.dot(Q.T)).T
        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*(W1+W2).T - Lam_W1.T - Lam_W2.T).T

        # projections
        W1 = (W+Lam_W1/rho).clip(0)
        W2 = projection_spectral(W+Lam_W2/rho,projection_type=constraint)

        #====================== dual updates =================================#
        Lam_W1 = Lam_W1 + rho*(W - W1)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        cost_nmf = norm(W1.dot(Ptil.dot(X)) - X,'fro')**2/2
        cost_LC = alpha*norm( Q - A.dot(H), 'fro')**2/2
        cost = cost_nmf + cost_LC

        cost_nmf_list.append(cost_nmf)
        cost_LC_list.append(cost_LC)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W1.append(norm(W-W1,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check =============#
        check_exit =  (diffcost[iter] < tol)
        check_exit &= (  res_W1[iter] < tol)
        check_exit &= (  res_W2[iter] < tol)
        check_exit &= (   res_P[iter] < tol)
        check_exit &= (   res_H[iter] < tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:.2e}".format(cost_nmf)
            str_ += " LC={:.2e}".format(cost_LC)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W1={:3.2e}".format(res_W1[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
#    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_LC_list],
                        index=['total','nmf','LC']).T
    res = pd.DataFrame([res_W1,res_W2,res_P,res_H,diffcost],
                       index=['res_W1','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,W1,Ptil,A,cost,res





def lslc_spnmf_admm(X,y,r=5,alpha=1e-1,gam=1e-1,constraint='stiefel',rho=1e3,
                    max_iter=5000,
                  tol=1e-4,W_init='nndsvd',disp_freq=1000,silence=False):
    """ Least-squares Label constrained Spectral Projective NMF using ADMM.

    **LC** Idea from 2011 Z. Jiang (CVPR) - Learning a discriminative
    dictionary for sparse coding via label constrained k-svd

    Usage
    -----
    >>> W,P,H,Wtil,Ptil,A,w,cost,res = lslc_spnmf_admm(....)

    Parameters
    ----------
    X : ndarray of shape (n_features, n_samples)=(p,n)
        Data matrix with columns as data points
    y : array-like of shape [n_samples,]
        Label vector
    r : int (must be even for this LC-approach)
        Dimension of the embedding space (n_components)
    alpha : float
        Regularization for discriminative code
    gam : float
        Regularization on the least-squares fit
    constraint : ``'stiefel'`` or ``'convex'``
        The type of spectral constraint (default: ``'stiefel'``).  This
        determines the type of spectral projection applied.
    rho : float
        Augmented Lagrangian parameter
    max_iter : int
        max number of iterations
    tol : float
        convergence tolerance (on relative change in objective value and
        all primal residuals)
    W_init : None, rng, or 'nndsvd'
        Way to initialize W (and H).  If ``None`` or ``RandomState object``
        is supplied, both W and H will be initialized via non-negative
        random number.  If ``'nndsvd'``, will use deterministic initialization
        NNDSVD "Non-Negative Double Singular Value Decomposition", proposed by
        Boutsidis2008.
    disp_freq : int
        Frequency of update display (in iterations)
    silence : bool
        If True, don't print-out convergence detail at the end.

    History
    --------------
    **02/08/2016** - function created (forked from ``lc_spnmf_admm``)
    """
    p,n = X.shape
    alpha *= 1.0
    rho *= 1.0
    if r%2 != 0:
        raise ValueError("r must be an even integer")
    #%%=== initialize variables =====
    if W_init is None:
        W = np.random.rand(p,r)
        W /= norm(W,2)

        H = np.random.rand(r,n)
        H /= norm(H,2)
    elif isinstance(W_init, np.random.RandomState):
        rng = W_init

        W = rng.rand(p,r)
        W /= norm(W,2)

        H = rng.rand(r,n)
        H /= norm(H,2)
    elif W_init == 'nndsvd':
#        W = nnd_svd(X,r)
        W,H = nnd_svd(X,r,get_H=True)
    else:
        raise ValueError('Invalid input for "W_init"')

    P = np.zeros((r,p))
#    P = np.random.rand(r,p)

    # aux. variables
    W1 = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    Ptil = np.zeros(P.shape)

    # dual variables
    Lam_W1 = np.zeros(W.shape)
    Lam_W2 = np.zeros(W.shape)
    Lam_P = np.zeros(P.shape)
    Lam_H = np.zeros(H.shape)

    #--- variables for LC approach ---#
    # linear transformation matrix that parametrizes LC
    A = np.zeros((r,r))

    # get discriminative codes
    Q = get_discriminative_codes(y,r)

    # classifier
    w = np.zeros(r)
    #%%=== create empty lists for tracking updates ===
    # keep track of frobenius norm loss
    cost_list = []
    cost_nmf_list = []
    cost_LC_list = []
    cost_ls_list = []

    PX = Ptil.dot(X)
    cost_nmf = norm(W1.dot(PX) - X,'fro')**2/2
    cost_LC = alpha*norm( Q - A.dot(H), 'fro')**2/2
    cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2
    cost = cost_nmf + cost_LC + cost_ls

    cost_nmf_list.append(cost_nmf)
    cost_LC_list.append(cost_LC)
    cost_ls_list.append(cost_ls)
    cost_list.append(cost)

    # relative change in cost
    diffcost = []

    # keep track of relative primal residual
    res_W1 = []
    res_W2 = []
    res_P = []
    res_H = []

    # also keep track of diffW
#    diffW = []
    #%%=== run admm iterations ===
    eye_r = np.eye(r)

    # precompute term needed for inversion lemma
    K = solve(np.eye(n) + dot(X.T,X), X.T)



    start=time.time()
    for iter in range(max_iter):
#        W_old = W #<- to keep track of diffW

        #================ primal updates (P, W, H, Ptil, Wtil) ===============#
        # update for P involves inversion lemma...
        _R = dot(X,Lam_H.T)/rho - Lam_P.T/rho + dot(X,H.T) + Ptil.T
        P = (_R - X.dot( dot(K,_R) )).T
        Ptil = (P+Lam_P/rho).clip(0)

        # (W,H) updates
        H = solve(dot(W.T,W)+rho*eye_r + alpha*dot(A.T,A),
                  dot(W.T, X) + rho*dot(P,X) - Lam_H + alpha*dot(A.T,Q) )
        A = solve( H.dot(H.T), H.dot(Q.T)).T
        w = solve( H.dot(H.T), H.dot(y))
        W = solve(dot(H,H.T)+(2*rho)*eye_r,
                  dot(H, X.T) + rho*(W1+W2).T - Lam_W1.T - Lam_W2.T).T

        # projections
        W1 = (W+Lam_W1/rho).clip(0)
        W2 = projection_spectral(W+Lam_W2/rho,projection_type=constraint)

        #====================== dual updates =================================#
        Lam_W1 = Lam_W1 + rho*(W - W1)
        Lam_W2 = Lam_W2 + rho*(W - W2)
        Lam_P = Lam_P + rho*(P - Ptil)
        Lam_H = Lam_H + rho*(H - dot(P,X))

        #============ comptute objective values ==============================#
        # main loss function
        PX = Ptil.dot(X)
        cost_nmf = norm(W1.dot(PX) - X,'fro')**2/2
        cost_LC = alpha*norm( Q - A.dot(H), 'fro')**2/2
        cost_ls  = gam*norm(y - PX.T.dot(w)) **2/2
        cost = cost_nmf + cost_LC + cost_ls

        cost_nmf_list.append(cost_nmf)
        cost_LC_list.append(cost_LC)
        cost_ls_list.append(cost_ls)
        cost_list.append(cost)

        # relative change in cost
        diffcost.append(np.abs(cost_list[iter+1] - cost_list[iter])/cost_list[iter])

        # relative primal residuals
        res_W1.append(norm(W-W1,'fro')/norm(W,'fro'))
        res_W2.append(norm(W-W2,'fro')/norm(W,'fro'))
        res_P.append(norm(P-Ptil,'fro')/norm(P,'fro'))
        res_H.append(norm(H-P.dot(X),'fro')/norm(H,'fro'))

        # relative change in basis matrix
#        diffW.append( norm(W-W_old,'fro')/ norm(W_old, 'fro') )

        #=== termination check =============#
        check_exit =  (diffcost[iter] < tol)
        check_exit &= (  res_W1[iter] < tol)
        check_exit &= (  res_W2[iter] < tol)
        check_exit &= (   res_P[iter] < tol)
        check_exit &= (   res_H[iter] < tol)
        if check_exit and iter > 300: # allow 300 iterations of "burn-in"
            if not silence:
                print "Termination condition met.  Exit at iter =",iter,
            break
        #=============== print current progress ==============================#
#        if iter % disp_freq == 0:
        if (iter>1) and (iter % disp_freq == 0):
            str_ = "iter={:4} cost={:6.5e}".format(iter,cost)
            str_ += " nmf={:.2e}".format(cost_nmf)
            str_ += " LC={:.2e}".format(cost_LC)
            str_ += " LS={:.2e}".format(cost_ls)
            str_ += " diffcost={:3.2e}".format(diffcost[iter])
            str_ += " W1={:3.2e}".format(res_W1[iter])
            str_ += " W2={:3.2e}".format(res_W2[iter])
            str_ += " P={:3.2e}".format(res_P[iter])
            str_ += " H={:3.2e}".format(res_H[iter])
#            str_ += " diffW={:3.2e}".format(diffW[iter])
            print str_ + " ({:3.1f} sec)".format(time.time()-start)
    #%%---- end of admm for loop ---------------------------------------------#
#    cost_list = np.array(cost_list)

    # convert residual in pandas dataframe
    cost = pd.DataFrame([cost_list, cost_nmf_list, cost_LC_list,cost_ls_list],
                        index=['total','nmf','LC','cost_ls']).T
    res = pd.DataFrame([res_W1,res_W2,res_P,res_H,diffcost],
                       index=['res_W1','res_W2','res_P','res_H','diffcost']).T

    if not silence: print_time(start)
    return W,P,H,W1,Ptil,A,w,cost,res

class LC_SPNMF_ADMM(BaseEstimator,TransformerMixin):
    """ Label constrained Spectral Projective NMF using ADMM.

    History
    --------------
    **02/08/2016** - function created (forked from ``spnmf_admm``)
    """
    def __init__(self,r=5,alpha=1e-1,constraint='stiefel',rho=1e3,
                 max_iter=5000,tol=1e-4,W_init='nndsvd',
                 disp_freq=50000,silence=True):
        if r%2 != 0:
            raise ValueError("r must be an even integer")
        self.r = r
        self.alpha = alpha
        self.constraint = constraint
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.W_init = W_init
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        W,P,H,Wtil,Ptil,A,cost,res = lc_spnmf_admm(X.T, y, r=self.r,
            alpha=self.alpha,
            constraint=self.constraint, rho=self.rho, max_iter=self.max_iter,
            tol=self.tol, W_init=self.W_init, disp_freq=self.disp_freq,
            silence=self.silence)

        self.W_ = Wtil
        self.P_ = Ptil
        self.H_ = H
        self.A_ = A
        self.cost_ = cost
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_.T)


class LSLC_SPNMF_ADMM(BaseEstimator,TransformerMixin,ClassifierMixin):
    """ Least squares Label constrained Spectral Projective NMF using ADMM.

    History
    --------------
    **02/08/2016** - function created (forked from ``LC_SPNMF_ADMM``)
    """
    def __init__(self,r=5,alpha=1e-1,gam=1e-1,constraint='stiefel',rho=1e3,
                 max_iter=5000,tol=1e-4,W_init='nndsvd',
                 disp_freq=50000,silence=True):
        if r%2 != 0:
            raise ValueError("r must be an even integer")
        self.r = r
        self.alpha = alpha
        self.gam = gam
        self.constraint = constraint
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.W_init = W_init
        self.disp_freq = disp_freq
        self.silence = silence

    def fit(self,X,y):
        """WARNING: Here, to conform with scikit's convention, X is [n,p]
        shaped, so need to apply transpose before inputting to function"""
        W,P,H,Wtil,Ptil,A,w,cost,res = lslc_spnmf_admm(X.T, y, r=self.r,
            alpha=self.alpha, gam=self.gam,
            constraint=self.constraint, rho=self.rho, max_iter=self.max_iter,
            tol=self.tol, W_init=self.W_init, disp_freq=self.disp_freq,
            silence=self.silence)

        self.W_ = Wtil
        self.P_ = Ptil
        self.H_ = H
        self.A_ = A
        self.w_ = w
        self.cost_ = cost
        self.resid_ = res
        return self

    def transform(self,X,y=None):
        """WARNING: Here, to conform with scikit's convention, X is [n,p] shaped"""
        check_is_fitted(self,'P_')
        return np.dot(X, self.P_.T)

    def decision_function(self,X):
        check_is_fitted(self,'P_')
        # project data to low dimensional space
        H = np.dot(X, self.P_.T)

        # apply classifier
        score = H.dot(self.w_)
        return score

    def predict(self,X):
        check_is_fitted(self,'P_')

        # project data to low dimensional space
        H = np.dot(X, self.P_.T)

        # apply classifier
        score = H.dot(self.w_)
        ypr = np.sign(score)
        return ypr


def get_discriminative_codes(y,r):
    """ Get codes for **Label-constraint**

    Parameters
    ----------
    y : array-like of shape [n_samples,]
        label vector
    r : int divisible by 2
        Embedding space. For simplicity, I'm going to let the number of
        dimension for +1 and -1 both be the same at r/2

    Returns
    -------
    Q : array-like of shape

    Idea from 2011 Z. Jiang (CVPR) - Learning a discriminative dictionary
    for sparse coding via label constrained k-svd

    Created 02/08/2016
    """
    if r%2 != 0:
        raise ValueError("r must be an even integer")

    n = len(y)
    Q = np.zeros((r,n))
    q = np.zeros(r)
    qp = q.copy()
    qm = q.copy()
    qp[:r/2] = 1
    qm[r/2:] = 1
    for i in range( len(y)):
        if y[i]==+1:
            Q[:,i] = qp
        elif y[i]==-1:
            Q[:,i] = qm
    return Q


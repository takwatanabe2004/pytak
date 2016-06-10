"""
===============================================================================
This will contain my "final" version utility function.
===============================================================================
"""

import numpy as np
import scipy as sp
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import time
import inspect
from pprint import pprint
import warnings
import sklearn
from sklearn.utils import deprecated

#from visualize import tw_slicer
#%% === platform dependent functions (cell began 2016-01-23) ===
from socket import gethostname
hostname = gethostname()

#--- global file param for screensize ---#

# screensize i like on my office computer monitor setup
#def get_screen_info():
#    """ Get screen info (Created 2016-01-23)
#    """
#    if hostname == 'takanori-PC':
#        _L = -1440. # left-position
#        _T  = 22. # top-position
#        _W = 1440. # width at fullscreen
#        _H =878.   # height at fullscreen
#    return (_L,_T,_W,_H)
if hostname == 'takanori-PC':
    # on my home 27in computer (used tw.fig_get_geom())
    """" (x,y, width, height), x=0 means very left, y=0 means very top"""
    _L = -1440. # left-position
    _T  = 22. # top-position
    _W = 1440. # width at fullscreen
    _H =878.   # height at fullscreen

    _screen_default = (_L,        _T,   640,  640)
    _screen_full    = (_L,        _T,    _W,   _H)
    _screen_left    = (_L,        _T,  _W/2,   _H)
    _screen_right   = (_L+_W/2,   _T,  _W/2,   _H)

    # the +/-10 is to avoid mild overlap between "t" and "b"
    _screen_top     = (_L,        _T,  _W,  _H/2-10)
    _screen_bottom  = (_L, _T+_H/2+10,  _W,  _H/2-10)

    _screen_bl      = (_L,      _T+_H/2+10,  _W/2,  _H/2-10) # bottom-left
    _screen_br      = (_L+_W/2, _T+_H/2+10,  _W/2,  _H/2-10) # bottom-right
    _screen_tl      = (_L,      _T,         _W/2,  _H/2-10) # top-left
    _screen_tr      = (_L+_W/2, _T,         _W/2,  _H/2-10) # top-right
else:
    # on my sbia computer monitor
    _screen_default = (1500,     25,  640,  640)
    _screen_full    = (1500,     25, 1280,  924)
    _screen_left    = (1500,     25,  640,  924)
    _screen_right   = (2240,     25,  640,  924)
    _screen_top     = (1500,     25, 1280,  462)
    _screen_bottom  = (1500, 25+462, 1280,  462)
    _screen_bl      = (1500, 25+462,  640,  462) # bottom-left
    _screen_br      = (2240, 25+462,  640,  462) # bottom-right
    _screen_tl      = (1500,     25,  640,  462) # top-left
    _screen_tr      = (2240,     25,  640,  462) # top-right
#%%*** 0224/2016 :the following has been migrated from my nmf "util" script ***
#--- feature selection routines ---
def get_pvalue_ranksum(X,scores,impute = False,corr_type = 'pearson'):
    """ Get sum of ranking of pvalues between connectomes and clinical scores.

    Given data matrix and clinical scores, obtaining a ranking of features
    according to the **rank-sum** of correlation pvalues.  Used for feature
    selection.

    Usage
    -----
    Note: I sort the features selected for consistency (inverse fs-transform
    may get flaky (ie prone to bug) if I don't follow this rule

    >>> idx_ranking, df_ranking, dict_summary = get_pvalue_ranksum(X,scores)
    >>> idx_fs = np.sort(idx_ranking[:500])  # select top-500 features
    >>> X_fs = X[:,idx_fs]                   # apply feature selection

    We can get the "inverse-transform" operator as follow
    (this is why I like to apply the ``sort`` above for consistency)

    >>> # get "inverse-transform" operator
    >>> recon = lambda x: inverse_fs_transform(x, idx_fs, n_feat=3655)
    >>> X_recon = recon(X_fs) # X_recon has zeros in the columns
    >>>                       # where feature selection took place

    Parameters
    ----------
    X : ndarray of shape [n_subjects,n_features]
        Data matrix representing connectivity
    scores : pandas DataFrame of shape [n_subjects, n_scores]
        Clinical score matrix
    impute : bool (Default=False)
        Impute NAN scores with median value (default=False)
    corr_type : {'pearson', 'spearman','kendall'}
        Type of correlation

    Returns
    -------
    idx_rank : ndarray vector of shape [n_features,]
        Feature indices sorted from "most correlated" to "least correlated"
        with clinical scores (according to rank-sum of correlation pvalues).
        Used for feature selection (eg, ``X_fs = X[:,idx_rank[:100]]`` will
        take the top 100 features)
    df_ranking : pandas DataFrame of shape [n_features, 2+n_scores]
        DataFrame containing the final ``overall_ranking``, ``rank_sum`` of
        pvalues, and the individual pvalue rank for each clincal scores.
        **Note**: ``idx_rank`` is just the ``np.argsort`` of
        ``overall_ranking``
    dict_summary : dictionary
        Dictionary containing pvalue information.  I may drop this in the
        future (I only included this for sanity check)

    Details
    -------
    Given [n_subjects,n_edges] data matrix X and [n_subjects, num_scores]
    matrix of clinical scores, do the following:

    1. Compute univariate correlation between individual edges and clinical
       scores, and obtain ``pval_mat``, which is an [n_edges, num_scores]
       shaped matrix of correlation pvalues
    2. Rank the pvalues for each clinical score
       (results in an [n_edges, num_scores] matrix I call ``ranking``, which
       will correspond to the last 11 columns in the output ``df_ranking``)
    3. Obtain ``rank_sum``, which is a [n_edges,] shaped vector representing
       the sum of the ranking of pvalues across all scores.  I interpret this
       as a global metric representing how well correlated each individual
       edge is with **all** the clinical scores
    4. Obtain ``idx_rank``, a [n_edges,] shaped vector of

        - ``idx_rank[0]`` : index of the edge most correlated with scores (ie,
          edge with the smallest rank-sum, or most correlated with scores)
        - ``idx_rank[1]`` : index of the edge 2nd most correlated with scores
        - ``idx_rank[2]`` : index of the edge 3rd most correlated with scores
        - ...

    History
    --------
    - created 02/01/2016
    - updated 02/23/2016: added option ``corr_type`` with ``'pearson'`` as
      default (for backward compatibility)
    """
    n,p = X.shape

    #--- (optional) impute missing scores with median values
    if impute:
        from sklearn.preprocessing import Imputer
        imputer=Imputer(missing_values='NaN', strategy='median',axis=0)
        scores_imputed = imputer.fit_transform(scores.values)
        scores_imputed = pd.DataFrame(scores_imputed,columns=scores.columns)
        pval_mat = cross_corr_dropnans(X,scores_imputed,corr_type)[1]
    else:
        pval_mat = cross_corr_dropnans(X,scores,corr_type)[1]

    # get a ranking of pvalues
    pval_ranking = np.argsort(pval_mat,axis=0)
    pval_mat_sorted = np.zeros(pval_mat.shape)
    for i in range(pval_mat.shape[1]):
        idx_sort = pval_ranking[:,i]
        pval_mat_sorted[:,i] = pval_mat[idx_sort,i]

    dict_summary = {'pval_mat':pval_mat,
                    'pval_ranking':pval_ranking,
                    'pval_mat_sorted':pval_mat_sorted}

    from scipy.stats import rankdata
    ranking = np.zeros(pval_mat.shape,dtype=int)
    for i in range(pval_mat.shape[1]):
        ranking[:,i] = rankdata(pval_mat[:,i])

    rank_sum = ranking.sum(axis=1)

    idx_ranking = np.argsort(rank_sum)
    overall_ranking = rankdata(rank_sum)

    #--- create dataFrame of summary of ranking info ---#
    df_ranking=np.hstack( (overall_ranking[:,None], rank_sum[:,None],ranking ))
    df_ranking = pd.DataFrame(df_ranking)
    df_ranking.columns = ['overall_ranking','rank_sum'] + scores.columns.tolist()
    return idx_ranking, df_ranking, dict_summary


def inverse_fs_transform(x_fs, idx_fs, n_feat):
    """ Get reconstruction matrix for **inverse_transform** of feature selection

    Parameters
    ----------
    x_fs : ndarray of shape [n_feat_selected,]
        Input in the "feature-selection space".  Can be either a vector of
        shape [n_feat_selected,] or "design-matrix" form of shape
        [n_samples, n_feat_selected]
    idx_fs : ndarray of integers of shape [n_feat_selected,]
        Indices of feature to select
    n_feat : int
        Original number of features prior to feature selection

    Returns
    -------
    x_recon : ndarray
        ``x_fs`` back in the original feature space, with zeros inserted at
        indices that have been removed via feature-selection ``idx_fs``

    Usage
    -----
    Note: I sort the features selected for consistency (inverse fs-transform
    may get flaky (ie prone to bug) if I don't follow this rule

    >>> idx_ranking, df_ranking, dict_summary = get_pvalue_ranksum(X,scores)
    >>> idx_fs = np.sort(idx_ranking[:500])  # select top-500 features
    >>> X_fs = X[:,idx_fs]                   # apply feature selection

    We can get the "inverse-transform" operator as follow
    (this is why I like to apply the ``sort`` above for consistency)

    >>> # get "inverse-transform" operator
    >>> recon = lambda x: inverse_fs_transform(x, idx_fs, n_feat=3655)
    >>> X_recon = recon(X_fs) # X_recon has zeros in the columns
    >>>                       # where feature selection took place
    """
    from scipy import sparse

    # reconstruction matrix....using sparse matrix (just makes it easier for me)
    A = sparse.eye(n_feat,format='csr')[:,idx_fs]

    # apply inverse transform
    if len(x_fs.shape) == 1:
        x_recon = A.dot(x_fs) # vector of shape [n_feat,]
    elif len(x_fs.shape) == 2:
        x_recon = A.dot(x_fs.T).T # matrix of shape [n_samples, n_feat]
    else:
        raise RuntimeError('Input x must be either 1d or 2d array')
    return x_recon


def get_fs_pvalue_ranksum_indices(X,scores,k=500,impute=False,corr_type='pearson'):
    """ My "quick" routine for getting feature selection

    Usage
    -----
    >>> X,y,df = twio.get_tbi_connectomes(return_all_scores=True)
    >>> X /= X.max(axis=0)
    >>> X[np.isnan(X) | np.isinf(X)] = 0
    >>>
    >>> scores = df.ix[:,-11:]
    >>> idx_fs, idx_fs_vec, inv_fs = util.get_fs_pvalue_ranksum_indices(X,scores,k=1000)
    >>>
    >>> X_fs = X[:,idx_fs]

    History
    --------
    - updated 02/23/2016: added option ``corr_type`` with ``'pearson'`` as
      default (for backward compatibility)
    """
    n,p = X.shape
    idx_ranking = get_pvalue_ranksum(X,scores,impute,corr_type)[0]

    # sort the index for consistency
    idx_fs = np.sort(idx_ranking[:k])  # select top-k features

    idx_fs_vec = np.zeros(p,dtype=int)
    idx_fs_vec[idx_fs] = 1

    # "inverse-transform" operator to "undo" feature selection
    inv_fs = lambda x: inverse_fs_transform(x, idx_fs, n_feat=p)

    return idx_fs, idx_fs_vec, inv_fs


def plt_ttest_boxplot_H(H,y,threshold=0.05):
    """ Plot boxplots of NMF embedding coefficinets across two populations

    Update
    ------
    - 02/05/2016: ylim adjusted to ignore outlier
    """
    r = H.shape[1]

    tstats,pval,_ = ttest_fdr_corrected(H,y,alpha=0.05)
    idx_rej = (pval < threshold)

    def plt_get_nrow_ncol(r):
        if r < 7:               nrow,ncol = 2,3
        elif r in range(7,11):  nrow,ncol = 2,5
        elif r in range(11,13):  nrow,ncol = 3,4
        elif r in range(13,16): nrow,ncol = 3,5
        elif r in range(16,21): nrow,ncol = 4,5
        elif r in range(21,26): nrow,ncol = 5,5
        elif r in range(26,37): nrow,ncol = 6,6
        elif r in range(37,49): nrow,ncol = 7,8
        else:                   nrow,ncol = 8,8
        return nrow,ncol
    nrow,ncol = plt_get_nrow_ncol(r)

    figure('f')
    #fig,ax = plt.subplots(4,4)
    #ax.ravel()
    for idx in range(r):
    #    if idx > 4:
    #        break
    #    if 1:#idx_rej[idx]:
        n1 = (y==+1).sum()
        n2 = (y==-1).sum()
        xx1 = np.ones(n1)
        xx2 = 2*np.ones(n2)

        yy1 = H[y==+1,idx]
        yy2 = H[y==-1,idx]
        plt.subplot(nrow,ncol,idx+1)
#        plt.boxplot([yy1,yy2],positions=[1,2])
        # http://stackoverflow.com/questions/22028064/matplotlib-boxplot-without-outliers
        plt.boxplot([yy1,yy2],positions=[1,2],showfliers=False)
        plt.xlim((0,3))
        ylim = plt.gca().get_ylim()

        # overlay scatter-plot
        plt.scatter(xx1,yy1,color='r',marker='x') # disease group (+1)
        plt.scatter(xx2,yy2,color='b',marker='x') # control group (-1)
        plt.xticks([])
    #    plt.xlim((-1,4))
        plt.gca().set_ylim(ylim) # undo ylim change from scatterplot

        plt.title("t={:3.2e} (pval={:3.2e})".format(tstats[idx],pval[idx]),
                  fontsize=10,fontweight='bold')
        if idx_rej[idx]:
            plt.gca().set_axis_bgcolor('#ccffcc')



def plt_scatter_2group(H,y,w=None):
    """
    Update 02/16/2016: added 3rd argument w, which represents hyperplane
    """
    r = H.shape[1]
    if r==2:
        #%%
        figure()
        plt.scatter( H[y==+1,0], H[y==+1,1],color='r')
        plt.scatter( H[y==-1,0], H[y==-1,1],color='b')
        if w is not None:
            """
            http://scikit-learn.org/stable/auto_examples/svm/
                plot_separating_hyperplane.html
            http://stackoverflow.com/questions/10953997/
                python-scikits-learn-separating-hyperplane-equation
            """
            min_x,max_x = H[:,0].min(), H[:,0].max()
#            min_y,max_y = H[:,1].min(), H[:,1].max()
#            xx = np.linspace(min_x,max_x,101)
            xx = np.linspace(0,11,101)
#            yy = np.linspace(min_y,max_y,101)
#            XX,YY = np.meshgrid(xx,yy)
            if len(w) == r+1:
                b = -w[2]/w[0]
                a = -w[0]/w[1]
                z = a*xx - b
            else:
                z = -w[0]/w[1]*xx
            plt.plot(xx,z)
#            plt.xlim(min_x,max_x)
            plt.xlim((0,9))
            #%%

    if r==3:
        from mpl_toolkits.mplot3d import Axes3D
        figure()
        ax = plt.gcf().add_subplot(111, projection='3d')
        ax.scatter(H[y==+1,0], H[y==+1,1],H[y==+1,2], c='r')
        ax.scatter(H[y==-1,0], H[y==-1,1],H[y==-1,2], c='b')
        plt.show()

#%% ==== unsorted/uncategorized stuffs =====


def clf_get_auc(ytrue,score):
    """ Computer AUC for binary classification.

    Parameters
    ----------
    ytrue, score : array-like of shape=(n_subjects,)
        True label and classification scores.  ypred assumed to be +/- 1.


    Created 02/15/2016

    Updated 02/18/2016
    """
    #| somehow sklearn.metrics attribute no longer exist...02/18/2016
    #fpr,tpr,thresholds = sklearn.metrics.roc_curve(ytrue,score)
    from sklearn.metrics import roc_curve
    fpr,tpr,thresholds = roc_curve(ytrue,score)
    auc = sklearn.metrics.roc_auc_score(ytrue,score)
    return auc

def clf_get_bsr(ytrue,ypred):
    """ Computer balanced score/accuracy rate for binary classification.

    Parameters
    ----------
    ytrue, ypred : array-like of shape=(n_subjects,)
        True label and predicted label.  Assumed to be +/- 1.


    BSR = 0.5 (TP/P + TN/N) = (recall+specificity)/2

    Created 02/15/2016
    """
    P = (ytrue==+1).sum()
    N = (ytrue==-1).sum()

    # 'p' for positive, 'n' for negative
    idxp = (ytrue==+1).nonzero()[0]
    idxn = (ytrue==-1).nonzero()[0]

    # TP/TN = true  positives/negatives
    # FP/FN = false positives/negatives
    TP = np.sum(ytrue[idxp] == ypred[idxp])
    TN = np.sum(ytrue[idxn] == ypred[idxn])
    #return P,TP,N,TN
    #FP = np.sum(ytrue[idxp] != ypred[idxp])
    #FN = np.sum(ytrue[idxn] != ypred[idxn])

    # balanced score/accuracy rate
    bsr = (1.*TP/P + 1.*TN/N)/2
    return bsr

def clf_results(ytrue,score):
    """ Compute multiple classification performance metrics.

    Here I rely on sklearn's function, just as a sanity check with my other
    function ``clf_summary``

    Parameters
    ----------
    ytrue : array-like of shape=(n_subjects,)
        True label and classification scores.  Assumed to be +/- 1.
    score : array-like of shape=(n_subjects,)
        Classification score.  Function will still work with +/-1 binary
        prediction, but AUC will be meaningless in such case.

    Created 02/16/2016

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    """
    auc = clf_get_auc(ytrue,score)

    #| convert score to +/-1 binary prediction
    ypred = np.sign(score)
    bsr = clf_get_bsr(ytrue,ypred)
    acc = sklearn.metrics.accuracy_score(ytrue,ypred)
    f1  = sklearn.metrics.f1_score(ytrue,ypred)
    precision  = sklearn.metrics.precision_score(ytrue,ypred)
    recall  = sklearn.metrics.recall_score(ytrue,ypred)

    return pd.DataFrame([acc,precision,recall,f1,bsr,auc],
                   index=['acc','prec','recall','f1','bsr','auc']).T


def clf_results_extended(ytrue, score, get_ppv=False):
    """ Compute multiple classification performance metrics.

    Code forked from function ``clf_summary``, but updated the code to my taste
    (I may soon deprecate that function)

    Parameters
    ----------
    ytrue : array-like of shape=(n_subjects,)
        True label and classification scores.  Assumed to be +/- 1.
    score : array-like of shape=(n_subjects,)
        Classification score.  Function will still work with +/-1 binary
        prediction, but AUC will be meaningless in such case.

    Created 02/17/2016

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    """
    ypred = np.sign(score)
    #%%
    # 'p' for positive, 'n' for negative
    idx_ytrue_p = np.where(ytrue == +1)[0]
    idx_ytrue_n = np.where(ytrue == -1)[0]
    idx_ypred_p = np.where(ypred == +1)[0]
    idx_ypred_n = np.where(ypred == -1)[0]

    correct = (ytrue == ypred)
    mistake = (ytrue != ypred)

    # TP = true positives
    # TN = true negatives
    # FP = false positives
    # FN = false negatives
    TP = correct[idx_ytrue_p].sum()
    TN = correct[idx_ytrue_n].sum()
    FP = mistake[idx_ypred_p].sum()
    FN = mistake[idx_ypred_n].sum()

    """ define a function to handle division by zero annoyance """
    def div_handle(num, den):
        """ num = numerator, den = denominator"""
        try:
            # 1.* to convert to float for division
            val = 1.*num/den
        except ZeroDivisionError:
            val = np.nan
        return val

    # TPR = true positive rate  (aka sensitivity, recall, hit rate, power)
    # TNR = true negative rate  (aka specificity)
    # FPR = false positive rate (aka size, type I error rate)
    # FNR = false negative rate (aka miss rate, type II error rate, 1-TPR)
    # PPV = positive predictive value (aka precision)
    # NPV = negative predictive value
    TPR = div_handle(TP,TP+FN)
    TNR = div_handle(TN,TN+FP)
    FPR = div_handle(FP,TP+TN)
    FNR = div_handle(FN,FN+TP)
    PPV = div_handle(TP,TP+FP)
    NPV = div_handle(TN,TN+FN)
    F1  = div_handle(2.*TP, 2*TP+FP+FN)
    AUC = clf_get_auc(ytrue,score)
    BSR = clf_get_bsr(ytrue,ypred)

    ACC = 1.*np.sum(ytrue == ypred)/len(ytrue)

    #summary     = [ ACC,  TPR,  TNR,  FPR,  FNR,  AUC,  BSR,  F1,  PPV,  NPV]
    #score_type  = ['ACC','TPR','TNR','FPR','FNR','AUC','BSR','F1','PPV','NPV']
    if get_ppv:
        summary     = [ ACC,  TPR,  TNR,  AUC,  BSR,  F1,  PPV,  NPV]
        score_type  = ['ACC','TPR','TNR','AUC','BSR','F1','PPV','NPV']
    else:
        summary     = [ ACC,  TPR,  TNR,  AUC,  BSR]
        score_type  = ['ACC','TPR','TNR','AUC','BSR']
    summary = pd.DataFrame(summary, index=score_type).T
#    # let's try multi-index
#    if full:
#        summary     = [ACC, TPR, TNR, FPR, FNR, F1, PPV, NPV, TP, TN, FP, FN, TP+FP, TN+FN, len(ytrue)]
#        score_type  = ['ACC','TPR','TNR','FPR','FNR','F1','PPV','NPV','TP','TN','FP','FN','P','N','ALL']
#    else:
#        summary     = [ACC, TPR, TNR, FPR, FNR, F1, PPV, NPV]
#        score_type  = ['ACC','TPR','TNR','FPR','FNR','F1','PPV','NPV']

#    if multiIndex and full:
#        """http://pandas.pydata.org/pandas-docs/stable/advanced.html"""
#        #| currently have 11 items on this lsit
#        # summary = [ACC, TPR, TNR, TP, TN, FP, FN, FPR, FNR, PPV, NPV]
#        arrays = [np.array(['scores']*8+['counts']*7),
#                  np.array(score_type)]
#        summary = pd.DataFrame(pd.Series(summary, index=arrays),columns = ['value'])
#    else:
#        summary = pd.Series(data=summary, index=score_type, name='clf')
#        summary = pd.DataFrame(summary)

    # (11/01/2015) values like TPR, FNR may be NaN (division by zero), so replace with zero
    summary = summary.fillna(0)
    return summary


def reset():
    """Created 02/11/2016"""
    try:
        from IPython import get_ipython
        ipython=get_ipython()
        ipython.magic('reset -fs')
        plt.close('all')
    except Exception as inst:
        print type(inst) # the exception instance
        print "ipython function not loaded (ru running in python?)"

def get_ipython():
    """Created 04/08/2016

    More used to remind my self about IPython module
    Usage

    >>> ipython = tw.get_ipython()
    >>> ipython.magic('reset -fs')
    """
    import IPython
    return IPython.get_ipython()

def argsort(X,axis=None):
    """ My argsort, where "ties" are sorted according by order of occurence.

    Parameters
    ----------
    X : array_like
        array to sort
    axis : {None,0,1} [default=None]
        Axis along which to sort (only 0 and 1 is supported, for 2d matrix)
        If None, the flattened array is used.

    Sanity check snippets
    --------------------
    Sanity check on how "ties" are handled...comparison with numpy

    >>> y1 = np.random.permutation(y)
    >>> y2 = np.random.permutation(y)
    >>> y3 = np.random.permutation(y)
    >>> Y = np.vstack((y1,y2,y3)).T
    >>> idx_sort1 = np.argsort(Y,axis=0)
    >>> idx_sort2 = tw.argsort(Y,axis=0)
    >>> Y_sort1 = np.zeros(Y.shape)
    >>> Y_sort2 = np.zeros(Y.shape)
    >>> for i in range(Y.shape[1]):
    >>>     Y_sort1[:,i] = Y[idx_sort1[:,i], i]
    >>>     Y_sort2[:,i] = Y[idx_sort2[:,i], i]

    Sanity check that when there's no tie, this will be identical to numpy

    >>> XX = np.random.randn(500,100)
    >>> idx_sort1 = np.argsort(XX,axis=0)
    >>> idx_sort2 = tw.argsort(XX,axis=0)
    >>> print np.array_equal(idx_sort1,idx_sort2)
    >>> idx_sort1 = np.argsort(XX,axis=1)
    >>> idx_sort2 = tw.argsort(XX,axis=1)
    >>> print np.array_equal(idx_sort1,idx_sort2)
    >>> idx_sort1 = np.argsort(XX,axis=None)
    >>> idx_sort2 = tw.argsort(XX,axis=None)
    >>> print np.array_equal(idx_sort1,idx_sort2)

    History
    -------
    Created 02/11/2016
    """
    if axis is None:
        return sp.stats.rankdata(X,'ordinal').argsort()

    idx_argsort = np.zeros(X.shape,dtype=int)
    if axis==0:
        for i in range(X.shape[1]):
            idx_argsort[:,i] = sp.stats.rankdata(X[:,i],'ordinal').argsort()
    elif axis==1:
        for i in range(X.shape[0]):
            idx_argsort[i,:] = sp.stats.rankdata(X[i,:],'ordinal').argsort()
    return idx_argsort

def get_EDM(X):
    """ Simple wrapper to get EDM matrix.

    X : ndarray of shape [n,p]
        Data matrix

    EDM : ndarray of shape = [n,n]
        Euclidean distance matrix

    Created 02/03/2016
    """
    from sklearn.metrics.pairwise import pairwise_distances
    EDM=pairwise_distances(X,metric='euclidean')
    return EDM

def kde_1d(signal,x_grid=None):
    """ Return 1d kde of a vector signal (Created 01/24/2015)

    Todo: how are the kde's normalized?  (i want the kde to sum to 1....)

    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html

    Usage
    -----
    >>> x = np.linspace(0,1,401)
    >>> kde = tw.kde_1d(signal, x)
    >>> plt.plot(x, kde)
    >>> plt.grid('on')
    """
#    from scipy.stats.kde import gaussian_kde
#    if x is None:
#        x = np.linspace(0,1,401)
#
#    return gaussian_kde(signal)(x)
    from statsmodels.nonparametric.kde import KDEUnivariate
    kde = KDEUnivariate(signal)
    kde.fit()
    if x_grid is None:
        x_grid = np.linspace(0,1,401)
    #bin_space = x_grid[1]-x_grid[0]

    # kde estimate
    kde_est = kde.evaluate(x_grid)

    # normalize to pdf (need to come back on this....multiply by bin-spacing??)
    kde_est /= kde_est.sum()

    return kde_est, x_grid


def normalized_conn(x):
    """Edge normalize connectome x

    Added 12/17/2015

    x = [p,] vector of connectome that can be reshaped into [d,d] connectome matrix
    """
    W = sqform(x)
    d = W.shape[0]
    W_normalized = np.zeros( (d,d) )
    deg = W.sum(axis=0)
    for i in range(d):
        for j in range(i+1,d):
            W_normalized[i,j] = W[i,j] / (deg[i]*deg[j])
            W_normalized[j,i] = W_normalized[i,j]

    # normalize so max is 1
    W_normalized /= W_normalized.max()

    # set nans to 0
    W_normalized = np.nan_to_num(W_normalized)
    return dvec(W_normalized) # <- use dvec instead of sqform on purpose here

#tmp = normalized_conn( X[0] )


def normalized_connmat(X_design):
    """Apply normalization to all row-vectors in Xdesign

    Added 12/17/2015
    """
    n,p = X_design.shape
    X_norm = np.zeros( (n,p) )
    for i in range(n):
        X_norm[i] = normalized_conn( X_design[i] )
    return X_norm


def get_train_test_validation_index(n, ntrain, nval, ntest):
    """Added 12/15/2015"""
    assert n == (ntrain+nval+ntest)
    idx = np.random.permutation(n)
    idx_tr = np.sort( idx[:ntrain] )
    idx_vl = np.sort( idx[ntrain:ntrain+nval] )
    idx_ts =  np.sort( idx[ntrain+nval:] )

    df_index = pd.DataFrame(np.zeros((n,3)), columns=['train','validation','test'])
    for i in range(n):
        if i in idx_tr:
            df_index.iloc[i,0]=1
        elif i in idx_vl:
            df_index.iloc[i,1]=1
        elif i in idx_ts:
            df_index.iloc[i,2]=1
    return idx_tr, idx_vl, idx_ts, df_index

#def get_train_test_validation_index_df(n, ntrain, nval, ntest):
#    """Added 12/15/2015"""
#    assert n == (ntrain+nval+ntest)
#    idx_tr, idx_val, idx_ts = get_train_test_validation_index(n, ntrain, nval, ntest)
#
#    df_index = pd.DataFrame(np.zeros((n,3)), columns=['train','validation','test'])
#    for i in range(n):
#        if i in idx_tr:
#            df_index.iloc[i,0]=1
#        elif i in idx_val:
#            df_index.iloc[i,1]=1
#        elif i in idx_ts:
#            df_index.iloc[i,2]=1
#    return df_index

def get_incmat_conn86(radius=50, return_coord=False):
    """Create adjacency matrix using scikit **(Created 12/05/2015)**
    """
    from sklearn.neighbors import radius_neighbors_graph

    # 3d coordinates of nodes
    xyz = get_mni_coord86()

    n_nodes = 86
    n_edges = 86*85/2

    #--- create [n_feat, 6] array indicating 6d coordinates of the edges ---
    cnt=0
    coord = np.zeros((n_edges,6))
    for i in range(n_nodes):
        xyz1 = xyz[i,:]
        for j in range(i+1,n_nodes):
            xyz2 = xyz[j,:]
            coord[cnt,:] = np.concatenate([xyz1,xyz2])
            cnt+=1

    # adjacency matrix
    A = radius_neighbors_graph(coord,radius=radius,metric='euclidean',include_self=False)

    # incidence matrix
    C = adj2inc(A)

    if return_coord:
        return C, A, coord
    else:
        return C, A


def unmask(w, mask):
    """Unmask an image into whole brain, with off-mask voxels set to 0.

    Parameters
    ----------
    w : ndarray, shape (n_features,)
      The image to be unmasked.

    mask : ndarray, shape (nx, ny, nz)
      The mask used in the unmasking operation. It is required that
      mask.sum() == n_features.

    Returns
    -------
    out : 3d of same shape as `mask`.
        The unmasked version of `w`
    """
    data = np.zeros(mask.shape)
    if mask.dtype != np.bool:
        mask.astype(bool)
    if np.count_nonzero(mask) != w.shape[0]:
        raise RuntimeError('Mask nnz should equal feature size!')
    data[mask.nonzero()] = w
    return data


def sort_list_str_case_insensitive(list_var):
    """ Sort list of string with case insensitivity (creatd 11/20/2015)

    Simple, but I often forget the syntax

    http://stackoverflow.com/questions/10269701/
    case-insensitive-list-sorting-without-lowercasing-the-result

    Internal
    --------
    >>> return sorted(list_var, key=lambda s: s.lower())
    """
    return sorted(list_var, key=lambda s: s.lower())

def keyboard(banner=None):
    import code
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return

def ttest_twosample_fixnan(X,y):
    """Two sample ttest, with possible NaNs taken care of (created 11/20/2015)

    Scipy's ``sp.stats.ttest_ind`` is nice, but often yields NAN values.
    It's scary since I often presume with my analysis incognizant of the
    presence of NANs (eg, my custom gui screwed up, and it took me a long time
    that the presence of NAN in an input array was causing this).

    Here, ``tstats=NAN`` will be replaced by 0, and ``pval=NAN`` will be replaced
    by 0 (ie, least significant as possible)

    Parameters
    -----------
    X : ndarray of shape [n,p]
        Design matrix
    y : ndarray of [n,]
        Binary label vector

    Returns
    -------
    tstats, pval : ndarray of shape [n,]
        "Cleaned" tstats and pvalues
    """
    from scipy.stats import ttest_ind
    tstats, pval = ttest_ind( X[y==+1,:], X[y==-1,:] )

    # presents of nans will often screw things up (like my hemiplot)...make a wrapper to clean this?
    idx_nan = np.isnan(pval).nonzero()[0]
    pval[idx_nan] = 1 # for pval, set nan to 1 (least significant)

    idx_nan = np.isnan(tstats).nonzero()[0]
    tstats[idx_nan] = 0
    return tstats, pval

def ttest_twosample_test(X,y,alpha=0.05,return_as_df=False):
    """ Two-sample ttest (nan corrected) (created 01/26/2016)

    Returns test result

    Parameters
    ----------
    X : ndarray of shape [n,p]
        Data matrix (samples as row vectors)
    y : ndarray of shape [n,]
        Label vector
    return_as_df : bool (default=False)
        Return result as [n,3] shaped pandas DataFrame

    Returns
    -------
    tstats : ndarray of shape [n,]
        Vector of tstats
    pval : ndarray of shape [n,]
        pvalues
    idx_rejected : array, bool (shape [n,])
        True if a hypothesis is rejected, False if not
    """
    tstats, pval = ttest_twosample_fixnan(X,y)
    idx_rejected = (pval<alpha).astype(int)

    if return_as_df:
        df = pd.DataFrame([tstats,pval, idx_rejected],
                          index=['tstats','pval','rejected']).T
        return df
    else:
        return tstats, pval, idx_rejected

def ttest_fdr_corrected(X,y,alpha=0.05,return_as_df=False):
    """ FDR corrected ttest pvalue (created 11/20/2015)

    http://statsmodels.sourceforge.net/devel/generated/
    statsmodels.sandbox.stats.multicomp.fdrcorrection0.html#statsmodels.sandbox.stats.multicomp.fdrcorrection0

    Updates
    -------
    - 11/20/2015: created function
    - 01/25/2016: added option ``return_as_df``

    Parameters
    ----------
    X : ndarray of shape [n,p]
        Data matrix (samples as row vectors)
    y : ndarray of shape [n,]
        Label vector
    return_as_df : bool (default=False)
        Return result as [n,3] shaped pandas DataFrame

    Returns
    -------
    tstats : ndarray of shape [n,]
        Vector of tstats
    pval_corr : ndarray of shape [n,]
        pvalues adjusted for multiple hypothesis testing to limit FDR
    idx_rejected : array, bool (shape [n,])
        True if a hypothesis is rejected, False if not
    """
    tstats, pval = ttest_twosample_fixnan(X,y)

    from statsmodels.sandbox.stats.multicomp import fdrcorrection0
    idx_rejected, pval_corr = fdrcorrection0(pval, alpha=0.05)
    idx_rejected = idx_rejected.astype(int)

    #tstats[~idx_rejected] = 0
    #^^^commented out on 12/03/2015

    if return_as_df:
        df = pd.DataFrame([tstats,pval_corr, idx_rejected],
                          index=['tstats','pval','rejected']).T
        return df
    else:
        return tstats, pval_corr, idx_rejected


def plt_symm_xaxis():
    xlim_abs = max(np.abs(plt.gca().get_xlim()))
    plt.gca().set_xlim(-xlim_abs,xlim_abs)


def corr_dropnans(vec1, vec2, corr_type = 'pearson'):
    """ Compute Pearson correlation and p-value, with NAN indices dropped internally

    **(created 11/19/2015)**

    Originally created this for computing correlation between
    **classification-score** clf_score and **clinical-score**

    Parameters
    ----------
    vec1, vec2 : ndarray
        1d-ndarray of same length to correlate with.
        Indices with NAN in either array will be dropped in the calculation
        of correlation and pvalue
    corr_type : {'pearson', 'spearman','kendall'}
        Type of correlation

    Returns
    --------
    correlation : scalar float
        Correlation coefficient
    pvalue : scalar float
        2-tailed p-vallue

    Dev-file
    ---------
    ``pnc_analyze_clf_summary_1118.py``

    Updates
    -------
    **02/09/2016** - added argument ``corr_type``, with default ``'pearson'``
    """
    from scipy.stats import pearsonr,spearmanr,kendalltau

    # indices with: NOT infinity and NOT nan
    idx = np.isfinite(vec1) & np.isfinite(vec2)

    if corr_type == 'pearson':
        return pearsonr(vec1[idx], vec2[idx])
    elif corr_type == 'spearman':
        return spearmanr(vec1[idx], vec2[idx])
    elif corr_type == 'kendall':
        return kendalltau(vec1[idx], vec2[idx])
    else:
        err_msg = "Argument 'corr_type' must either be 'pearson', "+\
                  "'spearman', or 'kendall'"
        raise ValueError(err_msg)


def cross_corr_dropnans(data_mat1, data_mat2,corr_type='pearson'):
    """ Compute matrix of cross-correlation and pvalues from two matrices

    Parameters
    -----------
    data_mat1 : ndarray or pd.DataFrame of shape [n,ncol1]
        Data matrix to be correlated with the other input with the same number
        of rows
    data_mat2 : ndarray or pd.DataFrame of shape [n,ncol2]
        Data matrix to be correlated with the other input with the same number
        of rows
    corr_type : {'pearson', 'spearman','kendall'}
        Type of correlation

    Returns
    --------
    corrmat : ndarray of shape [n_col1, n_col2]
        Matrices of cross-correlation (Pearson's correlation)
    pvalmat : ndarray of shape [n_col1, n_col2]
        Matrices of two-side pvalue from Pearson's correlation

    Remark
    -------
    http://stackoverflow.com/questions/24432101/
    correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix

    According to Stackoverflow above, there's no built-in method for obtaining
    cross-correlation-matrix of p values (must loop)

    Dev-file
    ---------
    ``pnc_analyze_clf_summary_1118.py``

    Updates
    -------
    **02/09/2016** - added argument ``corr_type``, with default ``'pearson'``
    """
    # if supplied data is a DataFrame, extract values (code below assumes ndarray)
    if isinstance(data_mat1, pd.DataFrame):
        data_mat1 = data_mat1.values
    if isinstance(data_mat2, pd.DataFrame):
        data_mat2 = data_mat2.values

    assert data_mat1.shape[0] == data_mat2.shape[0], 'number of rows disagree!'

    nrow, ncol1 = data_mat1.shape
    ncol2 = data_mat2.shape[1]

    # initalize two-zero matrix: corrmat and pmat
    corrmat = np.zeros((ncol1, ncol2))
    pvalmat = np.zeros((ncol1, ncol2))

    for i1 in range(ncol1):
        for i2 in range(ncol2):
            corr, pval = corr_dropnans(data_mat1[:,i1],
                                       data_mat2[:,i2], corr_type)
            corrmat[i1,i2] = corr
            pvalmat[i1,i2] = pval

    return corrmat, pvalmat

def record_str_to_list(file_list_in):
    """
    Handy when dealing with a numpy record array of strings - situation
    encountered when I use ``scipy.io.matlab.loadmat`` to load a matlab
    struct variable.

    Used to be in ``tak.data_io.ndarray_string_to_list``
    """
    file_list = []
    # i can never memorize the awful code below, so made this function
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    return file_list

def threshold_L1conc(w,perc, return_threshold = False):
    """ Threshold strategy used mostly in my feature selection analysis.

    Created on 11/3/2015

    Idea:

    - I want to know where most of the "weights" in the weight vector ``w``
      is *contentrating* at.
    - This is done by computing the L1 norm of ``w``, and identify the set of
      coefficients that contribute to ``perc``% of the norm

    Paramters
    ----------
    w : ndarray
        ndarray you want to clip off
    perc : float (between 0. and 1.)
        Percentage level of what portion to cutoff at
    return_threshold : bool (default = False)
        Return the cutoff value or not

    Returns
    -------
    w_thresh : ndarray
        Thresholded version of the input ``w``, where
        norm(w_thresh)/norm(w)= ``perc``
    threshold : float (optional)
        Scalar value of threshold value returned if ``return_threshold`` is
        set to True

    Dev file
    ----------
    ``/home/takanori/work-local/tak-ace-ibis/python/analysis/__proto/proto_L1_threshold_1103.py``
    """
    w_abs = np.abs(w)

    #=== sort in descending order ===#
    idx_sort = np.argsort(w_abs)[::-1]
    w_abs_sort = w_abs[idx_sort]

    # index to get the original index-ordering after the sort takes place
    # http://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    idx_revsort = np.argsort(idx_sort)
    assert np.array_equal(w_abs, w_abs_sort[idx_revsort])

    # portion of L1 norm concentrated at each sort-points
    portion = np.cumsum(w_abs_sort)/w_abs.sum()

    # apply threshold
    w_abs_sort_thresh = w_abs_sort
    w_abs_sort_thresh[portion>perc] = 0

    # reverse back to original order
    w_abs_thresh = w_abs_sort_thresh[idx_revsort]

    # find the indices to drop off
    idx_to_keep = np.nonzero(w_abs_thresh)[0]

    # apply threshold to the original data (w/o the absolute value'ing)
    w_thresh = np.zeros( w.shape )
    w_thresh[idx_to_keep] = w[idx_to_keep]

    if return_threshold:
        threshold = np.abs(w_thresh[np.nonzero(w_thresh)[0]].min())
        return w_thresh, threshold
    else:
        return w_thresh
#%%===== Sparse related stuffs (including spectral graph theory) ======
import scipy.sparse as sparse

def save_sparse_csr(filename,array):
    """ Save sparse csr array on disk.

    http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

    Example
    ------
    >>> save_sparse_csr('adjmat_brute',adjmat_brute)
    >>> adjmat_brute = load_sparse_csr('adjmat_brute.npz')
    """
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    """ Load sparse csr array from disk (don't forget the ``.npz`` file extension)

    http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

    Example
    ------
    >>> save_sparse_csr('adjmat_brute',adjmat_brute)
    >>> adjmat_brute = load_sparse_csr('adjmat_brute.npz')
    """
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def diffmat_1d(n):
    row = np.arange(0,n-1)
    col = np.arange(1,n)
    data = np.ones((n-1))
    A = sparse.coo_matrix( (data, (row,col)), shape=(n-1,n))

    row = np.arange(0,n-1)
    col = np.arange(0,n-1)
    data = np.ones((n-1))
    B = sparse.coo_matrix( (data, (row,col)), shape=(n-1,n))

    C= A-B
    C.tocsr()
    return C

def diffmat_2d(nx,ny):
    Dx = diffmat_1d(nx)
    Dy = diffmat_1d(ny)

    Cx = sparse.kron(sparse.eye(ny), Dx)
    Cy = sparse.kron(Dy, sparse.eye(nx))
    C = sparse.vstack([Cx,Cy])
    return C.tocsr()

def diffmat_3d(nx,ny,nz):
    # % Create 1-D difference matrix for each dimension
    Dx = diffmat_1d(nx)
    Dy = diffmat_1d(ny)
    Dz = diffmat_1d(nz)

    # create kronecker structure needed to create the difference operator for
    # each dimension (see my research notes)
    Ix = sparse.eye(nx)
    Iy = sparse.eye(ny)
    Iz = sparse.eye(nz)

    Iyx = sparse.kron(Iy,Ix)
    Izy = sparse.kron(Iz,Iy)

    # create first order difference operator for each array dimension
    Cx = sparse.kron(Izy,Dx)
    Cy = sparse.kron(Iz, sparse.kron(Dy,Ix))
    Cz = sparse.kron(Dz, Iyx)

    # create final difference matrix
    C = sparse.vstack([Cx,Cy,Cz])
    return C.tocsr()

def inc2adj(C):
    """ Convert incidence matrix to adjacency matrix.

    I do this by computing the Laplacian matrix, and then removing its diagonal
    elements (which turns out to be the degree of the nodes).
    The non-diagonal part of the Laplacian matrix will have an entry of -1 at
    at the point of adjacency, so just flip the sign of it at the end.

    To see what I mean here, take a look at
    https://en.wikipedia.org/wiki/Laplacian_matrix#Example

    Parameters
    -----------
    C : sparse matrix in CSR format, shape = [n_edges, n_nodes]
        Incidence matrix, with every row of C has a single -1 and +1 entry

    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_nodes, n_nodes]
        Adjacency matrix (symmetric), with A[i,j] = 1 if node_i and node_j is
        connected.

    Alternative using networkx
    ---------------------------
    >>> import networkx as nx
    >>> A = nx.from_scipy_sparse_matrix(adjmat)
    >>> L = nx.laplacian_matrix(A)
    >>> C = nx.incidence_matrix(A, oriented=True).T
    """
    L = C.T.dot(C) # laplacian matrix
    A = L - sparse.diags(L.diagonal(),0) # remove degree from the diagonal
    A = - A # flip sign to have value of +1 at point of adjacency
    return A.tocsr()

def adj2inc(A):
    """ Convert adjacency matrix to incidence matrix.

    Parameters
    -------
    A : sparse matrix in CSR format, shape = [n_nodes, n_nodes]
        Adjacency matrix (symmetric), with A[i,j] = 1 if node_i and node_j is
        connected.

    Returns
    -----------
    C : sparse matrix in CSR format, shape = [n_edges, n_nodes]
        Incidence matrix, with every row of C has a single -1 and +1 entry

    Alternative using networkx
    ---------------------------
    >>> import networkx as nx
    >>> A = nx.from_scipy_sparse_matrix(adjmat)
    >>> L = nx.laplacian_matrix(A)
    >>> C = nx.incidence_matrix(A, oriented=True).T
    """
    vNode1,vNode2=sparse.triu(A).nonzero()
    n_nodes = A.shape[0]
    n_edges = vNode1.shape[0]
    #df= pd.DataFrame([vNode1,vNode2],index=['node1','node2']).T

    rows = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    cols = np.concatenate([vNode1,vNode2])
    data = np.concatenate([-np.ones(n_edges), np.ones(n_edges)])
    # update 11/13/2015: decided not to convert to int-type, as
    # sp.sparse.lingalg operations apparently are not supported for ints...
    #data = np.concatenate([-np.ones(n_edges), np.ones(n_edges)]).astype(int)
                                         # 11/13/2015 removed this^^^^^^^^^^

    C = sparse.coo_matrix( (data, (rows,cols)), shape=(n_edges,n_nodes))
    return C.tocsr()


#%%====== gui and widgets =====
from matplotlib.widgets import Slider as _Slider
from matplotlib.widgets import Button as _Button
from matplotlib.widgets import RadioButtons as _RadioButtons
class VolumeSlicer(object):
    """ Class I wrote for visualizing 3d volume slice by slice

    Warning: docstring incomplete!
    ----------
    - Documentation here highly incomplete.  i have other important stuffs to do at the moment...
    - The code itself to be is rather straight-forward...
    - When in doubt, use ``object.__dict__.keys()`` to see what attributes I have on the gui, and see
      how they interact with different **events, callbacks, and widgets**

    Development
    -----------
    ``/python/analysis/__proto/proto_3d_slice_viewer_1028.py``

    Parameters
    ----------
    volume : ndarray
        3d ndarray of shape [nx,ny,nz]

    Attributes (main ones)
    -----------
    volume : ndarray
        3d ndarray of shape [nx,ny,nz]
    x,y,z : int
        integer values between [0,nx-1],[0,ny-1],[0,nz-1]
    nx,ny,nz : int
        shape of input volume
    intensity : float
        intensity value of the volume at the current [x,y,z] coordinate
    fig : Figure object
        Figure object
    axes : Length 3 list of Axes object for subplots
        Length 3 Axes object containing subplot 0, 1, 2
    vline, hline : Length 3 list of Lines object for vertical and horizontal line ``self.__init_lines()``
        Length 3 list
    text_slice : Length 3 list of Text objects
        Contains slice location on top-left of each subplots (initialized in ``self.__init_text()``
    text_intensity : Length 3 list of Text objects
        Text Objects for each subplots, where the text is displayed on top of each subplots
        showing the current intensity value
    radio :

    Attributes (more obscure ones)
    -------------------------------
    **Slider (xyz)**

    -

    Methods
    ---------

    **Initializer**

    - ``__init_text``: creates length-3 list of ``text_slice``
    - ``_init_lines``: creates length-3 list of ``vline,hline``
    - ``connect``: connects to all the events we need
        - currently have ``scroll_event`` and ``button_press_event``

    **Slider widget -- (x,y,z) slice update and slice display**

    These are defined in ``__init__``...these are callbacks when slider values are changed via GUI

    - ``init().update_slider_x()``
    - ``init().update_slider_y()``
    - ``init().update_slider_z()``

    On the other hand, these methods are defined in ``self..`` space, and are
    meant to modify the **Slider** position when other events that modify the
    current **(x,y,z)** position takes place (namely ``button_press_event`` and
    ``scroll_event``).

    Also importantly, these events will update the slice images displayed on the subplots

    - ``update_slider_x_external_event(self)``: updates xslider value and update xslice image display
    - ``update_slider_y_external_event(self)``: updates yslider value and update yslice image display
    - ``update_slider_z_external_event(self)``: updates zslider value and update zslice image display


    **Slider widget -- clims (vmin, vmax)**

    - ``update_slider_vmin(self,val)`` : callback when ``self.slider_vmin`` occurs
    - ``update_slider_vmax(self,val)`` : callback when ``self.slider_vmax`` occurs
    - ``change_clim(self,vmin,vmax)``` : update clims (called when above ``update_slider`` function gets called)


    References
    ----------
    - http://matplotlib.org/users/event_handling.html
    - http://matplotlib.org/api/widgets_api.html
    - http://matplotlib.org/api/backend_bases_api.html
    - Motivated from the ``DraggableRectangle`` demo at http://matplotlib.org/1.4.0/users/artists.html

    Rst ref
    -------
    - http://docutils.sourceforge.net/docs/user/rst/quickref.html

    Events that you can connect to
    -----------
    `[ref] <http://matplotlib.org/users/event_handling.html>`_

    .. csv-table::
       :header: Event name, Class, Description
       :delim: -

       ``button_press_event`` -  MouseEvent_ - mouse button is pressed
       ``button_release_event`` -    MouseEvent_ - mouse button is released
       ``draw_event`` -  DrawEvent_ - canvas draw
       ``key_press_event`` - KeyEvent_ - key is pressed
       ``key_release_event`` -   KeyEvent_ - key is released
       ``motion_notify_event`` - MouseEvent_ - mouse motion
       ``pick_event`` -  PickEvent_ - an object in the canvas is selected
       ``resize_event`` -    ResizeEvent_ - figure canvas is resized
       ``scroll_event`` -    MouseEvent_ - mouse scroll wheel is rolled
       ``figure_enter_event`` -  LocationEvent_ - mouse enters a new figure
       ``figure_leave_event`` -  LocationEvent_ - mouse leaves a figure
       ``axes_enter_event`` -    LocationEvent_ - mouse enters a new axes
       ``axes_leave_event`` -    LocationEvent_ - mouse leaves an axes

    .. _MouseEvent: http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.MouseEvent
    .. _DrawEvent: http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.DrawEvent
    .. _KeyEvent: http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.KeyEvent
    .. _LocationEvent: http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.LocationEvent
    .. _PickEvent: http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.PickEvent
    .. _ResizeEvent: http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.ResizeEvent
    """
    def __init__(self, volume, clim=None, cmap=None):
        """
        Parameters
        ----------
        volume : ndarray
            3d array
        clim : length-2 tuple of float values (default=None) (**Added 11/09/2015**)
            the default vlim range for the clims slider.  if None, the min/max
            value from ``volume`` will be used
        cmap : string indicating desired colormap (default=None)  (**Added 11/09/2015**)
            User specified colormap.  For list of choicees, see http://matplotlib.org/examples/color/colormaps_reference.html
        """
        nx,ny,nz = volume.shape
        x,y,z = (nx-1)/2, (ny-1)/2, (nz-1)/2

        self.volume = volume
        self.x = x
        self.y = y
        self.z = z

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.cmap = cmap

        # initialize figure and axes (length 3 list of subplots)
        self.fig, self.axes = plt.subplots(1, 3)

        #%%==== set subplot properties ===#
        # (play around with plt.subplot_tool() gui to be the parameters you like)
        # http://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
        # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplots_adjust
        plt.subplots_adjust(
            left=0.05,
            right = .95,
            bottom=0.05,
            top = 0.9, # here i leave some margin since i don't want the image to interfere the title
            wspace = 0.02,
            )
        """
        Note: the following are the defult values:
            left  = 0.125  # the left side of the subplots of the figure
            right = 0.9    # the right side of the subplots of the figure
            bottom = 0.1   # the bottom of the subplots of the figure
            top = 0.9      # the top of the subplots of the figure
            wspace = 0.2   # the amount of width reserved for blank space between subplots
            hspace = 0.2   # the amount of height reserved for white space between subplots
        """

        #====================== initialize slice display =====================#
        # http://matplotlib.org/api/axes_api.html
        # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axis
        # (decided not to create list...found it to obscure away detail and code readability)
        #=====================================================================#
        """ ditched "list" approach
        slices = [volume[x,:,:],volume[:,y,:],volume[:,:,z]]
        for i, slice in enumerate(slices):
            self.axes[i].imshow(slice.T, cmap="gray", origin="lower")
            self.axes[i].axis('off')
        """
        xslice = self.volume[self.x,:,:]
        yslice = self.volume[:,self.y,:]
        zslice = self.volume[:,:,self.z]

        # intensity-value attribute at current coordinate point
        self.intensity = self.volume[self.x, self.y, self.z]

        #=====================================================================#
        # self note about the "image", "axes", and "image"
        # note: decided not to take the output "image" object, as these are
        # accessible from self.axes[0].images
        # (same way these axes can be extracted from self.fig.axes or self.fig.get_axes())
        # - self.axes[0].get_images() ==== self.axes[0].images
        # - self.fig.axes ==== self.fig.get_axes()
        #=====================================================================#
        """ Here is a summary of the Artists the figure contains
                (from http://matplotlib.org/1.4.0/users/artists.html)
            ================      ===============================================================
            Figure attribute      Description
            ================      ===============================================================
            axes                  A list of Axes instances (includes Subplot)
            patch                 The Rectangle background
            images                A list of FigureImages patches - useful for raw pixel display
            legends               A list of Figure Legend instances (different from Axes.legends)
            lines                 A list of Figure Line2D instances (rarely used, see Axes.lines)
            patches               A list of Figure patches (rarely used, see Axes.patches)
            texts                 A list Figure Text instances
            ================      ===============================================================
        """
        # ignore above block comment, decided to create a list of images to make coding easier
        images = [None]*3 # init empty list of length 3

        #%%=== setting axis labels ===#
        # http://matplotlib.org/api/axis_api.html#matplotlib.axis.Axis.set_tick_params
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.tick_params
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xticklabels
        #self.axes[0].axis('off') #<-argh...doing this will hide all axis property, such as xtick)

        #self.axes[0].set_xlabel(xlabel='coronal',fontsize=15) # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xlabel
        #%% saggital slice (Axis=0)
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xticklabels
        images[0] = self.axes[0].imshow(xslice.T, cmap="gray", origin="lower")
        xticks   = [ 0,   self.ny/5,   self.ny/2,  self.ny*0.8,   self.ny*.97]
        xticklbl = [1, '(Anterior)',       'y', '(Posterior)',  self.ny]
        self.axes[0].set_xticks(xticks)
        self.axes[0].set_xticklabels(xticklbl,ha='center', fontsize=10)

        yticks   = [ 0,   self.nz/10,   self.nz/2,  self.nz*0.9,   self.nz*.97]
        yticklbl = [1, '(inf)',       'z', '(sup)',  self.nz]
        self.axes[0].set_yticks(yticks)
        self.axes[0].set_yticklabels(yticklbl,va='center', fontsize=10)

        self.axes[0].format_coord = Formatter(images[0],one_based_indexing=True) # my impixelinfo

        #plt.xticks( [5,60,100], ['fuck','this','shit'])
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xticklabels
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xticks
        #http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.xticks
        #self.fig.axes[0].tick_params(length=100) # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.tick_params
        #%% coronal slice (Axes = 1)
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xticklabels
        images[1] = self.axes[1].imshow(yslice.T, cmap="gray", origin="lower")

        xticks   = [ 0,   self.nx/5,   self.nx/2,  self.nx*0.8,   self.nx*.97]
        xticklbl = [1, '(Right)',       'x', '(Left)',  self.nx]
        self.axes[1].set_xticks(xticks)
        self.axes[1].set_xticklabels(xticklbl,ha='center', fontsize=10)

        #yticks   = [ 0,   self.nz/10,   self.nz/2,  self.nz*0.9,   self.nz*.97]
        #yticklbl = ['0', '(inf)',       'z', '(sup)',  self.nz]
        yticks=[] # <- the yticks is same as axes[0], so leave blank
        yticklabel=[]
        self.axes[1].set_yticks(yticks)
        self.axes[1].set_yticklabels(yticklabel,va='center', fontsize=10)
        #self.fig.axes[1].axis # <- the hierarchy
        #self.axes[1].axis('off')

        self.axes[1].format_coord = Formatter(images[1],one_based_indexing=True) # my impixelinfo
        #%% axial slice (Axes=2)
        """Note: here origin='upper', because for axial slice, in itksnap,
           as you go "down", your "z" value goes up"""
        images[2] = self.axes[2].imshow(zslice.T, cmap="gray", origin="upper")

        xticks   = [ 0,   self.nx/5,   self.nx/2,  self.nx*0.8,   self.nx*.97]
        xticklbl = ['1', '(Right)',       'x', '(Left)',  self.nx]
        self.axes[2].set_xticks(xticks)
        self.axes[2].set_xticklabels(xticklbl,ha='center', fontsize=10)

        yticks   = [ 0,   self.ny/10,   self.ny/2,  self.ny*0.9,   self.ny*.97]
        yticklbl = ['1', '(ant)',       'y', '(pos)',  self.ny]
        self.axes[2].set_yticks(yticks)
        self.axes[2].yaxis.tick_right()
        self.axes[2].set_yticklabels(yticklbl,va='center', fontsize=10)

        self.axes[2].format_coord = Formatter(images[2],one_based_indexing=True) # my impixelinfo

        #self.axes[2].axis('off')
        #%%=== update self ===
        self.images = images


        #%% connect to all relevant callers
        self.connect()

        #%%====== radio widget for cmap =========
        # http://matplotlib.org/api/widgets_api.html
        # http://matplotlib.org/api/widgets_api.html#matplotlib.widgets.RadioButtons
        #
        # For axisbg, see
        # http://matplotlib.org/1.3.1/api/pyplot_api.html#matplotlib.pyplot.colors
        # http://matplotlib.org/1.3.1/api/axes_api.html#matplotlib.axes.Axes.set_axis_bgcolor
        rax = plt.axes([0.01, 0.82, 0.07, 0.125], axisbg='white')
        rax.set_title('cmap',fontsize=12,fontweight='bold')
        """Note: I purposely added white-space so there's margin between the
                 radio-button and the labels"""
        if self.cmap is None:
            self.radio = _RadioButtons(rax, (' Gray', ' Jet', ' Seismic'))
        else:
            self.radio = _RadioButtons(rax, (' Gray', ' Jet', ' Seismic', ' '+self.cmap))
        self.rax = rax

        # annoying, but can't find a better way to modify fontsize of rbutton legend than below
        # (label list and circle list, where list length = # rbuttons)
        for i, _ in enumerate(self.radio.labels):
            # http://matplotlib.org/api/text_api.html
            self.radio.labels[i].set_fontsize(11)

            # http://matplotlib.org/api/patches_api.html#matplotlib.patches.Circle
            self.radio.circles[i].set_radius(0.095) # 0.05 the default; see line 656 on widgets.py

        self.radio.on_clicked(self.change_cmap)
        #self.fig.axes[2].images[0].set_cmap('jet') # <- it works here...why?

        #%%== radio widget to change scroll jump ==
        # set default scroll step (this attribute will get modified on radio2 click)
        self.scroll_steps = 1

        rax2 = plt.axes([0.09, 0.82, 0.07, 0.125], axisbg='white')
        rax2.set_title('scroll-steps',fontsize=12,fontweight='bold')

        # http://matplotlib.org/api/patches_api.html#matplotlib.patches.Circle
        self.radio2 = _RadioButtons(rax2, (' 1', ' 2', ' 5'), activecolor='red')
        self.rax2 = rax2

        # annoying, but can't find a better way to modify fontsize of rbutton legend than below
        # (label list and circle list, where list length = # rbuttons)
        for i, _ in enumerate(self.radio2.labels):
            # http://matplotlib.org/api/text_api.html
            self.radio2.labels[i].set_fontsize(11)

            # http://matplotlib.org/api/patches_api.html#matplotlib.patches.Circle
            self.radio2.circles[i].set_radius(0.095) # 0.05 the default; see line 656 on widgets.py

        self.radio2.on_clicked(self.change_scroll_step)
        #%%=== Bunch of initialization of objects below ===
        #%% show lines indicating current slice position
        self._init_lines()
        #%% set text of slice lcoations
        self._init_text()
        #%% after all text objects are defined, update titles
        self.update_titles()
        #%% ==== Slider of slices ====
        #%% define slider location, width, and height
        _left=0.22
        _bottom=0.82
        _space = 0.05
        _height = 0.02
        _width = 0.7
        self.ax_x = plt.axes([_left, _bottom+_space*2, _width, _height], axisbg='white')
        self.ax_y = plt.axes([_left, _bottom+_space, _width, _height], axisbg='white')
        self.ax_z = plt.axes([_left, _bottom, _width, _height], axisbg='white')

        """
        http://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Slider
        class matplotlib.widgets.Slider(ax, label, valmin, valmax, valinit=0.5,
                                        valfmt=u'%1.2f', closedmin=True,
                                        closedmax=True, slidermin=None,
                                        slidermax=None, dragging=True, **kwargs)
        """
        valfmt = u'%3d'
        # tricky here with the indexing; the slider values will be subtracted by 1 in the callers
        # (here, for gui-display, i want it to be like matlab/itksnap...1-based indexing)
        # (but internally we have 0-based indexing, so for array data accessing, subtract of one from slider values)
        slider_x = _Slider(self.ax_x, 'x', 1, self.nx, valinit=(self.nx)/2,valfmt=valfmt)
        slider_y = _Slider(self.ax_y, 'y', 1, self.ny, valinit=(self.ny)/2,valfmt=valfmt)
        slider_z = _Slider(self.ax_z, 'z', 1, self.nz, valinit=(self.nz)/2,valfmt=valfmt)

        def update_slider_x(val):
            #print vars(val)
            #print int(slider_x.val)
            self.x = int(slider_x.val)-1 # -1 to get back to 0 based indexing
            self.update_intensity() # <- update intensity value of current (x,y,z) coord
            self.update_xslice()
            self.update_lines()
            self.update_titles()

        def update_slider_y(val):
            #print int(slider_y.val)
            self.y = int(slider_y.val)-1
            self.update_intensity() # <- update intensity value of current (x,y,z) coord
            self.update_yslice()
            self.update_lines()
            self.update_titles()

        def update_slider_z(val):
            #print int(slider_z.val)
            self.z = int(slider_z.val)-1
            self.update_intensity() # <- update intensity value of current (x,y,z) coord
            self.update_zslice()
            self.update_lines()
            self.update_titles()

        slider_x.on_changed(update_slider_x)
        slider_y.on_changed(update_slider_y)
        slider_z.on_changed(update_slider_z)

        # add sliders to self since we need to cal these to sync with other shit
        self.slider_x = slider_x
        self.slider_y = slider_y
        self.slider_z = slider_z
#        #%% --- reset button for xyz slider ---
#        reset_xyz = plt.axes([0.8, 0.025, 0.1, 0.04])
#        button = Button(reset_xyz, 'Reset', color=axcolor, hovercolor='0.975')
#        def reset(event):
#            sfreq.reset()
#            samp.reset()
#        button.on_clicked(reset)
        #%% === create "reset" button for xyz slider above ====
        """To get back the axes rect, use:
        self.ax_x.get_position() or self.ax_x.get_window_extent()
        """
        self.ax_reset_button_xyz = plt.axes([0.9, 0.95, 0.075, 0.04])
        self.reset_button_xyz = _Button(self.ax_reset_button_xyz, 'Reset\n(xyz pos)',
                                        color='w', hovercolor='y')
        self.reset_button_xyz.label.set_fontsize(13)

        def reset_slider_xyz(event):
            """ reset is a built-in method in the Slider widget
            ('effin docstring doesn't explain the methods....have to go to
             api directly for the class methods
             http://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Button
            """
            self.slider_x.reset()
            self.slider_y.reset()
            self.slider_z.reset()
        self.reset_button_xyz.on_clicked(reset_slider_xyz)
        #%% === slider for clims ====
        """ Set range of clim"""
        if clim is None:
            # 11/13: symmetrized version (make this an option in the future)
            magn_ = np.maximum( self.volume.min(), self.volume.max() )
            self.vmin_min = -magn_
            self.vmax_max = +magn_
            # non-symm
            #self.vmin_min = self.volume.min()
            #self.vmax_max = self.volume.max()
        else:
            self.vmin_min = clim[0]
            self.vmax_max = clim[1]
        # default vmin,vmax value
        self.vmin = self.vmin_min
        self.vmax = self.vmax_max

        for im in self.images:
            im.set_clim(self.vmin, self.vmax)

        self.ax_vmin = plt.axes([_left, 0.02, _width, _height], axisbg='white')
        self.ax_vmax = plt.axes([_left, 0.05, _width, _height], axisbg='white')
        """
        http://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Slider
        class matplotlib.widgets.Slider(ax, label, valmin, valmax, valinit=0.5,
                                        valfmt=u'%1.2f', closedmin=True,
                                        closedmax=True, slidermin=None,
                                        slidermax=None, dragging=True, **kwargs)
        """
        range_ = self.volume.max() - self.volume.min()
        self.slider_vmin = _Slider(self.ax_vmin, 'vmin', self.vmin_min,
                             self.vmax_max-0.01*range_, color='y',valinit=self.vmin,valfmt='%5.2f')
        self.slider_vmax = _Slider(self.ax_vmax, 'vmax', self.vmin_min+0.01*range_,
                             self.vmax_max, color='y', valinit=self.vmax,valfmt='%5.2f')

        """ The callback here are defined outside __init__ since I need to update
            the self.vmin and self.vmax attributes for updating color intensity"""
        self.slider_vmin.on_changed(self.update_slider_vmin)
        self.slider_vmax.on_changed(self.update_slider_vmax)
        #%% === Create reset button for clims
        """To get back the axes rect, use:
        self.ax_x.get_position() or self.ax_x.get_window_extent()
        """
        self.ax_reset_button_clim = plt.axes([0.09, 0.02, 0.06, 0.05])
        self.reset_button_clim = _Button(self.ax_reset_button_clim, 'Reset\n(clim)',
                                         color='w', hovercolor='y')
        self.reset_button_clim.label.set_fontsize(13)

        def reset_slider_clim(event):
            """ reset is a built-in method in the Slider widget
            ('effin docstring doesn't explain the methods....have to go to
             api directly for the class methods
             http://matplotlib.org/api/widgets_api.html#matplotlib.widgets.Button
            """
            self.slider_vmin.reset()
            self.slider_vmax.reset()
        self.reset_button_clim.on_clicked(reset_slider_clim)

        #%% === colorbar ===
        #self.ax_vmax = plt.axes([_left, 0.05, _width, _height], axisbg='white')
        self.ax_cbar = self.fig.add_axes([_left, 0.08, _width,_height*1.5])
        self.cbar = self.fig.colorbar(self.images[2],
                                      cax=self.ax_cbar,
                                      orientation='horizontal',
                                      #ticks = cbar_ticks(self.images[2])
                                      )
        """
        Change colorbar fontsize and label locations via pad
        Ref: http://stackoverflow.com/questions/29074820/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbas
        API: http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.tick_params
        """
        self.cbar.ax.tick_params(labelsize=16, pad = -42.1,
                                 length = 5, # ticklength
                                 top = True, # tick on top
                                 )
        #cbar.set_label('Coefficient weights') # <- just know this exists
        #%%_______ END OF INIT________

    def connect(self):
        """connect to all the events we need

        - http://matplotlib.org/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.mpl_connect

        Returns
        -------
        **cid** - callbackid
        """
        self.cid_scroll = self.fig.canvas.mpl_connect(
            'scroll_event', self.on_scroll)
        self.cid_on_click = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_click)
        #self.cidmotion = self.rect.figure.canvas.mpl_connect(
        #    'motion_notify_event', self.on_motion)

    def _init_text(self,size=14,color='white'):
        """Initialize text indicating slice location on the image.

        Without this, the text will keep overlaying on top of each other.
        These text_slide list objects will be updated via ``text.set_text()`` method

        Attributes returned
        -------------------
        text_slice : Length 3 list of Text objects
            Contains slice location on top-left of each subplots (initialized in ``self.__init_text()``
        text_intensity : Length 3 list of Text objects
            Text Objects for each subplots, where the text is displayed on top of each subplots
            showing the current intensity value

        """
        #=== define list of text objects for "text_slice" (the shit on the top-left of each subplot) ===
        self.text_slice = [None]*3
        self.text_slice[0] = self.axes[0].text(2,self.nz-2,'x={:3}'.format(self.x+1),color=color,size=size,va='top')
        self.text_slice[1] = self.axes[1].text(2,self.nz-2,'y={:3}'.format(self.y+1),color=color,size=size,va='top')

        # the y coord goes in reverse direction in itksnap; so flip vertical position of text
        self.text_slice[2] = self.axes[2].text(2,+2,'z={:3}'.format(self.z+1),color=color,size=size,va='top')
        #self.fig.canvas.draw()

        #=== define list of text objects for "text_intensity" (the shit on the top-middle of each subplot) ===
        self.text_intensity = [None]*3
        self.text_intensity[0] = self.axes[0].text(self.ny/2,self.nz-2,'val={:4.2f}'.format(self.intensity),color=color,size=size,va='top',ha='center')
        self.text_intensity[1] = self.axes[1].text(self.nx/2,self.nz-2,'val={:4.2f}'.format(self.intensity),color=color,size=size,va='top',ha='center')
        self.text_intensity[2] = self.axes[2].text(self.nx/2,       +2,'val={:4.2f}'.format(self.intensity),color=color,size=size,va='top',ha='center')

    def _init_lines(self, lw=1, color='r'):
        """ Initialize the lines indicating current slice location.

        Only used in the ``self.__init__`` method

        Ref
        ---
        http://stackoverflow.com/questions/17819260/how-can-i-delete-plot-lines-that-are-created-with-mouse-over-event-in-matplolib
        """
        # initialize empty list
        self.vline = [None]*3
        self.hline = [None]*3
        self.vline[0] = self.axes[0].axvline(self.y,lw=lw,color=color)
        self.hline[0] = self.axes[0].axhline(self.z,lw=lw,color=color)

        self.vline[1] = self.axes[1].axvline(x=self.x, lw=lw,color=color)
        self.hline[1] = self.axes[1].axhline(y=self.z, lw=lw,color=color)

        self.vline[2] = self.axes[2].axvline(x=self.x, lw=lw,color=color)
        self.hline[2] = self.axes[2].axhline(y=self.y, lw=lw,color=color)

        #print self.vline[0].get_data()
        #print self.hline[0].get_data()

    def update_intensity(self):
        """self.update_intensity() # <- update intensity value of current (x,y,z) coord"""
        self.intensity = self.volume[self.x, self.y, self.z]

    def update_lines(self):
       """ Update the lines indicating current slice location.

        Ref
        ---
        http://stackoverflow.com/questions/17819260/how-can-i-delete-plot-lines-that-are-created-with-mouse-over-event-in-matplolib
       """
       #print "(x,y,z) = ({:3},{:3},{:3})".format(self.x,self.y,self.z)
#       self.vline[1].set_ydata(self.x) # <- somehow doesn't work...use full data aproach
#       self.hline[1].set_xdata(self.z)
       self.vline[0].set_data(2*[self.y],      [0,1])
       self.hline[0].set_data(     [0,1], 2*[self.z])

       self.vline[1].set_data(2*[self.x],      [0,1])
       self.hline[1].set_data(     [0,1], 2*[self.z])

       self.vline[2].set_data(2*[self.x],      [0,1])
       self.hline[2].set_data(     [0,1], 2*[self.y])
       plt.draw()

    def update_slider_vmin(self,val):
        """
        Note
        -----
        - The callback here are defined outside __init__ since I need to update
          the self.vmin and self.vmax attributes for updating color intensity
        """
        self.vmin = val
        #self.image[0].set_clim(self.vmin, self.vmax)

        for im in self.images:
            im.set_clim(self.vmin, self.vmax)

        self.fig.canvas.draw()
        plt.draw()

    def update_slider_vmax(self,val):
        """
        Note
        -----
        - The callback here are defined outside __init__ since I need to update
          the self.vmin and self.vmax attributes for updating color intensity
        """
        self.vmax = val

        for im in self.images:
            im.set_clim(self.vmin, self.vmax)

    def change_clim(self,vmin,vmax):
        """
        """
        for i,im in enumerate(self.images):
            im.set_clim(self.vmin,self.vmax)
            #self.images[i].set_cmap(u'jet')
            #self.images[i].figure.canvas.draw()

        # always do canvas.draw() to update figure
        self.fig.canvas.draw()
                    #%%__done with method__
        # always draw on canvas before leaving any function
        self.fig.canvas.draw()

    def update_slider_x_external_event(self):
        """Update the xslider widget and values when external events occurs.

        Specifically, when the x position changes from either ``self.on_scroll`` or
        ``on_click``, I need to ensure the slider values are in sync as well.
        """
        # Again, careful with the indexing (sliders are for display purpose, so +1)
        self.slider_x.set_val(self.x+1)
        self.update_xslice()

    def update_slider_y_external_event(self):
        """Update the yslider widget and values when external events occurs.

        Specifically, when the x position changes from either ``self.on_scroll`` or
        ``on_click``, I need to ensure the slider values are in sync as well.
        """
        # Again, careful with the indexing (sliders are for display purpose, so +1)
        self.slider_y.set_val(self.y+1) # <- update slider values
        self.update_yslice()

    def update_slider_z_external_event(self):
        """Update the zslider widget and values when external events occurs.

        Specifically, when the x position changes from either ``self.on_scroll`` or
        ``on_click``, I need to ensure the slider values are in sync as well.
        """
        # Again, careful with the indexing (sliders are for display purpose, so +1)
        self.slider_z.set_val(self.z+1)
        self.update_zslice()

    def update_xslice(self):
        self.images[0].set_data(self.volume[self.x,:,:].T)

    def update_yslice(self):
        self.images[1].set_data(self.volume[:,self.y,:].T)

    def update_zslice(self):
        self.images[2].set_data(self.volume[:,:,self.z].T)

    def change_cmap(self,label):
        """ label = string of the selected radio button (rbutton1)

        So in this case, if Radio " Gray" is selected, we get string " Gray" in label
        """
        # i added whitespace on the label, so remove that
        label=label[1:].lower() # also lowercase the string
        #print label

        # change colormap for each images
        #import time
        #self.images[0].set_cmap(u'jet')
        #self.images[0].figure.canvas.draw()
        for i,im in enumerate(self.images):
            im.set_cmap(label)
            #self.images[i].set_cmap(u'jet')
            #self.images[i].figure.canvas.draw()

        # always do canvas.draw() to update figure
        self.fig.canvas.draw()

    def change_scroll_step(self,label):
        """ rbutton2"""
        # i added whitespace on the label, so remove that, and convert to int
        self.scroll_steps = int(label[1:])

    def update_titles(self,fs=14,fw='bold'):
        # +1 added to titles since i want consistency with itksnap (1 based indexing)
        #=== display main suptitle (may drop this...kinda useless and wasteful of space on canvas ===#
        val = self.volume[self.x, self.y, self.z] # current intensity value
        plt.suptitle('(x,y,z) = ({:3},{:3},{:3}),   '.format(self.x+1,self.y+1,self.z+1) +
                     'Value = {:4.3e},       '.format(val) + # on suptitle, be generous on precision
                     '(nx,ny,nz) = ({:3},{:3},{:3})'.format(*self.volume.shape))

        # add subplot titles
        self.axes[0].set_title("(y,z) = ({:3}, {:3})".format(self.y+1,self.z+1), fontsize=fs,fontweight=fw)
        self.axes[1].set_title("(x,z) = ({:3}, {:3})".format(self.x+1,self.z+1), fontsize=fs,fontweight=fw)
        self.axes[2].set_title("(x,y) = ({:3}, {:3})".format(self.x+1,self.y+1), fontsize=fs,fontweight=fw)

        #self.axes[0].set_title('x=%3d    ' % (self.x) + '(y,z) = (%3d,%3d)' %
        #                        (self.y+1,self.z+1),size=12)
        #self.axes[0].set_title('x=%3d    ' % (self.x) + '(y,z) = (%3d,%3d)' %
        #                        (self.y+1,self.z+1),size=12)
        #self.axes[1].set_title('y={:3}    '.format(self.y) + '(x,z) = '+
        #                        '({:3},{:3})'.format(self.x+1,self.z+1),size=12)
        #self.axes[2].set_title('z={:3}    '.format(self.z) + '(x,y) = '+
        #                        '({:3},{:3})'.format(self.x+1,self.y+1),size=12)

        #=== add current slice location on top-left corner of each subplot ===#
        self.text_slice[0].set_text('x={:3}'.format(self.x+1))
        self.text_slice[1].set_text('y={:3}'.format(self.y+1))
        self.text_slice[2].set_text('z={:3}'.format(self.z+1))

        #=== add current intensity value at all subplots ===#
        self.text_intensity[0].set_text('val={:5.4f}'.format(self.intensity))
        self.text_intensity[1].set_text('val={:5.4f}'.format(self.intensity))
        self.text_intensity[2].set_text('val={:5.4f}'.format(self.intensity))

    def on_scroll(self, event):
        """ Scroll over slice at a time via mouse scroll"""
        step = self.scroll_steps
        #pprint(vars(event)) # print attributes of the event
        if event.inaxes == self.axes[0]:
            #print "in subplot1 (x = {:2})".format(self.x)
            if event.button=='up':
                self.x = np.clip(self.x+step, 0, self.nx-1)
            else:
                self.x = np.clip(self.x-step, 0, self.nx-1)
            #self.update_xslice() # <- the slider will take care of this update
            self.update_slider_x_external_event()
            #self.axes[0].set_title('{}'.format(self.x))
            #plt.draw()
            #self.ind = np.clip(self.ind+1, 0, self.slices-1)
            #print "in subplot1"
            #sys.stdout.flush()
        elif event.inaxes == self.axes[1]:
            if event.button=='up':
                self.y = np.clip(self.y+step, 0, self.ny-1)
            else:
                self.y = np.clip(self.y-step, 0, self.ny-1)
            #self.update_yslice() # <- the slider will take care of this update
            self.update_slider_y_external_event()
        elif event.inaxes == self.axes[2]:
            if event.button=='up':
                self.z = np.clip(self.z+step, 0, self.nz-1)
            else:
                self.z = np.clip(self.z-step, 0, self.nz-1)
            #self.update_zslice() # <- the slider will take care of this update
            self.update_slider_z_external_event()
        else:
            return # not on any of the subplot Axes...do nothing

        # update intensity attribute at current coordinate point
        self.intensity = self.volume[self.x, self.y, self.z]
        #self.update_titles()
        #self.update_lines()
        #self.update_all_but_image()
        self.fig.canvas.draw()

    def on_click(self,event):
        """ A ``button_press_event``, where we jump to the clicked location"""
        #pprint(vars(event)) # print attributes of the event
        if event.inaxes == self.axes[0]:
            self.y = int(event.xdata) # type caseted to int
            self.z = int(event.ydata)

            self.update_slider_y_external_event()
            self.update_slider_z_external_event()

            #== these aren't need; slider update above will take care of them ==#
            #self.update_yslice()
            #self.update_zslice()
        elif event.inaxes == self.axes[1]:
            self.x = int(event.xdata) # type caseted to int
            self.z = int(event.ydata)

            self.update_slider_x_external_event()
            self.update_slider_z_external_event()

            #== these aren't need; slider update above will take care of them ==#
            #self.update_xslice()
            #self.update_zslice()
        elif event.inaxes == self.axes[2]:
            self.x = int(event.xdata) # type caseted to int
            self.y = int(event.ydata)

            self.update_slider_x_external_event()
            self.update_slider_y_external_event()

            #== these aren't need; slider update above will take care of them ==#
            #self.update_xslice()
            #self.update_yslice()
        else: # not on any of the subplot Axes...do nothing
            return # do nothing

        # update intensity attribute at current coordinate point
        self.intensity = self.volume[self.x, self.y, self.z]

        #self.update_lines()
        #self.update_titles()
        #self.update_all_but_image()
        self.fig.canvas.draw()
#%%========= ipython notebook stuffs (ipynb stuffs) =================#
def ipynb_print_toggle_code():
    """ print snippet for toggling code in ipynb

    References
    -------------
    - http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
    - http://stackoverflow.com/questions/9301466/python-formatting-large-text
    - http://stackoverflow.com/questions/10660435/pythonic-way-to-create-a-long-multi-line-string
    """
    import textwrap
    print textwrap.dedent("""
    from IPython.display import HTML

    HTML('''<script>
    code_show=true;
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    }
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
    """)

def ipynb_toc():
    """
    Copy and paste the following snippet as markdown cell at the top of notebook:

<h1 id="tocheading">Table of Contents</h1>
<div id="toc"></div>

    Then copy and paste the following as code cell

%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')

    See:
    https://github.com/kmahelona/ipython_notebook_goodies
    """
    print ipynb_toc.__doc__

def ipynb_print_toggle_code2():
    """ print snippet for toggling code in ipynb

    References
    -------------
    - http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
    - http://stackoverflow.com/questions/9301466/python-formatting-large-text
    - http://stackoverflow.com/questions/10660435/pythonic-way-to-create-a-long-multi-line-string
    """
    import textwrap
    str = textwrap.dedent("""
        from IPython.display import HTML

        HTML('''<script>
        code_show=true;
        function code_toggle() {
         if (code_show){
         $('div.input').hide();
         } else {
         $('div.input').show();
         }
         code_show = !code_show
        }
        $( document ).ready(code_toggle);
        </script>
        <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
        """)
    return str

#%%===== Connectome related stuffs ==============#
def get_nodes_dropped_from_95():
    """ Get the nodes dropped from the original 95 ROI (to arrive at the 86 ROI)

    **11/05/2015** Data i/o updated after migrating to new git repository.

    Usage
    -------
    >>> label_dropped, idx_dropped = get_nodes_dropped_from_95()

    Here's the output values

    >>> tw.get_nodes_dropped_from_95()[0] # labels dropped
    Out[39]: array([16, 19, 20, 27, 55, 56, 59, 96, 97]
    >>> tw.get_nodes_dropped_from_95()[1] # indices dropped
    Out[40]: array([39, 42, 43, 45, 47, 89, 90, 92, 94]

    Devfile
    -------
    ``~/tak-ace-ibis/python/analysis/__sanity_checks/check_desikan_nodes_to_drop.ipynb``
    """
    filepath = '/home/takanori/work-local/tak-ace-ibis/data_local'

    node86 = np.loadtxt(os.path.join(filepath,'86_labels.txt')).astype(int)
    node95 = np.loadtxt(os.path.join(filepath,'95_labels.txt')).astype(int)

    # node labels dropped
    label_dropped = np.setdiff1d(node95,node86)

    # indices of the nodes dropped
    idx_dropped = np.where(~np.in1d(node95,node86))[0]

    # np.sort needed since "set" operation apparently sorts shit
    assert(np.array_equal(label_dropped, np.sort(node95[idx_dropped])))

    return label_dropped, idx_dropped

def get_node_info86():
    """ Get node info in pd.DataFrame form.

    Updated 04/12/2016: moved ``get_data_path`` function from core.py to
    new file ``path.py``

    Update: (2016-01-23)

    **11/08/2015** Data i/o updated after migrating to new git repository.

    The csv file from which the DataFrame is created is created by script
    ``/home/takanori/work-local/tak-ace-ibis/python/data_post_oct2015/tw_node_info_86.py``

    Returns
    -------
    df_node : DataFrame of shape [n_node, n_col]
        DataFrame representing node-info


    Usage
    -----
    >>> df_node = get_node_info86()
    """
    from .path import get_data_path
    filepath = os.path.join(get_data_path(),'misc','node_info')
#    if get_hostname() == 'sbia-pc125':
#        filepath='/home/takanori/data/misc/node_info'
#    elif get_hostname() == 'tak-sp3':
#        filepath = "C:\\Users\\takanori\\Desktop\\data\\misc\\node_info"
#    else:
#        # on the computer cluster
#        filepath = '/cbica/home/watanabt/data/misc/node_info/'
    filename='tw_node_info_86.csv'
    return pd.read_csv(os.path.join(filepath,filename))


def get_edge_info86():
    """ Created 12/22/2015

    Updated 02/10/2016: use ``'name_short_h1'`` column instead
    """
    df_node_info = get_node_info86()
    cnt=0
    edge_label = []
    node_i_list = []
    node_j_list = []

    system_i_list = []
    system_j_list = []
    for i in range(86):
        for j in range(i+1,86):
            node_i = df_node_info['name_short_h1'].ix[i]
            node_j = df_node_info['name_short_h1'].ix[j]
#            node_i = df_node_info['name_short'].ix[i]
#            node_j = df_node_info['name_short'].ix[j]
#
#            hemi_i = df_node_info['hemisphere'].ix[i]
#            hemi_j = df_node_info['hemisphere'].ix[j]
#
#            node_i = '('+hemi_i+')'+node_i
#            node_j = '('+hemi_j+')'+node_j

            edge_label.append((node_i,node_j))
            node_i_list.append(node_i)
            node_j_list.append(node_j)

            system_i_list.append(df_node_info['system'].ix[i])
            system_j_list.append(df_node_info['system'].ix[j])
            cnt += 1

    df_edge_info = pd.DataFrame([node_i_list, node_j_list,system_i_list,system_j_list]).T
    df_edge_info.columns = ['node1','node2','node1_system','node2_system']
    return df_edge_info

def get_mni_coord86():
    df_node = get_node_info86()
    coord = df_node.ix[:,['xmni','ymni','zmni']].values
    return coord

def conn2design(connMat):
    """ Convert (d x d x n) stack of symmatric matrix into (n x p) design matrix

    See Also
    --------
    :func:`design2conn`:
        Reverse of this shiat

    Use case
    ---------
    >>> X = conn2design(connMat)
    """
    d,_,n = connMat.shape
    n_edges = d*(d-1)/2
    design = np.zeros((n,n_edges))
    for i in range(n):
        design[i] = sqform(connMat[:,:,i])
    return design

def design2conn(X):
    """ Convert (n x p) design matrix X into a (d x d x n) stack of symmatric matrix

    See Also
    --------
    :func:`conn2design`:
        Reverse of this shiat

    Use case
    ---------
    >>> connMat = design2conn(connMat)
    """
    n,p = X.shape
    d = np.int((1+np.sqrt(1+8*p))/2)
    connMat = np.zeros((d,d,n))
    for i in range(n):
        connMat[:,:,i] = sqform(X[i,:])
    return connMat
#%%========== Simple wrappers ================#
def sqform(W, force='no', checks=True):
    """ Wrapper to ``scipy.spatial.distance.squareform``

    Made this since I can never remember the above module location w/o google...

    **Below is a copy and paste of the docstring from the original function**

    ------------------

    Converts a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    Parameters
    ----------
    X : ndarray
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to 'tovector' or 'tomatrix',
        the input will be treated as a distance matrix or distance vector
        respectively.
    checks : bool, optional
        If `checks` is set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.

    Returns
    -------
    Y : ndarray
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.

    Notes
    -----

    1. v = squareform(X)

       Given a square d-by-d symmetric distance matrix X,
       ``v=squareform(X)`` returns a ``d * (d-1) / 2`` (or
       `${n \\choose 2}$`) sized vector v.

      v[{n \\choose 2}-{n-i \\choose 2} + (j-i-1)] is the distance
      between points i and j. If X is non-square or asymmetric, an error
      is returned.

    2. X = squareform(v)

      Given a d*d(-1)/2 sized v for some integer d>=2 encoding distances
      as described, X=squareform(v) returns a d by d distance matrix X. The
      X[i, j] and X[j, i] values are set to
      v[{n \\choose 2}-{n-i \\choose 2} + (j-u-1)] and all
      diagonal elements are zero.
    """
    from scipy.spatial.distance import squareform
    return squareform(W, force='no', checks=True)
#%%========== image display related stuffs =============================
def fig_legend_move(xpos=1.2,ypos=0.7):
    """ Used since pandas plot's legend often hides the important part of image.

    Source
    -------
    >>> plt.legend(bbox_to_anchor=(xpos,ypos))
    """
    plt.legend(bbox_to_anchor=(xpos,ypos))

def fig_legend_hide(ax):
    """ Hide legend from axes.  Kept from mneumonic

    Source
    -------
    >>> ax.legend().set_visible(False)
    """
    ax.legend().set_visible(False)
    #ax.legend(bbox_to_anchor=(1.0, 0.5))

def colorbar_cmap(cmap=None, **kwargs):
    """ My custom colorbar function that includes the cmap argument (clunky but works)

    Warning
    ---------
    FUNCTION NOT FULLY TESTED YET!

    Returns
    --------
    cbar : object
        colorbar object

    Description
    -------------
    I cannot set colorbar cmap easily...see https://github.com/matplotlib/matplotlib/issues/3644

    As an adhoc workaround, here I change the rcParams temporarily, display
    the colorbar, and then reset the original rcParams value.

    See the code to see wth i'm doing...
    """
    if cmap is None:
        warnings.warn('cmap not given...defeats the purpose of this function')
        cbar = plt.colorbar(**kwargs)
    else:
        cmap_original = mpl.rcParams['image.cmap']

        # change cmap temporarily for this function
        mpl.rcParams['image.cmap'] = cmap
        print mpl.rcParams['image.cmap']
        cbar = plt.colorbar(**kwargs)

        # revert back to the original cmap
        mpl.rcParams['image.cmap'] = cmap_original

    return cbar

def cbar_ticks(mappable=None):
    """ Convenience script

    Have colorbar only show tickts at [min, mid, max]

    Update (1030/2015)
    ----------------
    Added optional ``mappable`` parameter

    TODO
    -----
    Include integer as input to indicate how many "inbetween" markers to add

    Usage
    ------
    >>> plt.imshow(W)
    >>> plt.colorbar(orientation='horizontal', pad = pad,ticks=cbar_ticks())
    """
    if mappable is None:
        # try to get ScalarMappable object
        # http://stackoverflow.com/questions/13060450/how-to-get-current-plots-clim-in-matplotlib
        vmin, vmax = plt.gci().get_clim()
    else:
        vmin, vmax = mappable.get_clim()
    return np.array([vmin, (vmin+vmax)/2, vmax])

def set_xyticks(xx,yy):
    """ Show x,y values as ticks in imshow

    Usage
    -------
    >>> xx = np.log2(lam_grid)
    >>> yy = np.log2(gam_grid)
    >>> plt.imshow(grid_result)
    >>> plt.set_xyticks(xx=xx,yy=yy)
    """
    plt.xticks( range(len(xx)), xx, rotation=90)
    plt.xticks( range(len(yy)), yy)


def imtak(im, clim=None,multicursor=False,**kwargs):
    """
    Update 04/27/2016 - added clim option
    """
    from matplotlib.widgets import Cursor
    import mpldatacursor

    img=plt.imshow(im,interpolation='none',**kwargs)
    #plt.colorbar()

    if multicursor:
        display="multiple"
    else:
        display="single"

    dc = mpldatacursor.datacursor(
        formatter='(i, j) = ({i}, {j})\nz = {z:.2f}'.format,
        draggable=True,
        axes=plt.gca(),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
               edgecolor='magenta'), display=display)
    Cursor(plt.gca(),useblit=True, color='red', linewidth= 1 )

    ax = plt.gca()
    ax.format_coord = Formatter(img)

    if clim is not None:
        plt.gci().set_clim(clim)

    return img


def imtakk(im,clim=None):
    figure()
    imtak(im)
    if clim is not None:
        plt.gci().set_clim(clim)

def imconnmat_hemi_subplot_86(W,suptitle=None,ticks=None, fontsize=8):
    """ Display connectivity by hemisphere as subplots (left,right,inter-hemisphere) (created 10/24/2015)

    Demo usage: https://github.com/takwatanabe2004/tak-ace-ibis/blob/master/python/pnc/dump/proto_imconnmat_hemi_subplot_86_and_imgridsearch_subplot_2d.ipynb

    Parameters
    ----------
    W : ndarray
        Symmetric matrix representing connectivity
        (if vector is supplied, will be converted into a symmetric matrix internally)
    suptitle : optional
        Optional subplot title

    Note
    -----
    for now, just assume we're only dealing with the 86 parcellation
    (think about how to extend this function for other parcellation later)
    """
    backend = mpl.get_backend()

    #=========================================================================#
    # Colormap issue
    # https://github.com/matplotlib/matplotlib/issues/3644
    #-------------------------------------------------------------------------#
    # i cannot set colorbar cmap easily...as an adhoc workaround, store the
    # original cmap here, and set it back at the end
    #=========================================================================#
    cmap_original = mpl.rcParams['image.cmap']
    # change cmap temporarily for this function
    mpl.rcParams['image.cmap'] = 'seismic'


    # TODO: for now, this function assumes it's called from ipynb...modify
    #       figsize and cbar placement in the case of qt4 backend
    if backend == 'module://ipykernel.pylab.backend_inline':
        figsize=(16,6)
        fs_sup = 20
    elif backend == 'Qt4Agg':
        figsize=(20,8)
        fs_sup = 24 # sup title fontsize
    fig = plt.figure(figsize=figsize)

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=fs_sup)

    if len(W.shape) is 1:
        # W supplied is in vector form... convert to symmetric matrix form
        W = sqform(W)

    # get node info (used for tick labels)
    if ticks is None:
        df_node = get_node_info86()
        ticks = df_node.lobes.values # use lobes as default


    """ Update 11/03 - Only show yticks on the left-most subplot """
    plt.subplot(131)
    plt.title('Left hemisphere')
    imconnmat_hemi(W,ticks,ticks,fontsize=fontsize,hemi='left')
    plt.subplot(132)
    plt.title('Right hemisphere')
    imconnmat_hemi(W,ticks,ticks,fontsize=fontsize,hemi='right')
    plt.subplot(133)
    plt.title('Inter-hemisphere')
    imconnmat_hemi(W,ticks,ticks,fontsize=fontsize,hemi='inter')

    #=========================================================================#
    # Colorbar placement
    #-------------------------------------------------------------------------#
    # for now, place them on top of the subplot...
    # i had to tweek the values here...figure out how to automate this process
    #=========================================================================#
    # Make an axis for the colorbar on the right side
    # http://stackoverflow.com/questions/7875688/how-can-i-create-a-standard-colorbar-for-a-series-of-plots-in-python
    cax = fig.add_axes([0.15, 0.93, 0.7, 0.062]) # [left,bottom,width,height]01
    caxis_symm() # <- needed to update the symmetrized colorbar...
    cbar=fig.colorbar(plt.gci(), cax=cax, orientation='horizontal', ticks = cbar_ticks())
    #cax = fig.add_axes([0.99, 0.1, 0.01, 0.8]) # [left,bottom,width,height]01
    #fig.colorbar(plt.gci(), cax=cax, orientation='vertical')
    #plt.tight_layout()

    # change colorbar fontsize
    # http://stackoverflow.com/questions/29074820/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbas
    # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.tick_params
    cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('Coefficient weights') # <- just know this exists


    # set colormap back to original
    mpl.rcParams['image.cmap'] = cmap_original
    if hostname == 'takanori-PC':
        fig_set_geom((8,30,1904,860))
    else:
        fig_set_geom((1,31,1600,750))

def imconnmat_hemi(W, xtick_label = None, ytick_label = None, fontsize=8, hemi='L'):
    """ Show subset of connectome ('L','R', or 'inter')

    Warning
    -------
    Only works for the desikan86 parcellation, where the first half is from the
    left hemisphere and the other half is from the right....

    Parameters
    -----------
    W : ndarray
        Symmetric matrix representing connectivity
        (if vector is supplied, will be converted into a symmetric matrix internally)
    hemi : string (default: 'L')
        string indicating what part to show.

        Supported (case insensitive):

        - 'L' or 'left'
        - 'R' or 'right'
        - 'I' or 'inter' (for interhemisphere)

    """
    if len(W.shape) is 1:
        # W supplied is in vector form... convert to symmetric matrix form
        W = sqform(W)

    idx_L = range(0,43)
    idx_R = range(43,86)

    #%% np.ix_ for extracting submatrix in numpy via indexing...
    # ref: http://stackoverflow.com/questions/19161512/numpy-extract-submatrix
    #      http://docs.scipy.org/doc/numpy/reference/generated/numpy.ix_.html
    #%%
    if hemi.lower() in ('l','left'):
        if (xtick_label is not None) and (ytick_label is not None):
            imconnmat( W[np.ix_(idx_L,idx_L)], xtick_label[idx_L], ytick_label[idx_L],fontsize)
        elif (xtick_label is not None) and (ytick_label is None):
            imconnmat( W[np.ix_(idx_L,idx_L)], xtick_label[idx_L], None,fontsize)
        elif (xtick_label is None) and (ytick_label is not None):
            imconnmat( W[np.ix_(idx_L,idx_L)], None, ytick_label[idx_L],None,fontsize)
        else:
            imconnmat( W[np.ix_(idx_L,idx_L)])
    elif hemi.lower() in ('r','right'):
        if (xtick_label is not None) and (ytick_label is not None):
            imconnmat( W[np.ix_(idx_R,idx_R)], xtick_label[idx_R], ytick_label[idx_R],fontsize)
        elif (xtick_label is not None) and (ytick_label is None):
            imconnmat( W[np.ix_(idx_R,idx_R)], xtick_label[idx_R], None,fontsize)
        elif (xtick_label is None) and (ytick_label is not None):
            imconnmat( W[np.ix_(idx_R,idx_R)], None, ytick_label[idx_R],None,fontsize)
        else:
            imconnmat( W[np.ix_(idx_R,idx_R)],fontsize=fontsize)
    elif hemi.lower() in ('i','inter'):
        if (xtick_label is not None) and (ytick_label is not None):
            imconnmat( W[np.ix_(idx_R,idx_L)], xtick_label[idx_L], ytick_label[idx_R],fontsize)
        elif (xtick_label is not None) and (ytick_label is None):
            imconnmat( W[np.ix_(idx_R,idx_L)], xtick_label[idx_L], None,fontsize)
        elif (xtick_label is None) and (ytick_label is not None):
            imconnmat( W[np.ix_(idx_R,idx_L)], None, ytick_label[idx_R],None,fontsize)
        else:
            imconnmat( W[np.ix_(idx_R,idx_L)],fontsize=fontsize)

    imtak(W)


def imconnmat(W, xtick_label = None, ytick_label = None, fontsize=8,
              colorbar='bottom'):
    """ Display connectivity matrix

    Created 10/19/2015

    Parameters
    -----------
    W : ndarray
        matrix representing connectivity (if a vector, will get converted into
        a symmetric matrix via sqform function)
    xtick_label : list/ndarray of strings of label
        list/ndarray of strings of label
    ytick_label : list/ndarray of strings of label
        list/ndarray of strings of label
    fontsize : default 8

    See also
    --------
    imgridsearch

    Update history
    --------------
    - 02/05/2016: added ``colorbar`` option
    """
#    figure()
    if len(W.shape) is 1:
        # W supplied is in vector form... convert to symmetric matrix form
        W = sqform(W)

    plt.pcolormesh(W,edgecolors='k',lw=1,cmap='seismic')
    if xtick_label is not None:
        plt.xticks(np.arange(.5,len(xtick_label)), xtick_label, fontsize=fontsize, rotation=90)
    if ytick_label is not None:
        plt.yticks(np.arange(.5,len(ytick_label)), ytick_label, fontsize=fontsize)
    # plt.tight_layout()
    _ = plt.axis('image') # <- removes the awkward boundary (comment out this line to see what i mean)
    plt.gca().invert_yaxis()
#    plt.colorbar()
    if colorbar in ['horizonta','vertical']:
        plt.colorbar(orientation=colorbar)


    # remove ticks
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        bottom='off',
        left='off',
        right='off',
        top='off'
        )
    # symmetrize caxis
    caxis_symm()
    imtak(W)


def imgridsearch(W, xtick_label = None, ytick_label = None, fontsize=8,
                 cmap='hot',show_max = False, show_cbar=False,**kwargs):
    """ Display gridsearch matrix

    Created 10/21/2015

    Parameters
    -----------
    W : ndarray
        matrix representing connectivity
    xtick_label : list/ndarray of strings of label
        list/ndarray of strings of label
    ytick_label : list/ndarray of strings of label
        list/ndarray of strings of label
    fontsize : default 8
    show_max : bool (default: False)
        show max location (ignores ties for now)
    show_cbar : bool (default: False)
        show colorbar

    See also
    --------
    imconnmat
    """
    #print kwargs
    cmap_original = mpl.rcParams['image.cmap']
    # change cmap temporarily for this function
    mpl.rcParams['image.cmap'] = cmap

    plt.pcolormesh(W,edgecolors='k',lw=1,cmap=cmap,**kwargs)
    if xtick_label is not None:
        plt.xticks(np.arange(.5,len(xtick_label)), xtick_label, fontsize=fontsize, rotation=90)
    if ytick_label is not None:
        plt.yticks(np.arange(.5,len(ytick_label)), ytick_label, fontsize=fontsize)
    # plt.tight_layout()
    _ = plt.axis('image') # <- removes the awkward boundary (comment out this line to see what i mean)
    plt.gca().invert_yaxis()
    #plt.colorbar()

    # remove ticks
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        bottom='off',
        left='off',
        right='off',
        top='off'
        )
    # symmetrize caxis
    #caxis_symm()

    # impixelinfo imitator
    imtak(W,**kwargs)
    if show_cbar:
        #plt.colorbar()
        plt.colorbar(plt.gci(),fraction=0.046,
                            pad=0.04,ticks = cbar_ticks(),
                            format='%.3f')


    if show_max:
        #
        row_max,col_max = np.unravel_index(W.argmax(), W.shape)
        assert( W.max() == W[row_max,col_max])
        plt.text(col_max+.5,row_max+.5, "{:3.3f}".format(W.max()),fontsize=14,
                 horizontalalignment='center',verticalalignment='center')

        #=== show all ties ===
        # find out all the peaks
        for idx_ties in argmax_ties(W):
            ymax = idx_ties[0]
            xmax = idx_ties[1]
            # note: 0.5 added to make ticks align nicely
            # (has to do with how pcolormesh work)
            #plt.text(xmax+0.5,ymax+0.5, r'$\mathbf{\times}$')
            plt.plot(xmax+0.5,ymax+0.5, 'bx', markersize=20,
                     markeredgewidth=3, alpha=0.3)

    mpl.rcParams['image.cmap'] = cmap_original


def imgridsearch_subplot_2d(acclist, xtick_label=None,ytick_label=None, xlabel=None, ylabel=None,
                            n_col=3,vmin=None, vmax=None,fontsize=11,suptitle=None,
                            markTies=True):
    """ Compact display of 2d gridsearch results for ipynb (created 10/24/2015)

    Assumes 2 hyperparameters.

    Function relies on ``imgridsearch``

    Demo usage: https://github.com/takwatanabe2004/tak-ace-ibis/blob/master/python/pnc/dump/proto_imconnmat_hemi_subplot_86_and_imgridsearch_subplot_2d.ipynb

    Parameters
    -----------
    accgrid : list of ndarray
        Each list element is a 2d array representing accuracy at different
        hyperparameter values.
    xtick_label : list/ndarray of strings of label
        list/ndarray of strings of label
    ytick_label : list/ndarray of strings of label
        list/ndarray of strings of label
    xlabel : string (default None)
        xlabel display
    ylabel : string (default None)
        ylabel display
    n_col : int (default 3)
        number of subplot columns to display per each row

    Warning
    --------
    I often get x/y and the rows/cols mixed up...always do sanity checks
    to ensure i didn't f up something

    See Also
    ---------
    :func:`imgridsearch`:
        this function calls this to create the nice "box" type display using
        imcolormesh
    :func:`imconnmat_hemi_sub_86`:
        same idea as this function
    """
    #%% parse inputs
    if xtick_label is None:
        xtick_label = np.arange(acclist[0].shape[1])
    if ytick_label is None:
        ytick_label = np.arange(acclist[0].shape[0])
    if xlabel is None:
        #xlabel = r'log$_2$($\lambda$)'
        xlabel='hyperparameter2'
    if ylabel is None:
        #ylabel = r'log$_2$($\alpha$)'
        ylabel='hyperparameter1'
    backend = mpl.get_backend()

    #=========================================================================#
    # Colormap issue
    # https://github.com/matplotlib/matplotlib/issues/3644
    #-------------------------------------------------------------------------#
    # i cannot set colorbar cmap easily...as an adhoc workaround, store the
    # original cmap here, and set it back at the end
    #=========================================================================#
    cmap_original = mpl.rcParams['image.cmap']
    # change cmap temporarily for this function
    mpl.rcParams['image.cmap'] = 'hot'

    if backend == 'module://ipykernel.pylab.backend_inline':
        figsize=(16,6)
        fs_sup = 20
    elif backend == 'Qt4Agg':
        figsize=(16,6)
        fs_sup = 24 # sup title fontsize

    #%% alright, let's begin
    for i,acc_grid in enumerate(acclist):
        if i%n_col == 0:
            fig = plt.figure(figsize=figsize)
            cnt = 1
            if suptitle is None:
                plt.suptitle('Cross-Validation Accuracy', fontsize=fs_sup)
            else:
                plt.suptitle(suptitle,fontsize=fs_sup)
        plt.subplot(1,n_col,cnt)
        cnt+=1

        # fugly if statements below, but does its job...
        if vmin is None and vmax is None:
            imgridsearch(acc_grid,xtick_label,ytick_label,fontsize=fontsize,show_max=True)
        elif vmin is None:
            # sometimes i like to specify the lowerbound of caxis only
            imgridsearch(acc_grid,xtick_label,ytick_label,fontsize=fontsize,show_max=True,vmin=vmin,vmax=acc_grid.max())
        elif vmin is not None and vmax is not None:
            imgridsearch(acc_grid,xtick_label,ytick_label,fontsize=fontsize,show_max=True,vmin=vmin,vmax=vmax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # optional: mark peak with "x" marks....including ties
        if markTies:
            # find out all the peaks
            for idx_ties in argmax_ties(acc_grid):
                ymax = idx_ties[0]
                xmax = idx_ties[1]
                # note: 0.5 added to make ticks align nicely
                # (has to do with how pcolormesh work)
                #plt.text(xmax+0.5,ymax+0.5, r'$\mathbf{\times}$')
                plt.plot(xmax+0.5,ymax+0.5, 'bx', markersize=20,
                         markeredgewidth=3, alpha=0.3)

        # argh...my ocd'ness is bothered by the trivial detail like below...
        if i+1 is 1:
            title_str = ' 1st CV fold'
        elif i+1 is 2:
            title_str = ' 2nd CV fold'
        elif i+1 is 3:
            title_str = ' 3rd CV fold'
        else:
            title_str = '{:2}th CV fold'.format(i+1)
        plt.title(title_str+' (Max = {:4.2f}%)'.format(acc_grid.max()*100))

        #=====================================================================#
        # Colorbar placement
        #---------------------------------------------------------------------#
        # for now, place them on top of the subplot...
        # i had to tweek the values here...figure out how to automate this process
        #=====================================================================#
        # Make an axis for the colorbar on the top side
        # http://stackoverflow.com/questions/7875688/how-can-i-create-a-standard-colorbar-for-a-series-of-plots-in-python
        cax = fig.add_axes([0.15, 0.93, 0.7, 0.062]) # [left,bottom,width,height]01
        cbar=fig.colorbar(plt.gci(), cax=cax, orientation='horizontal', ticks = cbar_ticks())
        #cax = fig.add_axes([0.99, 0.1, 0.01, 0.8]) # [left,bottom,width,height]01
        #fig.colorbar(plt.gci(), cax=cax, orientation='vertical')
        #plt.tight_layout()

        # change colorbar fontsize
        # http://stackoverflow.com/questions/29074820/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbas
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.tick_params
        cbar.ax.tick_params(labelsize=20)
        #cbar.set_label('Coefficient weights') # <- just know this exists


    # set colormap back to original
    mpl.rcParams['image.cmap'] = cmap_original

def caxis_symm():
    """ symmetrize caxis in imshow (called vmin, vmax in python)

    Created 10/19/2015
    """
    vmin, vmax = plt.gci().get_clim()
    v = np.max( [np.abs(vmin), vmax] )
    plt.gci().set_clim(vmin=-v,vmax=v)

def show_slices(data, x,y,z):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, 3)
   slices = [data[x,:,:],
             data[:,y,:],
             data[:,:,z]]
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

def nii_correct_coord_mat(vol):
    """ Correct for the "flip" in the nifti volume from matlab

    My old matlab scipt uses an nii reader that flips the first two dimension...
    So this function will correct for that flip

    Paramaeters
    ------------
    vol : 3d ndarray
        volume that got "flipped" by using nii_load in matlab

    Return
    --------
    vol : 3d array
        volume corrected for the flip

    Details (10/30/2015)
    -------
    - When i used the ``load_nii` script in matlab, the data gets flipped...
    - See also ``save_IBIS_volume_brainmask_0809.m``
    - To correct for this flip, do this:

    >>> eve176_mat = eve176_mat[::-1,:,:]
    >>> eve176_mat = eve176_mat[:,::-1,:]

    - In matlab, this is done by:

    .. code:: matlab

        eve176_mat = flipdim( eve176_mat, 1);
        eve176_mat = flipdim( eve176_mat, 2);

    Dev
    ----
    ``proto_nii_vol_flip_fix_1030.py``
    """
    vol = vol[::-1,:,:]
    vol = vol[:,::-1,:]
    return vol

#%%===== Plotting related stuffs =================#


class Formatter(object):
    """ My version of impixelinfo (used in ``imtak``)

    Parameters
    ----------
    z_is_int : bool (default=False)
        Display z value as integer (Feature added 10/28/2015)
    flag_one_based : bool (default=False)
        Show x,y info as 1-based indexing like Matlab (Feature added 10/28/2015)

    Ref
    ---
    I got this beautiful shit from stackoverflow:

    http://stackoverflow.com/questions/27704490/interactive-pixel-information-of-an-image-in-python

    Update
    ------
    06/02/2016 - impixelinfo functionality now available as default in imshow.
    code modified accordingly.
    """
    #
    def __init__(self, im, z_int = False, one_based_indexing = False):
        self.im = im
        self.z_int = z_int
        self.one_based_indexing = one_based_indexing

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]

        # the +0.5 offset makes the cursor location more accurate when hovering
        # over the pixel location
        if self.one_based_indexing:
            xx = int(x+1.5) # type cast to int
            yy = int(y+1.5)
        else:
            xx = int(x+.5)
            yy = int(y+.5)

#        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)
        if self.z_int:
            return '(x={:d}, y={:d}), z={:3d}'.format(xx, yy, z)
            #return '(x={:d}, y={:d}), z={:3d}'.format(int(x), int(y), z)
        else:
            #return '( i , j ) = ({:d}, {:d})    {:5.2e}'.format(int(y), int(x), z)
            #return '( i , j ) = ({:d}, {:d})    {:.2f}'.format(int(y), int(x), z)
            #return '( i , j ) = ({:d}, {:d})    {:.2f}'.format(xx,yy, z)
            return '( i , j ) = [{:d}, {:d}]   '.format(yy,xx)

def purge():
    """ My lazy close all """
    plt.close('all')

def imexp():
    """Fullscreen on my secondary monitor (10/01/2015)"""
    x,y,dx,dy=_screen_full
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(x,y,dx,dy)

def figure(pos='default',fignum=None):
    """ My figure generator with control over figure size and position

    Usage
    -------
    >>> fig = tw.figure('full')

    Parameter
    ---------
    pos : string or tuple of (x,y,dx,dy) (default='default', which gives default screensize)
        If a **tuple**, should be length 4 of ``(x,y,dx,dy)``, where

         - x = xstart position
         - y = ystart position
         - dx = xlength
         - dy = ylength

        If a **string**, the following options are recognized:

        - ``'f'`` or ``'full'`` (full screen)
        - ``'l'`` or ``'left'`` (left half of screen)
        - ``'r'`` or ``'right'`` (right half of screen)
        - ``'t'`` or ``'top'`` (top half of screen)
        - ``'b'`` or ``'bottom'`` (bottom half of screen)
        - ``'tl'`` or ``'topleft'``
        - ``'tr'`` or ``'topright'``
        - ``'bl'`` or ``'bottomleft'``
        - ``'br'`` or ``'bottomright'``

    fignum : int (optional)
        assign figure number (optional)

    Returns
    -------
    figure number

    Reference
    -----------
    - http://doc.qt.io/qt-4.8/qwidget.html for api
    - http://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    """
    if fignum is None:
        fig = plt.figure()
    else:
        fig = plt.figure(fignum)
    plt.plot()
    fig_set_geom(pos)
    return fig


def fig_set_geom(pos = _screen_default):
    """ Set figure dimension

    Usage
    -------
    >>> plt.figure()
    >>> tw.fig_set_geom('full')

    Parameter
    ---------
    pos : string or tuple of (x,y,dx,dy) (default='default', which gives default screensize)
        If a **tuple**, should be length 4 of ``(x,y,dx,dy)``, where

         - x = xstart position
         - y = ystart position
         - dx = xlength
         - dy = ylength

        If a **string**, the following options are recognized:

        - ``'f'`` or ``'full'`` (full screen)
        - ``'l'`` or ``'left'`` (left half of screen)
        - ``'r'`` or ``'right'`` (right half of screen)
        - ``'t'`` or ``'top'`` (top half of screen)
        - ``'b'`` or ``'bottom'`` (bottom half of screen)
        - ``'tl'`` or ``'topleft'``
        - ``'tr'`` or ``'topright'``
        - ``'bl'`` or ``'bottomleft'``
        - ``'br'`` or ``'bottomright'``


    Reference
    -----------
    - http://doc.qt.io/qt-4.8/qwidget.html for api
    - http://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    """
    if isinstance(pos, basestring):
        if pos is 'default':
            pos = _screen_default
        elif pos in ('f','full'):
            pos = _screen_full
        elif pos in ('l','left'):
            pos = _screen_left
        elif pos in ('r','right'):
            pos = _screen_right
        elif pos in ('t','top'):
            pos = _screen_top
        elif pos in ('b','bottom'):
            pos = _screen_bottom
        elif pos in ('tl','topleft'):
            pos = _screen_tl
        elif pos in ('tr','topright'):
            pos = _screen_tr
        elif pos in ('bl','bottomleft'):
            pos = _screen_bl
        elif pos in ('br','bottomright'):
            pos = _screen_br
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(*pos)

def fig_get_geom(as_ndarray=True):
    """ Get window geometry as either length 4 tuple or ndarray

    Parameters
    ------------
    as_ndarray : bool (default = True)
        Return as ndarray (if False, will return a tuple)

    Returns
    ---------
    If a **tuple**, should be length 4 of ``(x,y,dx,dy)``, where

         - x = xstart position
         - y = ystart position
         - dx = xlength
         - dy = ylength

    If an **ndarray**, an array of shape [4,] will be returned

    The below source code should clarify wtf i mean

    .. code:: python

        if as_ndarray:
            return np.array([x,y,dx,dy])
        else:
            return x,y,dx,dy

    Reference
    -----------
    - http://doc.qt.io/qt-4.8/qwidget.html for api
    - http://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    """
    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()
    x,y,dx,dy = geom.getRect()
    if as_ndarray:
        return np.array([x,y,dx,dy])
    else:
        return x,y,dx,dy

def subplots(*args,**kwargs):
    """ My wrapper to subplot function with screensize of my flavor (10/01/2015)

    Usage
    -------
    >>> fig, ax = tw.subplots(131)

    Function source
    ---------------
    .. code:: python

        fig,ax = plt.subplots(*args,**kwargs)
        mngr = plt.get_current_fig_manager()
        x,y,dx,dy = _screen_default
        mngr.window.setGeometry(x,y,dx,dy)
        return fig, ax
    """
    fig,ax = plt.subplots(*args,**kwargs)
    mngr = plt.get_current_fig_manager()
    x,y,dx,dy = _screen_default
    mngr.window.setGeometry(x,y,dx,dy)
    return fig, ax
#%%============= machine learning shiats =============================
def roc(ytrue, score, return_thresholds=False):
    """ Get ROC curve

    A wrapper to get roc-curve and auc curve...helpful since I sometimes go
    brain-dead and forget which sklearn module I need to access and how to
    appropriately call them.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values.

    return_thresholds : bool (default=False)
        return array of thresholds

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    auc : float
        Area under the ROC curve

    thresholds : array, shape = [n_thresholds] (optional output)
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    Usage
    ------
    fpr,tpr,auc = tw.roc(ytrue, score)
    """
    fpr,tpr,thresholds = sklearn.metrics.roc_curve(ytrue,score)
    auc = sklearn.metrics.roc_auc_score(ytrue,score)

    if return_thresholds:
        return fpr,tpr,auc, thresholds
    else:
        return fpr,tpr,auc


def zscore(X, return_scaler=False):
    """ zscore transform data matrix

    I like scikit's one since it handles NaNs that arises from 0-variance
    feature

    Input
    ------
    X : ndarray
        (n x p) data matrix

    Output
    -------
    Xz : ndarray
         (n x p) zscored matrix (mean 0, std = 1 over first axis)
    scaler : StandardScaler()
        StandardScaler instance after fitted on X

    Example
    --------
    >>> Xz, _ = zscore(X)
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    if return_scaler:
        return Xz, scaler
    else:
        return Xz


def zscore_training_testing(Xtr, Xts, return_scaler=False):
    """ zscore training and testing data

    Input
    ------
    Xtr : ndarray
        (ntr x p) data matrix for training
    Xts : ndarray
        (nts x p) data matrix for testing
    return_scaler : bool (default=False)
        return ``scaler`` object

    Output
    -------
    Xtrz : ndarray
         (ntr x p) zscored matrix (mean 0, std = 1 over first axis)
    Xtsz : ndarray
         (nts x p) data matrix for testing
         (the mean and std-dev from Xtr is applied to normalized Xts here)
    scaler : StandardScaler()
        StandardScaler instance after fitted on Xtr

    Example
    --------
    >>> Xtrz, Xtsz = zscore_training_testing(Xtr, Xts)

    Warning (10/26/2015)
    -------
    I added an option ``return_scaler`` as a 3rd argument.

    - Since i made the default for this ``False``, this affects my old code that assumed
      that the scaler gets returned.
    - Just be aware when running the code, make sure the 3rd output tuple ``_`` I assigned is removed.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    Xtrz = scaler.transform(Xtr)
    Xtsz = scaler.transform(Xts)
    if return_scaler:
        return Xtrz, Xtsz, scaler
    else:
        return Xtrz, Xtsz


def ttest_feature_sel(X,y, cutoff = 10, dataFrame = True):
    """ Feature selection based on two-sample ttest (equal variance)

    https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test

    Input
    ------
    X : ndarray
        (n x p) design matrix
    y : ndarray
        (n,) label vector (+1/-1 assumed)
    cutoff : int or float
        if int: number of features to return with smallest pvalues
        if float: threshold pvalue to return as features

    Output
    -------
    idx_fs : ndarray
        index locations of the features selected
        (length n_features if n_features = int)

    Example
    --------
    >>> idx_fs, pval_fs = ttest_feature_sel(X,y, cutoff = 0.05, dataFrame=False)
    >>> X_fs = X[:,idx_fs] # feature selection

    Or you can return results as pandas DataFrame (default)

    >>> ttest_fs = ttest_feature_sel(X,y,cutoff = 50)
    >>> X_fs = X[:,ttest_fs.index]

    """
    # sometimes labels ytrue, ypred \in {0,1}....map to {-1,+1}
    if np.array_equal(np.unique(y), np.array([0,1])):
        y = (y*2)-1

    _, pval = sp.stats.ttest_ind( X[y==+1,:], X[y==-1,:] )
    idx_sort = pval.argsort()

    if isinstance(cutoff, int):
        # return indices of top n_features ("top" = smalleset pvalues)
        idx = idx_sort[:cutoff]
    elif isinstance(cutoff, float):
        assert 0. < cutoff and cutoff < 1., \
            'floating value must be between 0 and 1'

        # return indices that is below a specified pvalue level
        idx = np.where(pval<cutoff)[0]

    # pvalues of the selected features
    pval_idx = pval[idx]

    # experimental: return as dataFrame
    if dataFrame:
        return pd.DataFrame({'index':idx, 'pvalue':pval_idx})
    else:
        return idx, pval_idx


def balance_binary_data(y,random_state=None):
    """ Get mask for balancing data via random subsampling on the dominant class

    **(Created 11/10/2015)**

    Parameters
    ----------
    y : array-like, [n_samples]
        Label vector containing labels of +1 or -1

    random_state : None, int or RandomState
        Pseudo-random number generator state used for random
        sampling. If None, use default numpy RNG for shuffling

    Returns
    -------
    mask : boolean ndarray, [n_samples]
        Binary mask vector you can apply on the label vector and data matrix,
        so that the number of +1 and -1 examples are equal.

    Example
    ---------
    >>> mask = tw.balance_binary_data(y,random_state=rng)
    >>> # drop indices
    >>> Xbal,ybal = X[mask], y[mask]
    >>> df_bal = df.iloc[mask,:].reset_index(drop=True)
    >>> df_disposed = df.iloc[~mask,:]

    Some snippets

    >>> idx_disposed = np.where(mask==False)[0]
    >>> idx_kept     = np.where(mask==True)[0]
    """
    rng = sklearn.utils.check_random_state(random_state)

    idx_pos = np.where(y==+1)[0]
    idx_neg = np.where(y==-1)[0]
    num_pos = (y==+1).sum()
    num_neg = (y==-1).sum()

    if num_pos > num_neg:
        # dispose random subjects from positive class to balance data
        idx_subsamp = rng.permutation(num_pos)[num_neg:]

        # data to dispose
        idx_dispose = idx_pos[idx_subsamp]
    elif num_pos < num_neg:
        # dispose random subjects from negative class to balance data
        idx_subsamp = rng.permutation(num_neg)[num_pos:]

        # data to dispose
        idx_dispose = idx_neg[idx_subsamp]
    else:
        # data already balanced; do nothing
        print "Data already balanced.  Return all indices"
        mask_to_keep = np.ones( y.shape[0], dtype=bool)
        return mask_to_keep

    # create binary mask
    mask_to_keep = np.ones( y.shape[0], dtype=bool)
    mask_to_keep[idx_dispose] = False
    return mask_to_keep
#%%--- grid search routines ---#
def grid_search_glmnet(X,y,clf,param_grid, cv=None, verbose=0,refit=True,
                       criterion='acc', return_full=False):
    """
        param_grid = {'alpha':10.**np.arange(-1,10,1),
                      'lambdas':10.**np.arange(1,-12,-1)}
    """
    from sklearn.cross_validation import StratifiedKFold
    if cv is None:
        print "Set to default 3-fold Stratified cross validation"
        cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=0)
    elif isinstance(cv,int):
        print "Set to {}-fold Stratified cross validation".format(cv)
        cv = StratifiedKFold(y, n_folds=cv, shuffle=True, random_state=0)

    alpha_grid  = param_grid['alpha']
    lambda_grid = param_grid['lambdas']

    n_alphas  = len(alpha_grid)
    n_lambdas = len(lambda_grid)

    acc_grid = np.zeros( (n_alphas, n_lambdas) )
    auc_grid = np.zeros( (n_alphas, n_lambdas) )
    f1_grid  = np.zeros( (n_alphas, n_lambdas) )
    bsr_grid = np.zeros( (n_alphas, n_lambdas) )
    nnz_mean_grid = np.zeros( (n_alphas, n_lambdas) )
    nnz_std_grid  = np.zeros( (n_alphas, n_lambdas) )

    start_time = time.time()
    for ialpha, alpha in enumerate(alpha_grid):
        clf.set_params(alpha=alpha)
        if verbose > 0:
            print("    (ialpha = {:2} out of {:2})".format(ialpha+1,n_alphas)),
            print_time(start_time)
        ypred=[]
        ytrue=[]
        score=[]
        nnz_inner = []
        for icv, (itr,its) in enumerate(cv):
            Xtr, Xts = X[itr], X[its]
            ytr, yts = y[itr], y[its]

            clf.fit(Xtr, ytr,lambdas=lambda_grid)
            ytrue.append(yts)

            if clf.out_n_lambdas_ != n_lambdas:
                #raise Exception('Something screwed up internally.  Exit code')
                print "**********   ",
                print "Something screwed up internally in glmnet",
                print "Output 0 for everything",
                print "   **********"
                sys.stdout.flush()
                ypred.append(np.zeros((Xts.shape[0],n_lambdas)))
                score.append(np.zeros((Xts.shape[0],n_lambdas)))
                nnz_inner.append(np.zeros(n_lambdas))
            else:
                # predict
                ypred.append(np.sign(clf.predict(Xts)))
                score.append(clf.decision_function(Xts))
                nnz_inner.append( (clf.coef_ != 0).sum(axis=0) )
        #******** END OF CV LOOP  ************#
        # convert list into ndarray
        ypred = np.concatenate(ypred)
        ytrue = np.concatenate(ytrue)
        score = np.concatenate(score)
        nnz_inner = np.vstack(nnz_inner)

        nnz_mean_grid[ialpha,:] = nnz_inner.mean(axis=0)
        nnz_std_grid[ialpha,:]  = nnz_inner.std(axis=0)

        for ilambda, _ in enumerate(lambda_grid):
            acc_grid[ialpha,ilambda] = sklearn.metrics.accuracy_score(ytrue,ypred[:,ilambda])
            auc_grid[ialpha,ilambda] = sklearn.metrics.roc_auc_score(ytrue,score[:,ilambda])
            f1_grid[ialpha,ilambda]  = sklearn.metrics.f1_score(ytrue,ypred[:,ilambda])
            bsr_grid[ialpha,ilambda] = clf_summary_short(ytrue,ypred[:,ilambda])[['TPR','TNR']].values.mean()
    #*** end of ialpha loop ***#
    #=========================================================================#
    # get the best parameter based on some user-specified criterion
    # ("ties" in the maximum value are handled by taking the indices of the
    #   first occurence of the tie)
    #=========================================================================#
    # select criterion for model selection (default: 'acc')
    criterion = criterion.upper() # <- for case insensitivity
    if criterion == 'ACC':
        crit_grid = acc_grid
    elif criterion == 'AUC':
        crit_grid = auc_grid
    elif criterion == 'F1':
        crit_grid = f1_grid
    elif criterion == 'BSR':
        crit_grid = bsr_grid

    # indices of best location (based on user-specified criterion)
    imax1,imax2 = np.unravel_index(crit_grid.argmax(), crit_grid.shape)

    # return all ties (if they exist)
    idx_best_ties = argmax_ties(acc_grid)

    # the best tuning parmaeters
    alpha_best  = param_grid['alpha'][imax1]
    lambda_best = param_grid['lambdas'][imax2]

    # return best parameters as dict
    param_best = {'alpha': alpha_best,
                  'lambda':lambda_best}

    # summary of CV classification
    summary = {#'ypred':ypred,
               #'ytrue':ytrue,
               #'score':score,
               'acc_grid': acc_grid,
#               'auc_grid': auc_grid,
#               'bsr_grid': bsr_grid,
#               'f1_grid': f1_grid,
               'idx_best1': imax1,
               'idx_best2': imax2,
               'idx_best_ties':idx_best_ties,
               'acc_best':acc_grid.max(),
               'total_time':time.time()-start_time,}

    if return_full:
        # add extra info
        summary['auc_grid'] = auc_grid
        summary['bsr_grid'] = bsr_grid
        summary['f1_grid'] = f1_grid
        summary['nnz_mean_grid'] = nnz_mean_grid
        summary['nnz_std_grid']  = nnz_std_grid

    # optional refit (default: True)
    if refit:
        clf.set_params(alpha=alpha_best)
        #| apparently i need a dummy variable for "lambdas" (can't pass a single argument?)
        clf.fit(Xtr,ytr,lambdas=np.array([lambda_best,100.,10.,1.,1e-1,1e-2]))

        #| remove the 2nd dummy lambda result
        clf.coef_ = clf.coef_[:,0]
        clf.intercepts_ = clf.intercepts_[:,0]
        clf.out_lambdas_ = clf.out_lambdas_[0]
        clf.null_dev_ = clf.null_dev_[0]
        clf.exp_dev_ = clf.exp_dev_[0]
        clf.out_n_lambdas_ = 1

        return clf, crit_grid, param_best, summary
    else:
        return crit_grid, param_best, summary

def grid_search_clf(X,y,clf,param_grid, **kwargs):
    """ My silly wrapper that determines the number of parameters to tune.

    Source code
    -------------

    .. code:: python

        if len(param_grid) is 1:
            return grid_search_clf_1d(X,y,clf,param_grid,**kwargs)
        elif len(param_grid) is 2:
            return grid_search_clf_2d(X,y,clf,param_grid,**kwargs)
        elif len(param_grid) is 3:
            return grid_search_clf_3d(X,y,clf,param_grid,**kwargs)

    Parameters
    -----------
    X : ndarray
        design matrix
    y : ndarray
        label vector of +1,-1
    clf : object
        Classifier object having the ``.fit`` and ``.predict`` method
    param_grid : dict
        - Key indicates the name of the grid parameter (to be set via the
          ``.set_params()`` method)
        - Value indicates the grid-range of the corresponding parameter
    cv : int or cross-validation generator, optional (default=None)
        If int, it is the number of folds.
        If None, 3-fold Stratified cross-validation is performed by default.
        Specific cross-validation objects can also be passed, see
        `sklearn.cross_validation module` for details.
    verbose : int, default=0
        Controls verbosity of output
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.

    Returns
    ---------
    clf_tuned : clf
        Tuned classifier (output suppressed if ``refit=False``)
    acc_grid : ndarray
        (n_param1 x n_param2) array of CV-accuracy at different values of tuning parameter
    param_best : dict
        dictionary containing key-value pair of the best parameter information,
        so we can simply set clf.set_params(**param_best) to set the classifier
        at the tuned form
    summary : dict
        dictionary containing "ytrue", "ypred", and "score", and other misc
        informatino (eg, index locations of the best parameters)

    Usage
    --------
    >>> clf_tuned, acc_grid, param_best, summary = grid_search_clf(X,y,clf,param_grid)

    If you want to skip refitting...

    >>> acc_grid, param_best, summary = grid_search_clf(X,y,clf,param_grid,refit=False)
    """
    if len(param_grid) is 1:
        return grid_search_clf_1d(X,y,clf,param_grid,**kwargs)
    elif len(param_grid) is 2:
        return grid_search_clf_2d(X,y,clf,param_grid,**kwargs)
    elif len(param_grid) is 3:
        return grid_search_clf_3d(X,y,clf,param_grid,**kwargs)


def grid_search_clf_1d(X,y, clf, param_grid, cv=None, verbose=0,refit=True,
                       criterion='acc', return_full=False, is_sparse=False):
    """ 1d gridsearch for binary classifier with ``fit`` and ``predict`` method

    Parameters
    -----------
    X : ndarray
        design matrix
    y : ndarray
        label vector of +1,-1
    clf : object
        Classifier object having the ``.fit`` and ``.predict`` method
    param_grid : dict
        - Key indicates the name of the grid parameter (to be set via the
          ``.set_params()`` method)
        - Value indicates the grid-range of the corresponding parameter
    cv : int or cross-validation generator, optional (default=None)
        If int, it is the number of folds.
        If None, 3-fold Stratified cross-validation is performed by default.
        Specific cross-validation objects can also be passed, see
        `sklearn.cross_validation module` for details.
    verbose : int, default=0
        Controls verbosity of output
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
    criterion : string (default='acc')
        Criterion score for the gridsearch model selection to optimize over.
        Currently supported criterion: ``'acc' (default), 'auc', 'f1', 'bsr'``
    return_full : bool (default=False) **Added 11/04/2015**
        Include extra info such as ``auc_grid, bsr_grid, f1_grid`` in the
        ``summary`` dict
    is_sparse : bool (default=False) **Added 11/04/2015**
        Flag indicating if classifier induces sparsity.  If set to true, ``summary``
        will return ``nnz_grid`` for number of nonzero features selected at that
        hyperparameter setup.

    Returns
    ---------
    clf_tuned : clf
        Tuned classifier (output suppressed if ``refit=False``)
    crit_grid : ndarray
        (n_param,) array of CV-criterion values at different values of tuning parameter.
        **Criterion** here is specified by the parameter **criterion** (default 'acc', accuracy)
    param_best : dict
        dictionary containing key-value pair of the best parameter information,
        so we can simply set clf.set_params(**param_best) to set the classifier
        at the tuned form
    summary : dict
        dictionary containing "ytrue", "ypred", and "score", and other misc
        informatino (eg, index locations of the best parameters)

    Development
    -----------
    See ``~/tak-ace-ibis/python/pnc/protocodes/1025_proto_grid_search_clf_1d.py``

    Usage
    --------
    >>> clf_tuned, acc_grid, param_best, summary = grid_search_clf_1d(X,y,clf,param_grid)
    >>> acc_grid, param_best, summary = grid_search_clf_1d(X,y,clf,param_grid,refit=False)
    """
    from sklearn.cross_validation import StratifiedKFold
    if cv is None:
        print "Set to default 3-fold Stratified cross validation"
        cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=0)
    elif isinstance(cv,int):
        print "Set to {}-fold Stratified cross validation".format(cv)
        cv = StratifiedKFold(y, n_folds=cv, shuffle=True, random_state=0)

    param = param_grid.keys()[0]
    grid  = param_grid.values()[0]

    n_grid = len(grid)

    acc_grid = np.zeros((n_grid))
    auc_grid = np.zeros((n_grid))
    f1_grid  = np.zeros((n_grid))
    bsr_grid = np.zeros((n_grid)) # "balanced score rate"...simply (TPR+TNR)/2
    if is_sparse: nnz_grid = np.zeros((n_grid))
    start_time = time.time()
    for igrid, value in enumerate(grid):
        clf.set_params(**{param:value})

        if verbose > 0:
            print("    (grid = {:2} out of {:2})".format(igrid+1,n_grid)),
            print_time(start_time)

        ypred=[]
        ytrue=[]
        score=[]

        # tune loop
        for icv, (itr,its) in enumerate(cv):
            Xtr, Xts = X[itr], X[its]
            ytr, yts = y[itr], y[its]

            clf.fit(Xtr,ytr)
            ypred.append(clf.predict(Xts))
            ytrue.append(yts)
            # TODO: include conditional statement for functions with no "decision_function" method
            score.append(clf.decision_function(Xts))
        #--- cv loop complete; convert list to ndarray ---#
        ypred = np.concatenate(ypred)
        ytrue = np.concatenate(ytrue)
        score = np.concatenate(score)

        # evaluate accuracy
        acc_grid[igrid] = sklearn.metrics.accuracy_score(ytrue,ypred)
        auc_grid[igrid] = sklearn.metrics.roc_auc_score(ytrue,score)
        f1_grid[igrid]  = sklearn.metrics.f1_score(ytrue,ypred)
        bsr_grid[igrid] = clf_summary_short(ytrue,ypred)[['TPR','TNR']].values.mean()

        if is_sparse:
            nnz_grid[igrid] = np.count_nonzero(clf.coef_)

    #=========================================================================#
    # get the best parameter based on some user-specified criterion
    # ("ties" in the maximum value are handled by taking the indices of the
    #   first occurence of the tie)
    #=========================================================================#
    # select criterion for model selection (default: 'acc')
    criterion = criterion.upper() # <- for case insensitivity
    if criterion == 'ACC':
        crit_grid = acc_grid
    elif criterion == 'AUC':
        crit_grid = auc_grid
    elif criterion == 'F1':
        crit_grid = f1_grid
    elif criterion == 'BSR':
        crit_grid = bsr_grid

    # indices of best location (based on user-specified criterion)
    imax = crit_grid.argmax()

    # return all ties (if they exist)
    idx_best_ties = argmax_ties(crit_grid)

    # the best tuning parmaeters
    param_best = param_grid[param][imax]

    # return best parameters as dict
    param_best = {param:param_best}

    # summary of CV classification
    summary = {'ypred':ypred,
               'ytrue':ytrue,
               'score':score,
               'acc_grid': acc_grid,
#               'auc_grid': auc_grid,
#               'bsr_grid': bsr_grid,
#               'f1_grid': f1_grid,
               'idx_best': imax,
               'idx_best_ties':idx_best_ties,
               'acc_best':acc_grid.max(),
               'total_time':time.time()-start_time,}

    if return_full:
        # add extra info
        summary['auc_grid'] = auc_grid
        summary['bsr_grid'] = bsr_grid
        summary['f1_grid'] = f1_grid

    if is_sparse:
        summary['nnz_grid'] = nnz_grid

    # optional refit (default: True)
    if refit:
        clf.set_params(**param_best)
        clf.fit(X,y)
        return clf, crit_grid, param_best, summary
    else:
        return crit_grid, param_best, summary

def grid_search_clf_2d(X,y, clf, param_grid, cv=None, verbose=0,refit=True,
                       criterion='acc', return_full=False, is_sparse=False):
    """ 2d gridsearch for binary classifier with ``fit`` and ``predict`` method

    Parameters
    -----------
    X : ndarray
        design matrix
    y : ndarray
        label vector of +1,-1
    clf : object
        Classifier object having the ``.fit`` and ``.predict`` method
    param_grid : dict
        - Key indicates the name of the grid parameter (to be set via the
          ``.set_params()`` method)
        - Value indicates the grid-range of the corresponding parameter
    cv : int or cross-validation generator, optional (default=None)
        If int, it is the number of folds.
        If None, 3-fold Stratified cross-validation is performed by default.
        Specific cross-validation objects can also be passed, see
        `sklearn.cross_validation module` for details.
    verbose : int, default=0
        Controls verbosity of output
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
    criterion : string (default='acc')
        Criterion score for the gridsearch model selection to optimize over.
        Currently supported criterion: ``'acc' (default), 'auc', 'f1', 'bsr'``
    return_full : bool (default=False) **Added 11/04/2015**
        Include extra info such as ``auc_grid, bsr_grid, f1_grid`` in the
        ``summary`` dict
    is_sparse : bool (default=False) **Added 11/04/2015**
        Flag indicating if classifier induces sparsity.  If set to true, ``summary``
        will return ``nnz_grid`` for number of nonzero features selected at that
        hyperparameter setup.

    Returns
    ---------
    clf_tuned : clf
        Tuned classifier (output suppressed if ``refit=False``)
    crit_grid : ndarray
        (n_param1 x n_param2) array of CV-criterion at different values of tuning parameter
        **Criterion** here is specified by the parameter **criterion** (default 'acc', accuracy)
    param_best : dict
        dictionary containing key-value pair of the best parameter information,
        so we can simply set clf.set_params(**param_best) to set the classifier
        at the tuned form
    summary : dict
        dictionary containing "ytrue", "ypred", and "score", and other misc
        informatino (eg, index locations of the best parameters)

    Development script
    -------------
    See ``~/tak-ace-ibis/python/pnc/protocodes/1025_proto_grid_search_clf_2d.py``

    Usage
    --------
    >>> clf_tuned, acc_grid, param_best, summary = grid_search_clf_2d(X,y,clf,param_grid)
    >>> acc_grid, param_best, summary = grid_search_clf_2d(X,y,clf,param_grid,refit=False)
    """
    from sklearn.cross_validation import StratifiedKFold
    if cv is None:
        print "Set to default 3-fold Stratified cross validation"
        cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=0)
    elif isinstance(cv,int):
        print "Set to {}-fold Stratified cross validation".format(cv)
        cv = StratifiedKFold(y, n_folds=cv, shuffle=True, random_state=0)

    param1 = param_grid.keys()[0]
    param2 = param_grid.keys()[1]
    grid1  = param_grid.values()[0]
    grid2  = param_grid.values()[1]

    n_grid1 = len(grid1)
    n_grid2 = len(grid2)

    acc_grid = np.zeros((n_grid1,n_grid2))
    auc_grid = np.zeros((n_grid1,n_grid2))
    f1_grid  = np.zeros((n_grid1,n_grid2))
    bsr_grid = np.zeros((n_grid1,n_grid2)) # "balanced score rate"...simply (TPR+TNR)/2
    if is_sparse: nnz_grid = np.zeros((n_grid1,n_grid2))
    start_time = time.time()
    for ig1, val1 in enumerate(grid1):
        clf.set_params(**{param1:val1})
        if verbose > 0:
            print("    (grid_1 = {:2} out of {:2})".format(ig1+1,n_grid1)),
            print_time(start_time)
        for ig2, val2 in enumerate(grid2):
            if verbose > 1:
                print("        (grid_2 = {:2} out of {:2})".format(ig2+1,n_grid2)),
                print_time(start_time)
            clf.set_params(**{param2:val2})
            ypred=[]
            ytrue=[]
            score=[]
            for icv, (itr,its) in enumerate(cv):
                Xtr, Xts = X[itr], X[its]
                ytr, yts = y[itr], y[its]

                clf.fit(Xtr,ytr)
                ypred.append(clf.predict(Xts))
                ytrue.append(yts)
                # TODO: include conditional statement for functions with no "decision_function" method
                score.append(clf.decision_function(Xts))
            #--- cv loop complete; convert list to ndarray ---#
            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)
            score = np.concatenate(score)

            # evaluate accuracy
            acc_grid[ig1,ig2] = sklearn.metrics.accuracy_score(ytrue,ypred)
            auc_grid[ig1,ig2] = sklearn.metrics.roc_auc_score(ytrue,score)
            f1_grid[ig1,ig2]  = sklearn.metrics.f1_score(ytrue,ypred)
            bsr_grid[ig1,ig2] = clf_summary_short(ytrue,ypred)[['TPR','TNR']].values.mean()
            if is_sparse:
                nnz_grid[ig1,ig2] = np.count_nonzero(clf.coef_)
    #=========================================================================#
    # get the best parameter based on some user-specified criterion
    # ("ties" in the maximum value are handled by taking the indices of the
    #   first occurence of the tie)
    #=========================================================================#
    # select criterion for model selection (default: 'acc')
    criterion = criterion.upper() # <- for case insensitivity
    if criterion == 'ACC':
        crit_grid = acc_grid
    elif criterion == 'AUC':
        crit_grid = auc_grid
    elif criterion == 'F1':
        crit_grid = f1_grid
    elif criterion == 'BSR':
        crit_grid = bsr_grid

    # indices of best location (based on user-specified criterion)
    imax1,imax2 = np.unravel_index(crit_grid.argmax(), crit_grid.shape)

    # return all ties (if they exist)
    idx_best_ties = argmax_ties(acc_grid)

    # the best tuning parmaeters
    param1_best = param_grid[param1][imax1]
    param2_best = param_grid[param2][imax2]

    # return best parameters as dict
    param_best = {param1:param1_best,
                  param2:param2_best}

    # summary of CV classification
    summary = {'ypred':ypred,
               'ytrue':ytrue,
               'score':score,
               'acc_grid': acc_grid,
#               'auc_grid': auc_grid,
#               'bsr_grid': bsr_grid,
#               'f1_grid': f1_grid,
               'idx_best1': imax1,
               'idx_best2': imax2,
               'idx_best_ties':idx_best_ties,
               'acc_best':acc_grid.max(),
               'total_time':time.time()-start_time,}

    if return_full:
        # add extra info
        summary['auc_grid'] = auc_grid
        summary['bsr_grid'] = bsr_grid
        summary['f1_grid'] = f1_grid

    if is_sparse:
        summary['nnz_grid'] = nnz_grid

    # optional refit (default: True)
    if refit:
        clf.set_params(**param_best)
        clf.fit(X,y)
        return clf, crit_grid, param_best, summary
    else:
        return crit_grid, param_best, summary

def grid_search_clf_3d(X,y, clf, param_grid, cv=None, verbose=0,refit=True,
                       criterion='acc', return_full=False, is_sparse=False):
    """ 3d gridsearch for binary classifier with ``fit`` and ``predict`` method

    Parameters
    -----------
    X : ndarray
        design matrix
    y : ndarray
        label vector of +1,-1
    clf : object
        Classifier object having the ``.fit`` and ``.predict`` method
    param_grid : dict
        - Key indicates the name of the grid parameter (to be set via the
          ``.set_params()`` method)
        - Value indicates the grid-range of the corresponding parameter
    cv : int or cross-validation generator, optional (default=None)
        If int, it is the number of folds.
        If None, 3-fold Stratified cross-validation is performed by default.
        Specific cross-validation objects can also be passed, see
        `sklearn.cross_validation module` for details.
    verbose : int, default=0
        Controls verbosity of output
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
    criterion : string (default='acc')
        Criterion score for the gridsearch model selection to optimize over.
        Currently supported criterion: ``'acc' (default), 'auc', 'f1', 'bsr'``
    return_full : bool (default=False) **Added 11/04/2015**
        Include extra info such as ``auc_grid, bsr_grid, f1_grid`` in the
        ``summary`` dict
    is_sparse : bool (default=False) **Added 11/04/2015**
        Flag indicating if classifier induces sparsity.  If set to true, ``summary``
        will return ``nnz_grid`` for number of nonzero features selected at that
        hyperparameter setup.

    Returns
    ---------
    clf_tuned : clf
        Tuned classifier (output suppressed if ``refit=False``)
    crit_grid : ndarray
        (n_param1 x n_param2) array of CV-criterion at different values of tuning parameter
        **Criterion** here is specified by the parameter **criterion** (default 'acc', accuracy)
    param_best : dict
        dictionary containing key-value pair of the best parameter information,
        so we can simply set clf.set_params(**param_best) to set the classifier
        at the tuned form
    summary : dict
        dictionary containing "ytrue", "ypred", and "score", and other misc
        informatino (eg, index locations of the best parameters)

    Warning
    ---------
    I never tested this function (as of 10/26/2015)

    Usage
    --------
    >>> clf_tuned, acc_grid, param_best, summary = grid_search_clf_3d(X,y,clf,param_grid)
    >>> acc_grid, param_best, summary = grid_search_clf_3d(X,y,clf,param_grid,refit=False)
    """
    from sklearn.cross_validation import StratifiedKFold
    if cv is None:
        print "Set to default 3-fold Stratified cross validation"
        cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=0)
    elif isinstance(cv,int):
        print "Set to {}-fold Stratified cross validation".format(cv)
        cv = StratifiedKFold(y, n_folds=cv, shuffle=True, random_state=0)

    param1 = param_grid.keys()[0]
    param2 = param_grid.keys()[1]
    param3 = param_grid.keys()[2]
    grid1  = param_grid.values()[0]
    grid2  = param_grid.values()[1]
    grid3  = param_grid.values()[2]

    n_grid1 = len(grid1)
    n_grid2 = len(grid2)
    n_grid3 = len(grid3)

    acc_grid = np.zeros((n_grid1,n_grid2,n_grid3))
    auc_grid = np.zeros((n_grid1,n_grid2,n_grid3))
    f1_grid  = np.zeros((n_grid1,n_grid2,n_grid3))
    bsr_grid = np.zeros((n_grid1,n_grid2,n_grid3)) # "balanced score rate"...simply (TPR+TNR)/2
    if is_sparse: nnz_grid = np.zeros((n_grid1,n_grid2,n_grid3))
    start_time = time.time()
    for ig1, val1 in enumerate(grid1):
        clf.set_params(**{param1:val1})
        if verbose > 0:
            print("    (grid_1 = {:2} out of {:2})".format(ig1+1,n_grid1)),
            print_time(start_time)
        for ig2, val2 in enumerate(grid2):
            if verbose > 1:
                print("        (grid_2 = {:2} out of {:2})".format(ig2+1,n_grid2)),
                print_time(start_time)
            clf.set_params(**{param2:val2})
            for ig3, val3 in enumerate(grid3):
                if verbose > 2:
                    print("        (grid_3 = {:2} out of {:2})".format(ig3+1,n_grid3)),
                    print_time(start_time)
                clf.set_params(**{param3:val3})
                ypred=[]
                ytrue=[]
                score=[]
                for icv, (itr,its) in enumerate(cv):
                    Xtr, Xts = X[itr], X[its]
                    ytr, yts = y[itr], y[its]

                    clf.fit(Xtr,ytr)

                    ypred.append(clf.predict(Xts))
                    ytrue.append(yts)
                    # TODO: include conditional statement for functions with no "decision_function" method
                    score.append(clf.decision_function(Xts))
                #--- cv loop complete; convert list to ndarray ---#
                ypred = np.concatenate(ypred)
                ytrue = np.concatenate(ytrue)
                score = np.concatenate(score)

                # evaluate accuracy
                acc_grid[ig1,ig2,ig3] = sklearn.metrics.accuracy_score(ytrue,ypred)
                auc_grid[ig1,ig2,ig3] = sklearn.metrics.roc_auc_score(ytrue,score)
                f1_grid[ig1,ig2,ig3]  = sklearn.metrics.f1_score(ytrue,ypred)
                bsr_grid[ig1,ig2,ig3]  = clf_summary_short(ytrue,ypred)[['TPR','TNR']].values.mean()
                if is_sparse:
                    nnz_grid[ig1,ig2] = np.count_nonzero(clf.coef_)
    #=========================================================================#
    # get the best parameter based on some user-specified criterion
    # ("ties" in the maximum value are handled by taking the indices of the
    #   first occurence of the tie)
    #=========================================================================#
    # select criterion for model selection (default: 'acc')
    criterion = criterion.upper() # <- for case insensitivity
    if criterion == 'ACC':
        crit_grid = acc_grid
    elif criterion == 'AUC':
        crit_grid = auc_grid
    elif criterion == 'F1':
        crit_grid = f1_grid
    elif criterion == 'BSR':
        crit_grid = bsr_grid

    # indices of best location (based on user-specified criterion)
    imax1,imax2,imax3 = np.unravel_index(crit_grid.argmax(), crit_grid.shape)

    # return all ties (if they exist)
    idx_best_ties = argmax_ties(acc_grid)

    # the best tuning parmaeters
    param1_best = param_grid[param1][imax1]
    param2_best = param_grid[param2][imax2]
    param3_best = param_grid[param3][imax3]

    # return best parameters as dict
    param_best = {param1:param1_best,
                  param2:param2_best,
                  param3:param3_best,}

    # summary of CV classification
    summary = {'ypred':ypred,
               'ytrue':ytrue,
               'score':score,
               'acc_grid': acc_grid,
#               'auc_grid': auc_grid,
#               'f1_grid': f1_grid,
#               'bsr_grid': bsr_grid,
               'idx_best1': imax1,
               'idx_best2': imax2,
               'idx_best3': imax3,
               'idx_best_ties':idx_best_ties,
               'acc_best':acc_grid.max(),
               'total_time':time.time()-start_time,}

    if return_full:
        # add extra info
        summary['auc_grid'] = auc_grid
        summary['bsr_grid'] = bsr_grid
        summary['f1_grid'] = f1_grid

    if is_sparse:
        summary['nnz_grid'] = nnz_grid

    # optional refit (default: True)
    if refit:
        clf.set_params(**param_best)
        clf.fit(X,y)
        return clf, crit_grid, param_best, summary
    else:
        return crit_grid, param_best, summary

# deprecated.  built my own wrapper class for this.
def __grid_search_rbf_svm_2d(X,y, param_grid, clf=None, cv=None, verbose=0,
                           refit=True, criterion=True):
    """ Gridsearch for rbf kernel with kernels precomputed.

    Usage
    --------
    >>> clf_tuned, acc_grid, param_best, summary = grid_search_rbf_svm_2d(X,y,param_grid,cv=cv)

    Precomputing the kernel makes things so much faster in libsvm.

    Input/Output and internals of the code is nearly identical to that of
    ``grid_search_clf_2d``

    Parameters (only different with ``grid_search_clf_2d``)
    -----------
    clf : default svm
        just included this as possible parameter, since maybe the flexibility
        of having this as input may be helpful...if not, remove in future

    Thoughts
    ----------
    Having internal code nearly identical really goes against the principle
    of DRY coding....with class inheritance I see how I can avoid this, but
    is there any solution for *not repeating my code* with methods?

    Development script
    -------------
    See ``~/tak-ace-ibis/python/pnc/protocodes/1025_proto_grid_search_clf_2d.py``
    """
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics.pairwise import rbf_kernel
    if clf is None:
        from sklearn.svm import SVC
        clf=SVC(kernel='precomputed')

    if cv is None:
        #cv_was_None = True
        cv = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=0)
    elif isinstance(cv,int):
        #cv_was_None = True
        cv = StratifiedKFold(y, n_folds=cv, shuffle=True, random_state=0)
    else:
        pass
        #cv_was_None = False

    param1 = param_grid.keys()[0]
    param2 = param_grid.keys()[1]
    grid1  = param_grid.values()[0]
    grid2  = param_grid.values()[1]

    n_grid1 = len(grid1)
    n_grid2 = len(grid2)

    acc_grid = np.zeros((n_grid1,n_grid2))
    auc_grid = np.zeros((n_grid1,n_grid2))
    f1_grid  = np.zeros((n_grid1,n_grid2))
    bsr_grid = np.zeros((n_grid1,n_grid2)) # "balanced score rate"...simply (TPR+TNR)/2
    start_time = time.time()
    for iC, C in enumerate(grid1):
        clf.set_params(**{param1:C})
        if verbose > 0:
            print("    (C = {:2} out of {:2})".format(iC+1,n_grid1)),
            print_time(start_time)
        for igam, gamma in enumerate(grid2):
            if verbose > 1:
                print("        (gamma = {:2} out of {:2})".format(igam+1,n_grid2)),
                print_time(start_time)
            #clf.set_params(**{param2:val2})
            ypred=[]
            ytrue=[]
            score=[]
            for icv, (itr,its) in enumerate(cv):
                Xtr, Xts = X[itr], X[its]
                ytr, yts = y[itr], y[its]

                # precompute kernel matrix
                Ktr = rbf_kernel(Xtr, gamma=gamma)
                Kts = rbf_kernel(Xts, Xtr, gamma=gamma)

                clf.fit(Ktr,ytr)
                ypred.append(clf.predict(Kts))
                ytrue.append(yts)
                # TODO: include conditional statement for functions with no "decision_function" method
                score.append(clf.decision_function(Kts))
                #print ytr.shape[0], clf.dual_coef_.shape, C,gamma

            #--- cv loop complete; convert list to ndarray ---#
            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            # evaluate accuracy
            acc_grid[iC,igam] = sklearn.metrics.accuracy_score(ytrue,ypred)
            auc_grid[iC,igam] = sklearn.metrics.roc_auc_score(ytrue,score)
            f1_grid[iC,igam]  = sklearn.metrics.f1_score(ytrue,ypred)
            bsr_grid[iC,igam] = clf_summary_short(ytrue,ypred)[['TPR','TNR']].values.mean()

    #=========================================================================#
    # get the best parameter based on some user-specified criterion
    # ("ties" in the maximum value are handled by taking the indices of the
    #   first occurence of the tie)
    #=========================================================================#
    # select criterion for model selection (default: 'acc')
    criterion = criterion.upper() # <- for case insensitivity
    if criterion == 'ACC':
        crit_grid = acc_grid
    elif criterion == 'AUC':
        crit_grid = auc_grid
    elif criterion == 'F1':
        crit_grid = f1_grid
    elif criterion == 'BSR':
        crit_grid = bsr_grid

    # indices of best location (based on user-specified criterion)
    imax1,imax2 = np.unravel_index(crit_grid.argmax(), crit_grid.shape)

    # return all ties (if they exist)
    idx_best_ties = argmax_ties(acc_grid)

    # the best tuning parmaeters
    param1_best = param_grid[param1][imax1]
    param2_best = param_grid[param2][imax2]

    # return best parameters as dict
    param_best = {param1:param1_best,
                  param2:param2_best}

    # summary of CV classification
    summary = {'ypred':ypred,
               'ytrue':ytrue,
               'score':score,
               'acc_grid': acc_grid,
               'auc_grid': auc_grid,
               'f1_grid': f1_grid,
               'idx_best1': imax1,
               'idx_best2': imax2,
               'idx_best_ties':idx_best_ties,
               'acc_best':acc_grid.max(),
               'total_time':time.time()-start_time,}

    # optional refit (default: True)
    if refit:
        # compute kernel matrix
        K = rbf_kernel(X,gamma=param2_best)
        clf.set_params(C=param1_best)
        clf.fit(K,y)
        #clf_tuned=SVC(kernel='precomputed')
        #clf_tuned.set_params(C=param1_best)
        #clf_tuned.fit(K,y)
        #summary['clf_tuned'] = clf_tuned
        return clf, acc_grid, param_best, summary
    else:
        return acc_grid, param_best, summary


def show_cv_gridsearch(ax,acc_grid, param_grid, iouter,clf_name=''):
    """ Display outercv gridsearch result (creatd 10/31/2015)

    Details
    -------
    - Before launching my "official" run of the script, I like to get an idea
      on what parameter values are within reasonable values of search range
    - seeing ``flag_show_plot`` below will give me an plot of the couter-cv
      accuracy...
    - when content with the "search-range", set flag to False, and launch script

    Usecase
    -------
    See for example:
    ``~/tak-ace-ibis/python/analysis/tbi/nested_cv_conn/##try_outercv_tbi.py``
    """
    if np.ndim(acc_grid) == 1:
        if len(ax.lines) == 0:
            # no lines drawn yet.  assign xlabels and what nots
            xlen=param_grid.values()[0].shape[0]
            plt.xticks(range(xlen))
            plt.gca().set_xlabel(param_grid.keys()[0])
            plt.gca().set_xticklabels(param_grid.values()[0], rotation=30,size=12)
            plt.title(clf_name)
        ax.plot(acc_grid,label='CV={}, max={:.3f}'.format(iouter+1,acc_grid.max()))
        plt.legend(fontsize=10)
        plt.grid('on')

    elif np.ndim(acc_grid) == 2:
        #plt.clf()
        figure('f')
        plt.title(clf_name+' (CV={:2}, max={:.3f})'.format(iouter+1,acc_grid.max()))
        ylabel = param_grid.keys()[0]
        xlabel = param_grid.keys()[1]
        ytick_label = param_grid.values()[0]
        xtick_label = param_grid.values()[1]
        fig_set_geom('f')
#        imgridsearch(acc_grid,xtick_label=xtick_label,ytick_label=ytick_label,
#                        show_max=True,fontsize=14,show_cbar=True)
        imgridsearch(acc_grid,xtick_label=xtick_label,ytick_label=ytick_label,
                        vmin=0.55, vmax=acc_grid.max(),show_max=True,fontsize=14,show_cbar=True)
        plt.gca().set_xlabel(xlabel)
        plt.gca().set_ylabel(ylabel)
    elif np.ndim(acc_grid) == 3:
        figure('f')
        show_3d_gridsearch(acc_grid,param_grid,iouter,clf_name)
        plt.title(clf_name+' (CV={:2}, max={:.3f})'.format(iouter+1,acc_grid.max()),
                  fontsize=16)
#    """TO DO: Insert 3d bubble plot i created a while ago"""

    # need these for the plots to update
    plt.draw()
    plt.pause(0.5)
    #plt.waitforbuttonpress()

    """ Return fig: I need this for the 1d line plot to overlay on top of each other"""
    return ax

def show_3d_gridsearch(acc_grid, param_grid, iouter,clf_name=''):
    """ Unrefined 3d scatter plot of cv accuracy....need major cleanup (10/31/2015)

    Code from my old ipynb script:
    ``+++pnc_rbfsvm_ttest_nested_cv.ipynb``
    """
    xlabel = param_grid.keys()[0]
    ylabel = param_grid.keys()[1]
    zlabel = param_grid.keys()[2]
    grid1 = param_grid.values()[0]
    grid2 = param_grid.values()[1]
    grid3 = param_grid.values()[2]
    xrange = range(len(param_grid.values()[0]))
    yrange = range(len(param_grid.values()[1]))
    zrange = range(len(param_grid.values()[2]))

    from mpl_toolkits.mplot3d import Axes3D
    try:
        ax = plt.gcf().add_subplot(111, projection='3d')
    except:
        ax = plt.figure(figsize=(9,9)).add_subplot(111, projection='3d')

    for ig1,val1 in enumerate(xrange):
        for ig2, val2 in enumerate(yrange):
            #gam = np.log10(gam).astype(int)
            for ig3, val3 in enumerate(zrange):
                sc=ax.scatter(val1,val2,val3,marker='o', vmin=0.5,vmax=acc_grid.max(), s=255,
                              c=acc_grid[ig1,ig2,ig3],cmap='hot')
                if acc_grid[ig1,ig2,ig3]==acc_grid.max():
                    ax.scatter(val1,val2,val3,marker='x',s=111,c='k',linewidths=3)

    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_zlabel(zlabel,fontsize=12)
    ax.set_xticks(xrange)
    ax.set_yticks(yrange)
    ax.set_zticks(zrange)
    ax.set_xticklabels(grid1,fontsize=8)
    ax.set_yticklabels(grid2,fontsize=8)
    ax.set_zticklabels(grid3,fontsize=8)
    plt.colorbar(sc)
    return ax

""" BELOW IS A FAILED VERION
#    xlabel = param_grid.keys()[0]
#    ylabel = param_grid.keys()[1]
#    zlabel = param_grid.keys()[2]
#    grid1 = param_grid.values()[0]
#    grid2 = param_grid.values()[1]
#    grid3 = param_grid.values()[2]
#
#    xlen=param_grid.values()[0].shape[0]
#    ylen=param_grid.values()[1].shape[0]
#    zlen=param_grid.values()[2].shape[0]
#    print xlen,ylen,zlen
#    xticks = range(xlen)
#    yticks = range(ylen)
#    zticks = range(zlen)
#
#    from mpl_toolkits.mplot3d import Axes3D
#    try:
#        ax = plt.gcf().add_subplot(111, projection='3d')
#    except:
#        ax = plt.figure(figsize=(9,9)).add_subplot(111, projection='3d')
#
#    for ig1,val1 in enumerate(grid1):
#        for ig2, val2 in enumerate(grid2):
#            #gam = np.log10(gam).astype(int)
#            for ig3, val3 in enumerate(grid3):
#                sc=ax.scatter(val1,val2,val3,marker='o', vmin=0.5,vmax=0.8, s=255,
#                              c=acc_grid[ig1,ig2,ig3],cmap='hot')
#                if acc_grid[ig1,ig2,ig3]==acc_grid.max():
#                    ax.scatter(val1,val2,val3,marker='x',s=111,c='k',linewidths=3)
##    ax.set_xticks(xticks)
##    ax.set_yticks(yticks)
##    ax.set_zticks(zticks)
##    ax.set_xticklabels(grid1)
##    ax.set_yticklabels(grid2)
##    ax.set_zticklabels(grid3)
#    ax.set_xticklabels(xticks)
#    ax.set_yticklabels(yticks)
#    ax.set_zticklabels(zticks)
#    ax.set_xlabel(xlabel)
#    ax.set_ylabel(ylabel)
#    ax.set_zlabel(zlabel)
#    plt.colorbar(sc)
#    plt.title(clf_name+' (CV=%d)'%iouter)
#    return ax
"""
#%%============= classification utility functions =============================
def clf_summary(ytrue, ypred, multiIndex=True, full=True):
    """ get dataframe representation of binary classification summary

    See ^proto-clf_summary.ipynb for the utility of this shiat
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    """
    # sometimes labels ytrue, ypred \in {0,1}....map to {-1,+1}
    if np.array_equal(np.unique(ytrue), np.array([0,1])):
        ytrue = (ytrue*2)-1
    if np.array_equal(np.unique(ypred), np.array([0,1])):
        ypred = (ypred*2)-1

    # 'p' for positive, 'n' for negative
    idxp = np.where(ytrue == +1)[0]
    idxn = np.where(ytrue == -1)[0]

    # TP = true positives
    # TN = true negatives
    # FP = false positives
    # FN = false negatives
    TP = np.sum(ytrue[idxp] == ypred[idxp])
    TN = np.sum(ytrue[idxn] == ypred[idxn])
    FP = np.sum(ytrue[idxp] != ypred[idxp])
    FN = np.sum(ytrue[idxn] != ypred[idxn])

    """ define a function to handle division by zero annoyance """
    def div_handle(num, den):
        """ num = numerator, den = denominator"""
        try:
            # 1.* to convert to float for division
            val = 1.*num/den
        except ZeroDivisionError:
            val = np.nan
        return val

    # TPR = true positive rate  (aka sensitivity, recall, hit rate, power, detection rate)
    # TNR = true negative rate  (aka specificity)
    # FPR = false positive rate (aka size, type I error rate)
    # FNR = false negative rate (aka miss rate, type II error rate, 1-TPR)
    # PPV = positive predictive value (aka precision)
    # NPV = negative predictive value
    TPR = div_handle(TP,TP+FN)
    TNR = div_handle(TN,TN+FP)
    FPR = div_handle(FP,TP+TN)
    FNR = div_handle(FN,FN+TP)
    PPV = div_handle(TP,TP+FP)
    NPV = div_handle(TN,TN+FN)
    F1  = div_handle(2.*TP, 2*TP+FP+FN)

    # 1.* to convert to float for division
    # TPR = 1.*TP/(TP + FN)
    # TNR = 1.*TN/(TN + FP)
    # FPR = 1.*FP/(TP + TN)
    # FNR = 1.*FN/(FN + TP)
    # PPV = 1.*TP/(TP + FP)
    # NPV = 1.*TN/(TN + FN)
    # F1 = 2. * TP / (2*TP + FP + FN)
    ACC = 1.*np.sum(ytrue == ypred)/len(ytrue)
    # summary = {'ACC':ACC,
    #                         'TPR':TPR,
    #                         'TNR':TNR,
    #                         'TP' :TP,
    #                         'TN' :TN,
    #                         'FP' :FP,
    #                         'FN' :FN,
    #                         'FPR':FPR,
    #                         'FNR':FNR,
    #                         'PPV':PPV,
    #                         'NPV':NPV,
    #                         'F1':F1}


    # let's try multi-index
    if full:
        summary     = [ACC, TPR, TNR, FPR, FNR, F1, PPV, NPV, TP, TN, FP, FN, TP+FP, TN+FN, len(ytrue)]
        score_type  = ['ACC','TPR','TNR','FPR','FNR','F1','PPV','NPV','TP','TN','FP','FN','P','N','ALL']
    else:
        summary     = [ACC, TPR, TNR, FPR, FNR, F1, PPV, NPV]
        score_type  = ['ACC','TPR','TNR','FPR','FNR','F1','PPV','NPV']

    if multiIndex and full:
        """http://pandas.pydata.org/pandas-docs/stable/advanced.html"""
        #| currently have 11 items on this lsit
        # summary = [ACC, TPR, TNR, TP, TN, FP, FN, FPR, FNR, PPV, NPV]
        arrays = [np.array(['scores']*8+['counts']*7),
                  np.array(score_type)]
        summary = pd.DataFrame(pd.Series(summary, index=arrays),columns = ['value'])
    else:
        summary = pd.Series(data=summary, index=score_type, name='clf')
        summary = pd.DataFrame(summary)

    # (11/01/2015) values like TPR, FNR may be NaN (division by zero), so replace with zero
    summary = summary.fillna(0)
    #| ditched below...dict results in arbitrary ordering
    # summary = pd.DataFrame({'ACC':ACC,
    #                         'TPR':TPR,
    #                         'TNR':TNR,
    #                         'TP' :TP,
    #                         'TN' :TN,
    #                         'FP' :FP,
    #                         'FN' :FN,
    #                         'FPR':FPR,
    #                         'FNR':FNR,
    #                         'PPV':PPV,
    #                         'NPV':NPV,
    #                         'F1':F1}, index=['clf_summary'])
    return summary.T # <-added a transpose to make data access easier
#     f1 = metrics.f1_score(ytrue,ypred)

def clf_summary_short(ytrue, ypred,add_bsr_auc=False):
    """ get dataframe representation of binary classification summary

    See ^proto-clf_summary.ipynb for the utility of this shiat
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    **Update 02/16/2016**

    - ``add_bsr_auc`` option added
    - apply ``np.sign(ypred)`` to allow the 2nd argument to be
      classification score
    """
    score = ypred          # <- update 02/16/2016
    ypred = np.sign(ypred) # <- update 02/16/2016

    # sometimes labels ytrue, ypred \in {0,1}....map to {-1,+1}
    if np.array_equal(np.unique(ytrue), np.array([0,1])):
        ytrue = (ytrue*2)-1
    if np.array_equal(np.unique(ypred), np.array([0,1])):
        ypred = (ypred*2)-1

    # 'p' for positive, 'n' for negative
    idxp = np.where(ytrue == +1)[0]
    idxn = np.where(ytrue == -1)[0]

    TP = np.sum(ytrue[idxp] == ypred[idxp])
    TN = np.sum(ytrue[idxn] == ypred[idxn])
    FP = np.sum(ytrue[idxp] != ypred[idxp])
    FN = np.sum(ytrue[idxn] != ypred[idxn])

    """ define a function to handle division by zero annoyance """
    def div_handle(num, den):
        """ num = numerator, den = denominator"""
        try:
            # 1.* to convert to float for division
            val = 1.*num/den
        except ZeroDivisionError:
            val = np.nan
        return val

    TPR = div_handle(TP,TP+FN)
    TNR = div_handle(TN,TN+FP)
    ACC = 1.*np.sum(ytrue == ypred)/len(ytrue)

    if add_bsr_auc:
        BSR = clf_get_bsr(ytrue,ypred)
        AUC = clf_get_auc(ytrue,score)
        summary     = [ACC, TPR, TNR,BSR,AUC]
        score_type  = ['ACC','TPR','TNR','BSR','AUC']
    else:
        summary     = [ACC, TPR, TNR]
        score_type  = ['ACC','TPR','TNR']
    summary = pd.Series(data=summary, index=score_type, name='clf')
    summary = pd.DataFrame(summary).T

    # (11/01/2015) values like TPR, FNR may be NaN (division by zero), so replace with zero
    summary = summary.fillna(0)
    return summary

def clf_summary_array(ytrue, ypred, short_summary=False):
    """ Create array of DF summary (created 10/20/2015)

    Useful with glmnet.

    Parameters
    -----------
    ytrue : ndarray
        (n x 1) array of the true +1/-1 label
    ypred : ndarray
        (n x k) array of predicted label (k sets of predictions)
    short_summary : bool (default: False)
        only return ACC, TPR, TNR

    Output
    -------
    dataframe of clf summary

    Example
    -------
    >>> from glmnet_py import LogisticNet
    >>> lognet.fit(Xtrain, ytrain)
    >>> ypred = (lognet.predict(Xtest) > 0.5).astype(int)
    >>> df_clf_summary = clf_summary_array(ytest, ypred)
    """
    if short_summary:
        df_clf_summary = clf_summary_short(ytrue, ypred[:,0])
    else:
        df_clf_summary = clf_summary(ytrue, ypred[:,0])

    # keep appending rows on to the above result
    for i in range(1,ypred.shape[1]):
        if short_summary:
            df_clf_summary = pd.concat([df_clf_summary, clf_summary_short(ytrue, ypred[:,i])])
        else:
            df_clf_summary = pd.concat([df_clf_summary, clf_summary(ytrue, ypred[:,i])])

    df_clf_summary.reset_index(inplace=True)
    del df_clf_summary['index']

    # (11/01/2015) values like TPR, FNR may be NaN (division by zero), so replace with zero
    df_clf_summary = df_clf_summary.fillna(0)
    return df_clf_summary

def clf_summary_array2(ytrue, ypred, short_summary=False):
    """ Create array of DF summary (created 11/19/2015)

    Similar to clf_summary_array, but now ``ytrue`` is the same shape of ``ypred``
    (with columns indicating different sample sets or permutation of labels)

    Created this since I often use random subsampling to balance label, so
    ``ytrue`` will go through different permutations

    Parameters
    -----------
    ytrue : ndarray
        (n x k) array of the true +1/-1 label
    ypred : ndarray
        (n x k) array of predicted label (k sets of predictions)
    short_summary : bool (default: False)
        only return ACC, TPR, TNR

    Output
    -------
    dataframe of clf summary

    Example
    -------
    >>> from glmnet_py import LogisticNet
    >>> lognet.fit(Xtrain, ytrain)
    >>> ypred = (lognet.predict(Xtest) > 0.5).astype(int)
    >>> df_clf_summary = clf_summary_array(ytest, ypred)
    """
    if short_summary:
        df_clf_summary = clf_summary_short(ytrue[:,0], ypred[:,0])
    else:
        df_clf_summary = clf_summary(ytrue[:,0], ypred[:,0])

    # keep appending rows on to the above result
    for i in range(1,ypred.shape[1]):
        if short_summary:
            df_clf_summary = pd.concat([df_clf_summary,
                                        clf_summary_short(ytrue[:,i], ypred[:,i])
                                        ])
        else:
            df_clf_summary = pd.concat([df_clf_summary,
                                        clf_summary(ytrue[:,i], ypred[:,i])
                                        ])

    df_clf_summary.reset_index(inplace=True)
    del df_clf_summary['index']

    # (11/01/2015) values like TPR, FNR may be NaN (division by zero), so replace with zero
    df_clf_summary = df_clf_summary.fillna(0)
    return df_clf_summary
#%% === array/matrix manipulation ===
def test_mydist_vs_sklearn(n=800, p=500, n_rep=100):
    """ Timing test between my EDM generator and Scikit's

    Output
    ------
    Elapsed time:  0.99 seconds
    Elapsed time:  1.50 second
    """
    from sklearn.metrics import pairwise_distances
    start1=time.time()
    A = np.random.randn(n,p)
    for i in range(n_rep):
        _ = pairwise_distances(A)
    print_time(start1)

    start2=time.time()
    for i in range(n_rep):
        _ = dist_euc(A)
    print_time(start2)


@deprecated("Use sklearn's distance function instead (also run test_mydist_vs_sklearn)")
def dist_euc(X):
    """ Compute euclidean distance matrix (EDM) (10/15/2015)

    **WELP NEVER FUCKING MIND, SCIKIT HAD IT...JUST AS FAST (AND MOST LIKELY
    FAR MORE ROBUST THAN THIS CRAP)**

    Developed in ^1015-try-my-own-gridsearch-2d.ipynb

    Input
    -----
    X : ndarray
        (n x p) design matrix

    Output
    ------
    EDM : ndarray
        (n x n) Euclidean distance matrix

    In Matlab
    -----------
    >>> bsxfun(@plus, sum((A.^2),2), sum((B.^2),2)') - 2*(A*(B'));
    """
    a = np.sum(X**2, axis=1)[:, np.newaxis]
    c = X.dot(X.T)
    EDM = np.sqrt(a + a.T - 2. * c)

    # some diagonal component are NaNs...fix this
    np.fill_diagonal(EDM, 0)

    # remove nans
    EDM = np.nan_to_num(EDM)

    return EDM


# def dist_euc2(A,B):
#    """ Compute euclidean distance matrix (EDM) (10/15/2015)

#    Developed in ^1015-try-my-own-gridsearch-2d.ipynb

#    In Matlab
#    -----------
#    >>> bsxfun(@plus, sum((A.^2),2), sum((B.^2),2)') - 2*(A*(B'));
#    """
#    a = np.sum(A**2,axis=1)[:,np.newaxis]
#    b = np.sum(B**2,axis=1).T
#    c = A.dot(B.T)
#    EDM = np.sqrt(a + b - 2.*c)

#    # some diagonal component are NaNs...fix this
#    #np.fill_diagonal(EDM,0)

#    # remove nans
#    np.nan_to_num(EDM)

#    return EDM


def dvec(mat, upper=True):
    """ Extract lower-triangular portion of a symmetric matrix

    Bleh:

    - i never needed this function....scipy.spatial.distance.squareform does it

    Note:

    - Default: extract upper-triangular portion.
    - Since python is row-major oredering, this default of extracting the
      upper-triangular portion ensures consistency with my dvec function
      in Matlab
    """
    d = mat.shape[0]
    if upper:
        idx = np.triu_indices(d, +1)
    else:
        idx = np.tril_indices(d, -1)
    return mat[idx]

def get_hostname():
    """ host == 'sbia-pc125' on my work computer'
    """
    import socket
    host = socket.gethostname()
    return host

def argmax_array(W):
    """ Return tuple of indices of max location in an numpy array

    (Created 10/23/2015)

    Useful for multidimensional array (cuz i can never memorize below snippet)

    Example
    --------
    >>> imax,jmax = argmax_array(acc_grid)
    >>> assert( acc_grid[imax,jmax], acc_grid.max() )
    """
    return np.unravel_index(W.argmax(), W.shape)

def argmax_ties(W, return_argmax=False):
    """ Returns a list of tuple representing argmax of an ndarray with ties

    Created 10/23/2015

    Output
    -------
    idx_ties : list of tuples
        list of tuples, where tuple-length is the array-dimension, and
        list-length is the number of ties

    Credit
    ------
    Idea directly from http://stackoverflow.com/questions/17568612/
    how-to-make-numpy-argmax-return-all-occurences-of-the-maximum

    Example
    -------
    >>> W = np.zeros( (10,10) )
    >>> W[5,2]=100
    >>> W[8,3]=100
    >>> W[3,1]=100
    >>> idx_ties = argmax_ties(W)
    >>> W[idx_ties[0]] == W.argmax() # <- shall be true
    """
    winners = np.argwhere(W == np.amax(W))
    num_ties = winners.shape[0]

    # return as list of tuples
    output = []
    for i in range(num_ties):
        output.append(tuple(winners[i,:]))

    if return_argmax:
        return output, W.argmax()
    else:
        return output

def drop_rowcol(W, idx_drop):
    """ Simple function to drop the specified row/cols from a square matrix W

    Sometimes used in my connectome analysis

    Created 1030/2015

    Usage
    ------
    Suppose W is a 95 x 95 square matrix of connectome, and i want to remove
    nodes to obtain 86 x 86 matrix...do below

    >>> _, idx_drop = tw.get_nodes_dropped_from_95()
    >>> W = tw.drop_rowcol(W, idx_drop)

    """
    W = np.delete(W, idx_drop, axis=0) # first drop rows
    W = np.delete(W, idx_drop, axis=1) # then  drop cols
    return W
#%% === General util functions ===
class Struct(object):
    """ Created 10/14/2015

    http://stackoverflow.com/questions/1305532/convert-python-dict-to-object

    >>> args = {'a': 1, 'b': 2}
    >>> s = Struct(**args)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def show_data_balance(y):
    """Print data balance info for binary label vector y

    Assumes y = +1 or y = -1

    (Created 11/06/2015)
    """
    n_pos = (y==+1).sum()
    n_neg = (y==-1).sum()
    n_ratio = 1.*n_pos/len(y)
    print("(+1) {:3} subjects, (-1) {:3} subjects ({:.2f}% is +1)".
            format(n_pos,n_neg,100*n_ratio))


def myprint(stuff):
    """

    https://docs.python.org/2/library/inspect.html
    https://docs.python.org/2/library/inspect.html#the-interpreter-stack

    Examples
    ------------
    >>> myprint(digits.keys())
    digits.keys() = ['images', 'data', 'target_names', 'DESCR', 'target']
    >>> myprint(digits.images.shape)
    digits.images.shape = (1797, 8, 8)

    About inspect stack
    ----------------------
    Each record is a tuple of six items:

    0. the frame object,
    1. the filename,
    2. the line number of the current line,
    3. the function name,
    4. a list of lines of context from the source code, and
    5. the index of the current line within that list.

    Example output
    ------------------
    >>> st = inspect.stack()[1]
    >>> for i,j in enumerate(st):
    >>>     print i,j
    0 <frame object at 0x400f2d0>
    1 <ipython-input-31-cfc6434307aa>
    2 27
    3 <module>
    4 [u'myprint(digits.images.shape']
    5 0
    """
# #     st = inspect.stack()
    st = inspect.stack()[1]
# #     print len(st)
#     for i,j in enumerate(st):
#         print i,j
# #     print(dir(stuff))
# #     print stuff.__getattribute__
    print "{} = {}".format(st[4][0][8:-2], stuff)
#    pprint("{} = {}".format(st[4][0][8:-2], stuff))


#%% -- from wes mckiness book --
def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)

#%% === Probability and random process ===
#%% random number
def rand(range=(0,1),*args):
    """Wrapper to np.random.rand, but allows me to specify location parameter

    https://docs.python.org/2/faq/programming.html#how-can-i-pass-optional-or-keyword-parameters-from-one-function-to-another
    http://stackoverflow.com/questions/3394835/args-and-kwargs
    """
    return np.random.rand(*args) * (range[1]-range[0]) + range[0]

#%% === PANDAS ===
def pd_check_pred(ytrue,ypred, err_type='clf'):
    """ Helper to check prediction error

    Created 04/10/2016

    Functionality self-explanatory from code
    """
    if err_type == 'clf':
        # classification
        err = (ytrue == ypred).astype(int)
        str_ = 'equal'
    else:
        # absolute error
        err = np.abs(ytrue-ypred)
        str_ = 'abs_err'
    return pd.DataFrame( [ytrue, ypred, err],
                          index=['ytrue','ypred',str_]).T

def pd_dir(obj, start_str='__'):
    """ Updated 04/02/2016

    Converted everything into a string!
    This way I can view everything in the variable explorer.

    Show 2column table of dir(obj)

    Input
    -----
    obj : object
        anything you can do dir(obj) on
    start_str : String (default: '_')
        filter option (useful to filter underscore ones)

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_dir(clf)
    """
    if start_str is not None:
        mylist = [x for x in dir(obj) if not x.startswith(start_str)]
    else:
        mylist = [x for x in dir(obj)]

    attrlist = []
    for x in mylist:
        try:
            attrlist.append(str(getattr(obj, x)))
        except Exception as err:
            attrlist.append('err: ' + str(err))

    typelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            typelist.append(str(type(attr)))
        except Exception as err:
            typelist.append('err: ' + str(err))

#    return pd.DataFrame([mylist, attrlist, typelist],
#                        index=['attr-name', 'attr-value', 'type']).T
    sizelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            if isinstance(attr, np.ndarray):
                sizelist.append(str(attr.shape))
            else:
                sizelist.append(str(len(attr)))
        except Exception as err:
            sizelist.append('err: ' + str(err))

    return pd.DataFrame([mylist, attrlist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T


def pd_dir_old(obj, start_str='__'):
    """ Created 10/15/2015

    Show 2column table of dir(obj)

    Input
    -----
    obj : object
        anything you can do dir(obj) on
    start_str : String (default: '_')
        filter option (useful to filter underscore ones)

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_dir(clf)
    """
    if start_str is not None:
        mylist = [x for x in dir(obj) if not x.startswith(start_str)]
    else:
        mylist = [x for x in dir(obj)]

    attrlist = []
    for x in mylist:
        try:
            attrlist.append(getattr(obj, x))
        except Exception as err:
            attrlist.append('err: ' + str(err))

    typelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            typelist.append(type(attr))
        except Exception as err:
            typelist.append('err: ' + str(err))

    sizelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            if isinstance(attr, np.ndarray):
                sizelist.append(attr.shape)
            else:
                sizelist.append(len(attr))
        except Exception as err:
            sizelist.append('err: ' + str(err))

    return pd.DataFrame([mylist, attrlist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T


def pd_fillna_taka(df):
    """Fill missing float values with mean, and "categorical" with mode

    Input
    ------
    df :
        data frame

    Output
    ------
    df :
        data frame

    fill_values :
        values filled in the nans

    """
    """Fill with mode value for dtype = "category" or "object" """
    df_out = df.copy()

    # loop through each column (may be better way, but will do)
    for colname in df_out.columns:
        cond1 = str(df_out[colname].dtype) in ['category', 'object']
        cond2 = df_out[colname].isnull().sum() != 0
        if cond1 and cond2:
            # set_trace()
            # mode()[0] since mode returns tuple
            df_out[colname].fillna(df_out[colname].mode()[0],inplace=True)
        elif cond2:
            df_out[colname].fillna(df_out[colname].mean(), inplace=True)
    return df_out

def pd_fillnan(df, p = 0.3):
    """ Fill in nan-values at random place with probability p

    Input
    -------
    df : data frame
        data frame object

    p : [0,1]
        probability of nan elements

    Output
    ------
    df: dataframe with nans

    Example
    ----------
    >>> df = pd.DataFrame(data = np.random.randn(10,6),columns=list('ABCDEF'))
    >>>
    >>> df['gender'] = pd.Series(['male']*7 + ['female']*3, dtype='category')
    >>> df['growth'] = pd.Series(['fast']*6 + ['slow']*2 + ['medium']*2)
    >>>
    >>> df = pd_fillnan(df,0.3)
    """
    df_out = df.copy() # <- create new object


    # for now, just usea loop
    for icol in xrange(df_out.shape[1]):
        #| print "({},{})".format(df.columns[icol], df.dtypes[icol])
        # insert nans at random row-indices
        mask = np.random.rand(df_out.shape[0])< p
        df_out.ix[mask, icol] = np.nan
    return df_out

def pd_fillna_mode(df):
    """Fill with mode value for dtype = "category" or "object" """
    df_out = df.copy()

    # loop through each column (may be better way, but will do)
    for colname in df_out.columns:
        cond1 = str(df_out[colname].dtype) in ['category', 'object']
        cond2 = df_out[colname].isnull().sum() != 0
        if cond1 and cond2:
            # set_trace()
            # mode()[0] since mode returns tuple
            df_out[colname].fillna(df_out[colname].mode()[0],inplace=True)
    return df_out


#%% -- stuffs i don't use too often anymore (4/12/2016) --
def pd_setdiff(df1,df2):
    """ Created 10/13/2015

    I like to use this to see what attributes are added when using the "fit"
    method in scikit.

    Example
    --------
    >>> # items prior to fitting
    >>> df_prefit = pd.DataFrame(dir(clf))
    >>>
    >>> # fit
    >>> clf.fit(Xtr, ytr)
    >>>
    >>> df_postfit = pd.DataFrame(dir(clf))
    >>>
    >>> pd_setdiff(df_postfit, df_prefit)
    """
    return pd.DataFrame(list(set(df1[0]) - set(df2[0])))


def pd_dict_to_DF(dict_var):
    """ Create DF of dict object with summary info.

    Handy in ipython notebook

    Created 10/17/2015

    """
    df = pd.DataFrame(pd.Series(dict_var)) # somehow need to make Series first (DF won't accept dict directly)
    df.columns = ['value'] # reassign column name

    # create column of "type" and "shape"
    type_list = []
    shape_list = []
    for i in xrange(df.shape[0]):
        val = df.ix[i,0]
        type_list.append(type(val))
        try:
            if isinstance(val, np.ndarray):
                shape_list.append(val.shape)
            else:
                shape_list.append(len(val))
        except Exception:
            shape_list.append('None')
    df['shape'] = shape_list
    df['type'] = type_list

    return df

def pd_attr(obj, attrlist):
    """ Created 10/16/2015

    Add doc later.
    Use case in ^1016-try-glmnet-stability-selection.ipynb

    Example
    -------
    >>> lognet = LogisticNet(alpha=1)
    >>> lognet_dir_prefit = dir(lognet)
    >>> lognet.fit(Xz, y)
    >>> attrlist = list(set(dir(lognet)) - set(lognet_dir_prefit))
    >>> tw.pd_attr(lognet, attrlist)
    """
    # ensure list of strings are sorted
    attrlist.sort()

    #
    valuelist = []
    for x in attrlist:
        try:
            valuelist.append(getattr(obj, x))
        except Exception as err:
            valuelist.append('err: ' + str(err))

    typelist = []
    for x in attrlist:
        try:
            attr = getattr(obj, x)
            typelist.append(type(attr))
        except Exception as err:
            typelist.append('err: ' + str(err))

    sizelist = []
    for x in attrlist:
        attr = getattr(obj, x)
        try:
            if isinstance(attr, np.ndarray):
                sizelist.append(attr.shape)
            else:
                sizelist.append(len(attr))
        except Exception as err:
            sizelist.append('None')
            # sizelist.append('err: ' + str(err))

    return pd.DataFrame([attrlist, valuelist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T

@deprecated("Function 'pd_underscore' to be replaced by 'pd_fit_attr'")
def pd_underscore(str_list):
    """ Created 10/13/2015

    Prints attributes ending with underscore, but not beginning with underscore

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_underscore(clf)
    """
    #
    mylist = [
        x for x in dir(str_list) if not x.startswith('_') and x.endswith('_')]
    return pd.DataFrame(mylist)


def pd_fit_attr(str_list):
    """ Created 10/14/2015

    Prints attributes ending with underscore, but not beginning with underscore

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_fit_attr(clf)
    """
    #
    mylist = [x for x in dir(str_list) if not x.startswith('_') and x.endswith('_')]
    return pd.DataFrame(mylist)


def pd_fit_attr2(obj):
    """ Created 10/15/2015

    Extension of pd_fit_attr

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_fit_attr(clf)
    """
    #
    mylist = [x for x in dir(obj) if not x.startswith('_') and x.endswith('_')]
    attrlist = []
    for x in mylist:
        try:
            attrlist.append(getattr(obj, x))
        except Exception as err:
            attrlist.append('err: ' + str(err))

    typelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            typelist.append(type(attr))
        except Exception as err:
            typelist.append('err: ' + str(err))

    sizelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            if isinstance(attr, np.ndarray):
                sizelist.append(attr.shape)
            else:
                sizelist.append(len(attr))
        except Exception as err:
            sizelist.append('err: ' + str(err))

    return pd.DataFrame([mylist, attrlist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T

def pd_class_signature(ClassObject, index_name=None):
    """ Create DataFrame of class siganture and defaults.

    Created 10/16/2015
    Can be handy for ipython notebook.

    Example
    --------
    >>> from sklearn.linear_model import RandomizedLasso
    >>> pd_class_signature(RandomizedLasso)

    """
    argspec = inspect.getargspec(ClassObject.__init__)

    # print inspect.getargspec(pd_class_signature)
    # print inspect.getargvalues(inspect.currentframe())
    # args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    # print args

    #argname = inspect.stack()[1][-2][0]
    """
    here argname will look something like this:
    >>> df1 = tw.pd_class_signature(RandomizedLasso)
        argname = 'df1 = tw.pd_class_signature(RandomizedLasso)'

    I want the shit inside the round bracket...use regex to extract that
    """
    # "group" to return the string matched by the RE
    #argname = re.search(r'\(\w+\)$', argname).group()
    # remove round bracket at beginning and end
    #argname = argname[1:-1]

    # print inspect.trace()

    # .defaults are tuples, so convert to list
    # (note: args[1:] to ignore "self" argument)
    # df = pd.DataFrame([argspec.args[1:], list(argspec.defaults)],
    #               index=['args', 'default'])
    if index_name is None:
        index_name = 'default'
    else:
        index_name = 'default ({})'.format(index_name)
    df = pd.DataFrame([list(argspec.defaults)], columns = argspec.args[1:],
                  index=[index_name])
    return df

def pd_prepend_index_column(df, colname='i', col_at_end = False):
    """ Prepend a column with entries 0, .., df.shape[0]-1

    Handy for visualizing dataFrames as tables, and keeping track of which row you're on

    Usage
    ------
    >>> df = tw.pd_prepend_index_column(df, 'index_original')

    Parameters
    -----------
    df : DataFrame
        DataFrame object to prepend on
    colname : string (default = 'index_original')
        Column name to assign on the prepended column
    col_at_end : bool (default = False)
        Append column at the end (although I don't usually use this, since
        it's easier to do "sanity-checks" by having the index-column next to the
        DataFrame's Index object)
    """
    df[colname] = range(df.shape[0])

    if col_at_end:
        return df

    # bring this column up front
    # http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    return df[cols]


# def pd_method_signature(some_method):
#     """ Create DataFrame of method/function siganture and defaults.

#     Created 10/16/2015
#     Can be handy for ipython notebook.

#     Example
#     --------
#     >>> from sklearn.linear_model import RandomizedLasso
#     >>> pd_class_signature(RandomizedLasso)

#     """
#     argspec = inspect.getargspec(some_method)

#     # .defaults are tuples, so convert to list
#     return pd.DataFrame([argspec.args, list(argspec.defaults)],
#                   index=['args', 'default'])

def get_fit_attr(str_list):
    """ Created 10/14/2015

    Prints attributes ending with underscore, but not beginning with underscore.
    These are generally stuffs appended after .fit() method has taken place.

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> get_fit_attr(clf)

    See Also
    ---------
    :func:`pd_underscore`:
    """
    #
    mylist = [x for x in dir(str_list) if not x.startswith('_') and x.endswith('_')]
    return mylist

def pd_get_column_info(df):
    """ Get column info of a DataFrame...as a DataFrame!

    Migrated from ``data_io.get_df_column_info`` on 11/05/2015

    I may pad on more columns to the output DataFrame in the future
    """
    fields = pd.DataFrame(
                 {'columns':df.columns,
                  'dtypes':df.dtypes,
                  'nan_counts':df.isnull().sum(),
                  })
    fields = fields.reset_index(drop=True)
    fields['nan_rate'] = 1.*fields['nan_counts']/df.shape[0]

    return fields

@deprecated('Use function plt_fix_xticklabels_rot in plot.py')
def pd_plot_fix_xtick_labels_rot():
    """ Fix the odd looking xtick when ``rot`` is used in pandas plot

    **Created 11/18/2015**
    (see ``pnc_analyze_clf_summary_1118.py`` for usage)

    Example
    -------
    corrmat.mean().plot(kind='bar', fontsize=14, rot=30)
    pd_plot_fix_xtick_labels_rot()
    """
    # list of text objects
    xtick_list = plt.gca().get_xticklabels()

    #text_list = [xtick.get_text() for xtick in xtick_list]
    #return text_list

    for xtick in xtick_list:
        xtick.set_ha('right')
    plt.draw()


def pd_plot_move_legend(loc=(1.25,0.7),fontsize=12):
    plt.gca().legend(bbox_to_anchor=loc, fontsize=fontsize)
#%%***************************************************************************#
#%% *              Above is made clean.....cleanup below later                #
#%%***************************************************************************#
#%% ========= Volume slicer ===============#
#"""
#Example of an elaborate dialog showing a multiple views on the same data, with
#3 cuts synchronized.
#
#This example shows how to have multiple views on the same data, how to
#embedded multiple scenes in a dialog, and the caveat in populating them
#with data, as well as how to add some interaction logic on an
#ImagePlaneWidget.
#
#The order in which things happen in this example is important, and it is
#easy to get it wrong. First of all, many properties of the visualization
#objects cannot be changed if there is not a scene created to view them.
#This is why we put a lot of the visualization logic in the callback of
#scene.activated, which is called after creation of the scene.
#Second, default values created via the '_xxx_default' callback are created
#lazyly, that is, when the attributes are accessed. As the establishement
#of the VTK pipeline can depend on the order in which it is built, we
#trigger these access by explicitely calling the attributes.
#In particular, properties like scene background color, or interaction
#properties cannot be set before the scene is activated.
#
#The same data is exposed in the different scenes by sharing the VTK
#dataset between different Mayavi data sources. See
#the :ref:`sharing_data_between_scenes` tip for more details.
#
#In this example, the interaction with the scene and the various elements
#on it is strongly simplified by turning off interaction, and choosing
#specific scene interactor styles. Indeed, non-technical users can be
#confused with too rich interaction.
#"""
## Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
## Copyright (c) 2009, Enthought, Inc.
## License: BSD Style.
#
##import numpy as np
#
#from traits.api import HasTraits, Instance, Array, \
#    on_trait_change
#from traitsui.api import View, Item, HGroup, Group
#
#from tvtk.api import tvtk
#from tvtk.pyface.scene import Scene
#
#from mayavi import mlab
#from mayavi.core.api import PipelineBase, Source
#from mayavi.core.ui.api import SceneEditor, MayaviScene, \
#                                MlabSceneModel
#
#################################################################################
## Create some data
##x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]
##data = np.sin(3*x)/x + 0.05*z**2 + np.cos(3*y)
#
#################################################################################
## The object implementing the dialog
#class VolumeSlicer(HasTraits):
#    # The data to plot
#    data = Array()
#
#    # The 4 views displayed
#    scene3d = Instance(MlabSceneModel, ())
#    scene_x = Instance(MlabSceneModel, ())
#    scene_y = Instance(MlabSceneModel, ())
#    scene_z = Instance(MlabSceneModel, ())
#
#    # The data source
#    data_src3d = Instance(Source)
#
#    # The image plane widgets of the 3D scene
#    ipw_3d_x = Instance(PipelineBase)
#    ipw_3d_y = Instance(PipelineBase)
#    ipw_3d_z = Instance(PipelineBase)
#
#    _axis_names = dict(x=0, y=1, z=2)
#
#
#    #---------------------------------------------------------------------------
#    def __init__(self, **traits):
#        super(VolumeSlicer, self).__init__(**traits)
#        # Force the creation of the image_plane_widgets:
#        self.ipw_3d_x
#        self.ipw_3d_y
#        self.ipw_3d_z
#
#
#    #---------------------------------------------------------------------------
#    # Default values
#    #---------------------------------------------------------------------------
#    def _data_src3d_default(self):
#        return mlab.pipeline.scalar_field(self.data,
#                            figure=self.scene3d.mayavi_scene)
#
#    def make_ipw_3d(self, axis_name):
#        ipw = mlab.pipeline.image_plane_widget(self.data_src3d,
#                        figure=self.scene3d.mayavi_scene,
#                        colormap='gray',
#                        plane_orientation='%s_axes' % axis_name)
#        return ipw
#
#    def _ipw_3d_x_default(self):
#        return self.make_ipw_3d('x')
#
#    def _ipw_3d_y_default(self):
#        return self.make_ipw_3d('y')
#
#    def _ipw_3d_z_default(self):
#        return self.make_ipw_3d('z')
#
#
#    #---------------------------------------------------------------------------
#    # Scene activation callbaks
#    #---------------------------------------------------------------------------
#    @on_trait_change('scene3d.activated')
#    def display_scene3d(self):
##        outline = mlab.pipeline.outline(self.data_src3d,
##                        figure=self.scene3d.mayavi_scene,
##                        )
#        self.scene3d.mlab.view(40, 50)
#        # Interaction properties can only be changed after the scene
#        # has been created, and thus the interactor exists
#        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
#            # Turn the interaction off
#            ipw.ipw.interaction = 0
#        self.scene3d.scene.background = (0, 0, 0)
#        # Keep the view always pointing up
#        self.scene3d.scene.interactor.interactor_style = \
#                                 tvtk.InteractorStyleTerrain()
#
#
#    def make_side_view(self, axis_name):
#        scene = getattr(self, 'scene_%s' % axis_name)
#
#        # To avoid copying the data, we take a reference to the
#        # raw VTK dataset, and pass it on to mlab. Mlab will create
#        # a Mayavi source from the VTK without copying it.
#        # We have to specify the figure so that the data gets
#        # added on the figure we are interested in.
#        outline = mlab.pipeline.outline(
#                            self.data_src3d.mlab_source.dataset,
#                            figure=scene.mayavi_scene,
#                            )
#        ipw = mlab.pipeline.image_plane_widget(
#                            outline,
#                            colormap='gray',
#                            plane_orientation='%s_axes' % axis_name)
#        setattr(self, 'ipw_%s' % axis_name, ipw)
#
#        # Synchronize positions between the corresponding image plane
#        # widgets on different views.
#        ipw.ipw.sync_trait('slice_position',
#                            getattr(self, 'ipw_3d_%s'% axis_name).ipw)
#
#        # Make left-clicking create a crosshair
#        ipw.ipw.left_button_action = 0
#        # Add a callback on the image plane widget interaction to
#        # move the others
#        def move_view(obj, evt):
#            position = obj.GetCurrentCursorPosition()
#            for other_axis, axis_number in self._axis_names.iteritems():
#                if other_axis == axis_name:
#                    continue
#                ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
#                ipw3d.ipw.slice_position = position[axis_number]
#
#        ipw.ipw.add_observer('InteractionEvent', move_view)
#        ipw.ipw.add_observer('StartInteractionEvent', move_view)
#
#        # Center the image plane widget
#        ipw.ipw.slice_position = 0.5*self.data.shape[
#                    self._axis_names[axis_name]]
#
#        # Position the view for the scene
#        views = dict(x=( 0, 90),
#                     y=(90, 90),
#                     z=( 0,  0),
#                     )
#        scene.mlab.view(*views[axis_name])
#        # 2D interaction: only pan and zoom
#        scene.scene.interactor.interactor_style = \
#                                 tvtk.InteractorStyleImage()
#        scene.scene.background = (0, 0, 0)
#
#
#    @on_trait_change('scene_x.activated')
#    def display_scene_x(self):
#        return self.make_side_view('x')
#
#    @on_trait_change('scene_y.activated')
#    def display_scene_y(self):
#        return self.make_side_view('y')
#
#    @on_trait_change('scene_z.activated')
#    def display_scene_z(self):
#        return self.make_side_view('z')
#
#
#    #---------------------------------------------------------------------------
#    # The layout of the dialog created
#    #---------------------------------------------------------------------------
#    view = View(HGroup(
#                  Group(
#                       Item('scene_y',
#                            editor=SceneEditor(scene_class=Scene),
#                            height=250, width=300),
#                       Item('scene_z',
#                            editor=SceneEditor(scene_class=Scene),
#                            height=250, width=300),
#                       show_labels=False,
#                  ),
#                  Group(
#                       Item('scene_x',
#                            editor=SceneEditor(scene_class=Scene),
#                            height=250, width=300),
#                       Item('scene3d',
#                            editor=SceneEditor(scene_class=MayaviScene),
#                            height=250, width=300),
#                       show_labels=False,
#                  ),
#                ),
#                resizable=True,
#                title='Volume Slicer',
#                )
#
#def slicer(data):
#    m = VolumeSlicer(data=data)
#    m.configure_traits()
#    return m
#%%____ OLD SHIT THAT IS POORLY DOCUMENTED AND CODED________
def dashed_message(message):
    dash = "#" + '=' *78 + '#'
    print '\n\n'+dash +'\n'+ message + '\n' + dash


def mycursor():
    from matplotlib.widgets import Cursor
    Cursor(plt.gca(),useblit=True, color='red', linewidth= 1 )

def imtak2(im, multicursor=False):
    from matplotlib.widgets import Cursor
    from xy_python_utils.matplotlib_utils import impixelinfo
    import mpldatacursor

    img=plt.imshow(im,interpolation='none')
    plt.colorbar()
    impixelinfo()

    if multicursor:
        display="multiple"
    else:
        display="single"

    dc=mpldatacursor.datacursor(formatter='(i, j) = ({i}, {j})\nz = {z:.2f}'.format,draggable=True,
               axes=plt.gca(),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
               edgecolor='magenta'), display=display)
#    datacursor(formatter='i:{i:d}\nj:{j:d}\nz:{z:.2f}'.format,
#               draggable=True,arrowprops=
#               dict(arrowstyle='->', connectionstyle='arc3,rad=0',
#                    edgecolor='magenta'))
#    Cursor(plt.gca(),useblit=True, color='red', linewidth= 1 )
    # return img,dc
    return img
#    plt.show()


def imtak_old(image):
     '''references
     http://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
     http://stackoverflow.com/questions/27704490/interactive-pixel-information-of-an-image-in-python
     '''
     np.set_printoptions(threshold='nan')
     fig, ax = plt.subplots()
     im = ax.imshow(image, interpolation='none')
     ax.format_coord = Formatter(im)
     mngr = plt.get_current_fig_manager()
     # to put it into the upper left corner for example:
 #    geom = mngr.window.geometry()
 #    x,y,dx,dy = geom.getRect()
     fig.colorbar(im)
     plt.show()
     mngr.window.setGeometry(-800,100,640, 545)
     return fig,im

if __name__ == "__main__":
    pass
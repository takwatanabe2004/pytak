"""
===============================================================================
This will contain my "final" version utility function.
===============================================================================
"""
#%% ===== global module load =====
import numpy as np
import scipy as sp
import pandas as pd
#import re
#import matplotlib.pyplot as plt
#import os
import sys
import time
#from pprint import pprint
#import warnings
import sklearn
#from sklearn.utils import deprecated as _deprecated

from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

#%% === cross-validation related stuffs ===
def cv_score_classifier(clf,X,y,cv=5,rng=None,silent=False):
    """ My cv_score with binary classifiers (created 1/21/2016)

    More freedom than scikit's cross_val_score?

    **Update 03/11/2016**
    ----------------------
    - return ytrue,ypred,score only

    **Update 01/24/2016**
    ----------------------
    - added option ``return_score`` (not a clf-methods have **score** associated with it)
    - return ACC, TPR, TNR as first 3 outputs

    Parameters (some doc from sklearn's cross_val_score)
    ----------
    clf : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning (+1 or -1....binary classification)

    cv : cross-validation generator or int, optional, default: 5
        A cross-validation generator to use. If int, determines
        the number of folds in StratifiedKFold.
    """
    from sklearn.cross_validation import StratifiedKFold
    #from tak import clf_summary_short
    #from tak import clf_results
    from tak import clf_results_extended
    if isinstance(cv, int):
        if rng is None:
            cv = StratifiedKFold(y,n_folds=cv,shuffle=True)
        else:
            cv = StratifiedKFold(y,n_folds=cv,shuffle=True,random_state=rng)
    ypred=[]
    ytrue=[]
    score=[]
    for itr,its in cv:
        Xtr,Xts = X[itr],X[its]
        ytr,yts = y[itr],y[its]

        clf.fit(Xtr,ytr)

        ytrue.append(yts)
        ypred.append(clf.predict(Xts))
        try:
            score.append(clf.decision_function(Xts))
        except:
            # if classifier doesn't have method "decision_function", return ypred
            score.append(clf.predict(Xts))

    ytrue = np.concatenate(ytrue)
    ypred = np.concatenate(ypred)
    score = np.concatenate(score)

    if not silent:
        #print clf_summary_short(ytrue,ypred)
        #print clf_results(ytrue,ypred)
        print clf_results_extended(ytrue,score)

    return ytrue,ypred,score

#%% === classification related stuffs ===
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
    

def clf_roc(ytrue, score, return_thresholds=False):
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
#%% === imported from tak.core.py (06/07/2016) ===
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
    
    
#%% === feature selection routines ===
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

def ttest_for_fs(X,y):
    """ Two-sample ttest for feature selection in scikit (created 10/31/2015)
    
    Unlike ``ttest_feature_sel``, this function is intended to be used with
    scikit-learn's ``Pipeline`` functionality (see example usage below)

    Note that the ``score`` that scikit uses for feature selection should be
    non-negative, so here the absolu value of the t-statistic is returned.

    See http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

    For **Pipelining** feature selection with classifiers, see http://scikit-learn.org/stable/modules/pipeline.html

    Parameters
    -------------
    X : array-like, shape = [n_samples, n_features]
        The training input samples.
    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).

    Returns
    -------
    score : float or array
        The calculated t-statistic in **absolute value** (to be used as ``score`` in Scikit's feature selection)
    pvalue : float or array
        The two-tailed p-value.

    Usage
    ------
    Select ``k=100`` features of top tstats significance using
    `SelectKBest <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>`_

    >>> from sklearn.feature_selection import SelectKBest
    >>> fs = SelectKBest(score_func=ttest_for_fs, k=100)
    >>> fs.fit(Xtrz,ytr)

    Can also threshold by pvalue ``alpha=0.01`` using
    `SelectFpr <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html>`_

    >>> from sklearn.feature_selection import SelectFpr
    >>> SelectFpr(score_func=ttest_for_fs, alpha=0.01)
    >>> fs.fit(Xtrz,ytr)
    >>> supp = fs.get_support()
    >>> pval_sorted = np.sort(fs.pvalues_[supp])
    >>> Xtrz_fs1 = fs.transform(Xtrz)
    >>> Xtsz_fs1 = fs.transform(Xtsz)

    Usage with scikit's pipeline
    -----------------------------
    For more info, see http://scikit-learn.org/stable/modules/pipeline.html

    Use ttest together linear svm

    >>> from sklearn.feature_selection import SelectKBest
    >>> ttest_fs = SelectKBest(score_func=ttest_for_fs, k=100)
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.pipeline import Pipeline
    >>> clf = Pipeline([
    ...   ('myttest', ttest_fs),
    ...   ('clfname', LinearSVC(loss='hinge'))
    ... ])

    Setting parameters in pipline has the syntax ``<estimator>__<parameter>``

    >>> params = dict(fsname__k=30,clfname__C=10)
    >>> clf.set_params(**params)
    >>> clf.fit(Xtrz,ytr)
    >>> ypr = clf.predict(Xtsz)

    The individual **estimators** of a pipeline are stored as a **dict** attribute ``named_steps``

    >>> ttest_fitted = clf.named_steps['fsname'] # same as above
    >>> print ttest_fitted.pvalues_
    >>> w = ttest_fitted.inverse_transform(ttest_fitted.transform(Xtrz[0,:])).ravel()
    >>> tw.imconnmat(w)

    Get the indices and the pvalues of the selected features
    (note: scikit returns a length-p bool vector, so extract the index location
    of the nonzeroes below)

    >>> idx_ = clf.named_steps['myttest'].get_support().nonzero()[0]
    >>> pval_ = clf.named_steps['myttest'].pvalues_[idx_]

    By default, above ``(idx_,pval_)`` is sorted by ``idx_``....you can sort by pvalue as follows:

    >>> idx_,pval_ = idx_[pval.argsort()], pval_[pval.argsort()]
    """
    tstats, pvalue = sp.stats.ttest_ind( X[y==+1,:], X[y==-1,:] )
    score = np.abs(tstats)
    #score = tstats
    return score,pvalue
    
    
    
def inv_fs_transform(x_fs, idx_fs, n_feat):
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
    
    History
    -------
    - 06/07/2016 - changed from ``inverse_fs_transform`` in ``core.py`` file
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
#%%=== my implementations ====
def fista_logistic_graphnet(X, y, w, lam, rho, C):
    """ GraphNet  **Created 11/24/2015**"""
    from nilearn.decoding.fista import mfista
    obj_func = lambda w: twd.logistic_loss(X,y,w) + gam * norm(C.dot(w),2)/2 + lam*norm(w,1)
    f1_grad = lambda w: twd.logistic_loss_grad(X,y,w) + gam * L.dot(w)

#%%=== my stuffs ====
def rfe_rbf(X,y,C,gamma,n_features_to_select=None,step=0.2,verbose=0):
    """Recursive feature elimination (RFE) when using RBF kernel

    - To see how RFE works with kernelized approach, see pg415 Sec 6.3 in
      2002 S. Guyon - Gene Selection for Cancer Classification using SVM
    - For speed, I didn't recreate the gram matrix Ki for every feature removed
      during the RFE algorithm....the code doesn't exactly reflect the equations
      in the paper.
    - otherwise, code tries to follow notations from Guyon's SVM-RFE algorithm
      on pg 396 of above paper (Sec 3.2)

    Parameters
    -----------
    X : ndarray, shape = [n_samples, n_features]
        The training input samples.
    y : ndarray, shape = [n_samples]
        The target values.
    C : float
        SVM regularizer (lower value gives more penalty)
    gamma : float
        Kernel coefficient for 'rbf' kernel.
    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.
    step : int or float, optional (default=0.2)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
    verbose : int, default=0
        Controls verbosity of output.

    Returns
    --------
    s : ndarray of ints, shape = [n_features_to_select]
        vector indicating subset of features selected by RFE
    r : wtf was this?
        feature ranked list
    s_list : [#iter x 1] list
        list that keeps track of the features selected during each step of the
        RFE algorithm

    Development
    -----------
    See ``~/tak-ace-ibis/python/pnc/protocodes/1025_proto_rfe_rbf.py``
    """
    #=========================================================================#
    # load relevant modules, and parse inputs
    #=========================================================================#
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVC
    n_features = X.shape[1]

    if n_features_to_select is None:
        n_features_to_select = n_features / 2

    if 0.0 < step < 1.0:
        step = int(max(1, step * n_features))
    else:
        step = int(step)
    if step <= 0:
        raise ValueError("Step must be >0")

    svm = SVC(kernel='precomputed', C=C, gamma=gamma)

    #=========================================================================#
    # setup for rfe algorithm
    #=========================================================================#
    s = np.arange(n_features,dtype=int) # initial feature list (subset of surviving features)
    r = np.array([]) # feature ranked list

    # to keep track of ranking in the original index space {0,...,p-1}, i keep
    # track of the feature ranking from the previous iteration
    prev_ranking = s

    #=========================================================================#
    # begin rfe algorithm
    #=========================================================================#
    # (#iter x 1) cell array that keeps track of the features selected
    #          during each iteration of the RFE algorithm
    s_list = []
    #%%
    while True:
        #%%
        n_features = s.shape[0] # number of features in the current iteration

        # train SVM on current feature set
        X_rfe = X[:,s]
        K = rbf_kernel(X_rfe, gamma=gamma)
        svm.fit(K, y)
        #%%
        #=====================================================================#
        # for efficiency, recreate kernel matrix using only the data points
        # that are support vectors
        #=====================================================================#
        idx_sv = svm.support_ # indices of support vectors, shape = [n_SV]

        # feature matrix corresponding to the SV data points
        X_rfe_sv = X_rfe[idx_sv,:]

        # kernel matrix consisting of SV-data points
        K_sv = K[np.ix_(idx_sv,idx_sv)]
        #K_sv2 = rbf_kernel(X_rfe_sv,X_rfe_sv,gamma=gamma)

        #=====================================================================#
        # libsvm flips the sign of alpha by the label, so fix this
        # (dual formulation of alpha enforces non-negativitiy...I like to have
        #  the variable names agree with conventional svm notations)
        #=====================================================================#
        alpha =  np.abs(svm.dual_coef_.ravel())

        #=====================================================================#
        # compute the first term in DJ(i) in pg.415 (ignore scaling 1/2)
        # $\alpha^T H \alpha$, where $H_ij = yi * yj * k(xi,xj)$
        #=====================================================================#
        yalpha = (alpha*y[idx_sv])[:,np.newaxis]; # shape to [n_sv,1]
        DJ1 = yalpha.T.dot(K_sv.dot(yalpha))
        # print DJ1
        #%%
        #==========================================================================%
        # loop over each features in s to compute the 2nd term in DJ
        # - here we have to recreate the kernel matrix with the j-th feature removed
        # - i don't actually recreate a new kernel matrix every iterations, but
        #   instread employ a more efficient implementation
        #   (see proto version of the code for details - tak_rfe_libsvm_proto2.m)
        #==========================================================================%
        DJ2 = np.zeros((n_features))
        A = np.sum(X_rfe_sv**2, axis=1)[:, np.newaxis]
        B = 2*(X_rfe_sv.dot(X_rfe_sv.T)) #% <- precomputing the *2 product speeds things up a bit

        #from sklearn.metrics import pairwise_distances
        #from scipy.linalg import norm
        #EDM1 = pairwise_distances(X_rfe_sv)**2
        #EDM2 = A+A.T - B
        #%%
        #*** in matlab ***#
        #A = sum(X_rfe_sv.^2, 2);
        #B = 2*(X_rfe_sv*X_rfe_sv');

        start_time=time.time()
        # rank features (90 seconds on p=3655...argh...)
        for j in range(n_features):
            if j%500==0:
                print("    (j = {:5} out of {:5})".format(j+1,n_features)),
                print_time(start_time)
            #======================================================================%
            # more efficient way to compute distance matrix with j-th feature removed
            #======================================================================%
            xj = X_rfe_sv[:,j] # the feature to remove
            Aj = A - xj**2

            xj2 = np.sqrt(2)*xj[:,np.newaxis]; # <- sqrt(2) to speed things up (deals with *2 scaling)
            Bj = B - (xj2.dot(xj2.T))
            eucdist = Aj + Aj.T - Bj

            # we want eucdist above to equal this
            # below gives: >>> 5.79879194039e-09, so good!
            #eucdist2 = pairwise_distances(np.delete(X_rfe_sv,j,axis=1))**2
            #print norm(eucdist-eucdist2)

            #======================================================================%
            # now we can create the rbf kernel matrix, and compute DJ
            #======================================================================%
            Ksv_j = np.exp(-gamma*eucdist)
            DJ2[j] = yalpha.T.dot(Ksv_j.dot(yalpha))

            # what we want - the 2nd term in DJ(i) (sec 6.3 Guyon)
            # K_minus_i =  gram matrix with jth feature removed
            #K_minus_i = rbf_kernel(np.delete(X_rfe_sv,j,axis=1),gamma=gamma)
            #DJ2_true = yalpha.T.dot(K_minus_i.dot(yalpha))
            #print DJ2[j]-DJ2_true # <- gives  1.77635684e-15, so good!
        #%% brute force approach (took like 280 seconds...)
        #| here explicitly create the kernel matrix with the j-th feature removed for each iteration
        #start_time = time.time()
        #DJ2b = np.zeros((n_features,1))
        #for j in range(n_features):
        #    if j%10==0:
        #        print("    (j = {:5} out of {:5})".format(j+1,n_features)),
        #        print_time(start_time)
        #    K_minus_i = rbf_kernel(np.delete(X_rfe_sv,j,axis=1),gamma=gamma)
        #    DJ2b[j] = yalpha.T.dot(K_minus_i.dot(yalpha))
        #%%
        # feature ranking according to DJ = DJ1 - DJ2
        idx_rank = np.argsort(np.abs(DJ1-DJ2)).ravel()[::-1] # last indexing to make it a descending sort
        #diff = np.abs(DJ1-DJ2).ravel()
        #plt.plot(diff[idx_rank])

        # ranking in original index space
        idx_best = prev_ranking[idx_rank]

        if n_features/2 > n_features_to_select:
            # take top-half of best features
            s = idx_best[:n_features/2]

            # throw the bottom-half away as "useless"
            # (note the order i append this)
            r = np.r_[idx_best[n_features/2:], r]
            s_list.append(s)
        else:
            # take the desired number of features and exit RFE loop
            s = idx_best[:n_features_to_select]
            r = np.r_[idx_best[n_features_to_select:],r]
            s_list.append(s)
            break

        prev_ranking = idx_best
    return s,r,s_list


#%% === WRAPPERS TO OTHER PACKAGES ===
from sklearn.svm.base import BaseSVC as _BaseSVC
from sklearn.metrics.pairwise import rbf_kernel as _rbf_kernel
from sklearn.feature_selection import RFE as _RFE

def stab_sel_glmnet_logreg(X,y, alpha=1.0, n_lambdas=100,sample_rate=0.5,
                    threshold=0.5,n_rep=100,verbose=0,random_state=None):
    """ Stability selection via glmnet logistic regression

    Glmnet equation
    ----------------
    .. math::

        L(\\mathbf{w}) =
            \\text{Loss}(\\mathbf{X,y,w}) +
            \\lambda \\Big( (1 - \\alpha) \\frac{1}{2} \\| \\mathbf{w} \\|^2_2 +
            \\alpha  \\| \\mathbf{w} \\|_1 \\Big)

    Stability selection equation (eqn 7 in Meinshausen paper)
    -----------------------------
    .. math::

        \\hat{S}^\\text{stable} = \\Big\\{k:
        \\max_{\\lambda\\in\\Lambda}\\hat{\\Pi}^\\lambda_k
        \\ge \\pi_\\text{thr} \\Big\\}

    Parameters
    -----------
    alpha : float (default 1.0)
        Glmnet parameter - value between 0 and 1 (alpha = 1 gives Lasso)
    n_lambdas : int (default: 100)
        Glmenet parameter - number of lambda values to insert
    sample_rate : float (default 0.5)
        Stabsel parameter - The ratio of data to subsample for each stability selection iteration
    n_rep : int (default 100)
        Stabsel parameter - Number of sampling with replacement of data
    threshold : float or ndarray of float (default 0.5)
        Stabsel parameter - selection threshold of stability selection
        (:math:`\\pi_\\text{thr}` iin eqn 7 in Meinshausen paper)
    verbose : int, default=0
        Controls verbosity of output
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    Returns
    --------
    support : (p x len(threshold)) ndarray :math:`\\hat{S}^\\text{stable}`
        Support set returned from stability selection:

    sel_probmax : (p,) ndarray
        Max selection probability (values between 0 and 1).
        Corresponds to the following expression in eqn 7 Meinshausen:

        .. math::

            \\max_{\\lambda\\in\\Lambda}\\hat{\\Pi}^\\lambda_k

    Example
    ---------
    >>> support,sel_probmax = stab_sel_glmnet(X,y,threshold=[0.5,1],n_rep=100)
    >>> support.sum(axis=0) # <- gives nnz for each threshold level

    References
    ----------
    Stability selection
    Nicolai Meinshausen, Peter Buhlmann
    Journal of the Royal Statistical Society: Series B
    Volume 72, Issue 4, pages 417-473, September 2010
    DOI: 10.1111/j.1467-9868.2010.00740.x
    """
    if verbose > 0:
        print("Stability selection with the following parameters:\n"+
              "       alpha = {:4.3f}\n".format(alpha)+
              "   n_lambdas = {:3}\n".format(n_lambdas)+
              " sample_rate = {:4.3f}\n".format(sample_rate)+
              "       n_rep = {:3}\n".format(n_rep)+
              "   threshold = {}\n".format(threshold))
    rng = check_random_state(random_state)

    n,p = X.shape

    glmnet = LogisticGlmNet(alpha=alpha, n_lambdas=n_lambdas)
    coef_counter = np.zeros( (p,n_lambdas), dtype=int)

    # number of data points to resample (without replacement)
    n_subsamp = np.ceil(sample_rate*n).astype(int)

    # run stability selection on the training data
    start_time = time.time()
    for irep in range(n_rep):
        #=====================================================================#
        # verbosity block
        #=====================================================================#
        if verbose >1:
            if verbose == 2:
                disp_freq = 25 # how often to display progress
            elif verbose == 3:
                disp_freq = 10
            elif verbose == 4:
                disp_freq = 5
            else:
                disp_freq = 1
            if (irep+1)%disp_freq == 0:
                print("    (Repetition = {:4} out of {:4})".format(irep+1,n_rep)),
                print_time(start_time)
        #---------------------------------------------------------------------#
        # random subsamples
        idx_sub = rng.permutation(n)[:n_subsamp]
        glmnet.fit( X[idx_sub], y[idx_sub])

        # sometimes glmnet has convergence issues...if so, resample data and try again
        while glmnet.out_n_lambdas_ != n_lambdas:
            #ipdb.set_trace()
            print("Something messed up internally in glmnet.  "+
                   "Resample and fit again.")
            idx_sub = rng.permutation(n)[:n_subsamp]
            glmnet.fit( X[idx_sub], y[idx_sub])

        coef_counter += (glmnet.coef_ != 0).astype(int)

    coef_rate = 1.*coef_counter/n_rep
    sel_probmax = coef_rate.max(axis=1)

    # (p x len_grid) binary mask of feature-selection matrix (array broadcasting)
    support = sel_probmax[:,np.newaxis] > threshold
    if np.isscalar(threshold):
        support = support.ravel()
    #idx_fs = np.zeros( (p, len_grid), dtype=bool )
    #for i,thresh in enumerate(grid_thresh):
    #    idx_fs[:,i] = coef_max > thresh
    #np.array_equal(idx_fs, coef_max[:,np.newaxis] > grid_thresh)

    return support, sel_probmax
    

class RFESVM(_RFE):
    """Simple wrapper class using linear-svm with hinge loss with RFE

    **Created 11/07/2015**

    I made this so I can access both the ``C`` parameter of SVM and ``n_features_to_select``
    from RFE via the ``select_params`` method (thus works with my 2d-grid-search script)

    Class inherits from ``RFE``, so for see docstring there for details.
    """
    def __init__(self, C=1, loss='hinge',n_features_to_select=1, step=1,verbose=0):
        from sklearn.svm import LinearSVC

        self.C = C
        self.loss = loss
        self.estimator = LinearSVC(loss=loss)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.estimator_params=None


class PrecomputedRBFSVM(_BaseSVC):
    """ My wrapper class for RBF-SVM with precomputed kernel (much faster to precompute kernel)

    The ``.fit()`` and ``.predict()`` methods will internally precompute the
    rbf kernel matrix, so that this Class can be used directly in my
    nested CV scripts.

    Important note
    -----------------
    - For prediction, we need the training data matrix ``Xtr``
    - However, I don't want to break the streamline of ``.fit(Xtr,ytr)`` and
      ``.predict(Xts)``...(ie, don't wanna have to do ``.predict(Xts,Xtr)``,
      which reads awkward and breaks code consistency)
    - ultra adhoc work-around I adopted is to create the attribute ``self.Xtr_``
      after ``.fit(Xtr,ytr)``, so the training data matrix is accessible during
      ``.predict(Xts)``
    - after prediction is complete, just do
    >>> del clf.Xtr_

    - usually, above approach (tampering attributes externally) is forbidden in OOP,
      but sufficient for my usecase here
      (just make sure I am 100% i no longer need to do prediction before deleting)

    Development script
    ------------------
    ``/home/takanori/work-local/tak-ace-ibis/python/pnc/protocodes/1026_proto_my_precomputed_rbfsvc_class.py``
    """

    def __init__(self, C=1.0, gamma=1):
        """ Above init values from ``SVC`` class"""

        #== below from BaseLibSVM==#
        #self.kernel = 'precomputed'
        #self.degree = degree
        self.gamma = gamma
        #self.coef0 = coef0 # <- only relevant for 'poly' and sigmoid' kernel
        #self.tol = tol
        self.C = C
        #self.nu = nu
        #self.epsilon = epsilon # <- for SVR me thinks...which is irrevalnt for this
        #self.shrinking = shrinking
        #self.probability = probability
        #self.cache_size = cache_size
        #self.class_weight = class_weight
        #self.verbose = verbose
        #self.max_iter = max_iter
        #self.random_state = random_state

    def fit(self, X, y,sample_weight=None):
        """Directly from ``BaseLibSVM`` docstring
        """
        from sklearn.svm import SVC

        # generate kernel matrix
        K = _rbf_kernel(X,gamma=self.gamma)
        svm = SVC(C=self.C, kernel='precomputed',gamma=self.gamma)
        svm.fit(K,y)

        # sadly, i can't think of a better way to store X....sounds wasteful of memory
        # but for now, resort to this
        #
        # After prediction is compleete, just do:
        # >>> del del clf.Xtr_
        # you shouldn't be able to do this in a real OOP, but for here it's fine...
        # just make sure I am 100% i no longer need to do prediction before deleting
        self.Xtr_ = X # <- this is needed to create the test kernel matrix for ``predict``
        self.svm_ = svm

        return self

    def predict(self, X):
        # precompute testing kernel matrix
        check_is_fitted(self, 'svm_')
        Kts = _rbf_kernel(X,self.Xtr_,gamma=self.gamma)
        return self.svm_.predict(Kts)

    def decision_function(self,X):
        # precompute testing kernel matrix
        check_is_fitted(self, 'svm_')
        Kts = _rbf_kernel(X,self.Xtr_,gamma=self.gamma)
        return self.svm_.decision_function(Kts)


class SpamFistaFlatWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper class for using SPAMS toolbox classifiers (single-class)

    UPDATE 11/04/2015
    -----------------
    Replaced attribute ``w_`` with ``coef_``, for consistency with scikit's
    convention.  I'll try to stick with this conventino with every methods
    I come up with from now on.

    http://scikit-learn.org/stable/developers/#rolling-your-own-estimator

    Parameters
    ----------
    w0 : ndarray
        (p x 1) initial estimate
    return_optim_info : bool (default: True)
        return optim info

        >>> print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' \\
        >>> %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))

    kwargs : kwargs
        bunch of kwargs that'll create the dict for the "param" variable in spams.fistaFlat
        (note: I didn't include the ADMM related ones, which are undocumented in SPAMS)

    kwargs (ones i care about)
    ------------------------------
        loss : str (default: 'logistic')
            (choice of loss, see above)
        regul : str (default: 'l1')
            (choice of regularization, see function proximalFlat)
        lambda1 : (default 1e-1)
            (regularization parameter)
        lambda2 : (default: 0)
            (optional, regularization parameter, 0 by default)
        lambda3 : (default: 0)
            (optional, regularization parameter, 0 by default)
        max_it : (default: 100)
            (optional, maximum number of iterations, 100 by default)
        intercept : (default: False)
            (optional, do not regularize last row of W, false by default).
        tol : (default: 1e-4
            (optional, tolerance for stopping criteration, which is a relative duality gap
            if it is available, or a relative change of parameters).
        verbose : bool (default: False)
            (optional, verbosity level, false by default)

    kwargs (ones i probably won't touch much)
    -------------------------------------
        pos : bool (default: False)
            (optional, adds positivity constraints on the coefficients)
        transpose : bool (default: False)
            (optional, transpose the matrix in the regularization function)
        size_group : (default: 1)
            (optional, for regularization functions assuming a group structure)
        groups : (int32, default: None)
            (optional, for regularization functions assuming a group
            structure, see proximalFlat)
        numThreads : (default: -1)
            (optional, number of threads for exploiting multi-core
            / multi-cpus. By default, it takes the value -1
            which automatically selects all the available CPUs/cores).
        it0 : (default: 10)
            (optional, frequency for computing duality gap, every 10 iterations by default)
        gamma : float (default: 1.5)
            (optional, multiplier for increasing the parameter L in fista, 1.5 by default)
        L0 : (default: 0.1)
            (optional, initial parameter L in fista, 0.1 by default,
            should be small enough)
        fixed_step : (default: False)
            (deactive the line search for L in fista and use L0 instead)
        ista : (default: False)
            (optional, use ista instead of fista, false by default).
        subgrad : (default: False)
            (optional, if not ista, use subradient descent instead of fista, false by default).
        a : (default: 1.0)
            (optional, if subgrad, the gradient step is a/(t+b)
        b : (default: 0.)
            (optional, if subgrad, the gradient step is a/(t+b)
            also similar options as proximalFlat
        linesearch_mode : (default: 0)
            (line-search scheme when ista=true)

            - 0 : default, monotonic backtracking scheme
            - 1 : monotonic backtracking scheme, with restart at each iteration
            - 2 : Barzilai-Borwein step sizes (similar to SparSA by Wright et al.)
            - 3 : non-monotonic backtracking
        compute_gram : (default: False)
            (optional, pre-compute X^TX, false by default).

    Methods
    --------
    - ``fit``
    - ``predict``
    - ``decision_function``

    Supported loss
    ----------------
      - if loss='square' and regul is a regularization function for vectors,
        the entries of Y are real-valued,  W = [w^1,...,w^n] is a matrix of size p x n
        For all column y of Y, it computes a column w of W such that
          w = argmin 0.5||y- X w||_2^2 + lambda1 psi(w)

      - if loss='square' and regul is a regularization function for matrices
        the entries of Y are real-valued,  W is a matrix of size p x n.
        It computes the matrix W such that
          W = argmin 0.5||Y- X W||_F^2 + lambda1 psi(W)

      - loss='square-missing' same as loss='square', but handles missing data
        represented by NaN (not a number) in the matrix Y

      - if loss='logistic' and regul is a regularization function for vectors,
        the entries of Y are either -1 or +1, W = [w^1,...,w^n] is a matrix of size p x n
        For all column y of Y, it computes a column w of W such that
          w = argmin (1/m)sum_{j=1}^m log(1+e^(-y_j x^j' w)) + lambda1 psi(w),
        where x^j is the j-th row of X.

      - if loss='logistic' and regul is a regularization function for matrices
        the entries of Y are either -1 or +1, W is a matrix of size p x n
          W = argmin sum_{i=1}^n(1/m)sum_{j=1}^m log(1+e^(-y^i_j x^j' w^i)) + lambda1 psi(W)

      - if loss='multi-logistic' and regul is a regularization function for vectors,
        the entries of Y are in {0,1,...,N} where N is the total number of classes
        W = [W^1,...,W^n] is a matrix of size p x Nn, each submatrix W^i is of size p x N
        for all submatrix WW of W, and column y of Y, it computes
          WW = argmin (1/m)sum_{j=1}^m log(sum_{j=1}^r e^(x^j'(ww^j-ww^{y_j}))) + lambda1 sum_{j=1}^N psi(ww^j),
        where ww^j is the j-th column of WW.

      - if loss='multi-logistic' and regul is a regularization function for matrices,
        the entries of Y are in {0,1,...,N} where N is the total number of classes
        W is a matrix of size p x N, it computes
          W = argmin (1/m)sum_{j=1}^m log(sum_{j=1}^r e^(x^j'(w^j-w^{y_j}))) + lambda1 psi(W)
        where ww^j is the j-th column of WW.

      - loss='cur' useful to perform sparse CUR matrix decompositions,
          W = argmin 0.5||Y-X*W*X||_F^2 + lambda1 psi(W)

    Supported regul
    ----------------
        if regul='l0'
            argmin 0.5||u-v||_2^2 + lambda1||v||_0
        if regul='l1'
            argmin 0.5||u-v||_2^2 + lambda1||v||_1
        if regul='l2'
            argmin 0.5||u-v||_2^2 + 0.5lambda1||v||_2^2
        if regul='elastic-net'
            argmin 0.5||u-v||_2^2 + lambda1||v||_1 + lambda1_2||v||_2^2
        if regul='fused-lasso'
            argmin 0.5||u-v||_2^2 + lambda1 FL(v) + ...
                              ...  lambda1_2||v||_1 + lambda1_3||v||_2^2
        if regul='linf'
            argmin 0.5||u-v||_2^2 + lambda1||v||_inf
        if regul='l1-constraint'
            argmin 0.5||u-v||_2^2 s.t. ||v||_1 <= lambda1
        if regul='l2-not-squared'
            argmin 0.5||u-v||_2^2 + lambda1||v||_2
        if regul='group-lasso-l2'
            argmin 0.5||u-v||_2^2 + lambda1 sum_g ||v_g||_2
            where the groups are either defined by groups or by size_group,
        if regul='group-lasso-linf'
            argmin 0.5||u-v||_2^2 + lambda1 sum_g ||v_g||_inf
        if regul='sparse-group-lasso-l2'
            argmin 0.5||u-v||_2^2 + lambda1 sum_g ||v_g||_2 + lambda1_2 ||v||_1
            where the groups are either defined by groups or by size_group,
        if regul='sparse-group-lasso-linf'
            argmin 0.5||u-v||_2^2 + lambda1 sum_g ||v_g||_inf + lambda1_2 ||v||_1
        if regul='trace-norm-vec'
            argmin 0.5||u-v||_2^2 + lambda1 ||mat(v)||_*
           where mat(v) has size_group rows

        if one chooses a regularization function on matrices
        if regul='l1l2',  V=
            argmin 0.5||U-V||_F^2 + lambda1||V||_{1/2}
        if regul='l1linf',  V=
            argmin 0.5||U-V||_F^2 + lambda1||V||_{1/inf}
        if regul='l1l2+l1',  V=
            argmin 0.5||U-V||_F^2 + lambda1||V||_{1/2} + lambda1_2||V||_{1/1}
        if regul='l1linf+l1',  V=
            argmin 0.5||U-V||_F^2 + lambda1||V||_{1/inf} + lambda1_2||V||_{1/1}
        if regul='l1linf+row-column',  V=
            argmin 0.5||U-V||_F^2 + lambda1||V||_{1/inf} + lambda1_2||V'||_{1/inf}
        if regul='trace-norm',  V=
            argmin 0.5||U-V||_F^2 + lambda1||V||_*
        if regul='rank',  V=
            argmin 0.5||U-V||_F^2 + lambda1 rank(V)
        if regul='none',  V=
            argmin 0.5||U-V||_F^2

        for all these regularizations, it is possible to enforce non-negativity constraints
        with the option pos, and to prevent the last row of U to be regularized, with
        the option intercept
    """

    def __init__(self,
                 w0=None,
                 return_optim_info=True,
                 loss="logistic",
                 regul="l1",
                 lambda1=1e-1,
                 lambda2=0.,
                 lambda3=0.,
                 max_it=100,
                 tol=1e-4,
                 L0=0.1,
                 fixed_step=False,
                 gamma=1.5,
                 a=1.0,
                 b=0.,
                 it0=10,
                 compute_gram=False,
                 resetflow=False,
                 verbose=False,
                 pos=False,
                 ista=False,
                 subgrad=False,
                 size_group=1,
                 groups = None,
                 transpose=False,
                 linesearch_mode=0,
                 numThreads =-1):
        self.w0 = w0
        self.return_optim_info = return_optim_info
        self.loss = loss
        self.regul = regul
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_it = max_it
        self.tol = tol
        self.L0 = L0
        self.fixed_step = fixed_step
        self.gamma = gamma
        self.a = a
        self.b = b
        self.it0 = it0
        self.compute_gram = compute_gram
        self.resetflow = resetflow
        self.verbose = verbose
        self.pos = pos
        self.ista = ista
        self.subgrad = subgrad
        self.size_group = size_group
        self.groups = groups
        self.transpose = transpose
        self.linesearch_mode = linesearch_mode
        self.numThreads = numThreads

    def _get_param(self):
        param={}
        param['loss'] = self.loss
        param['regul'] = self.regul
        param['lambda1'] = self.lambda1
        param['lambda2'] = self.lambda2
        param['lambda3'] = self.lambda3
        param['max_it'] = self.max_it
        param['tol'] = self.tol
        param['L0'] = self.L0
        param['fixed_step'] = self.fixed_step
        param['gamma'] = self.gamma
        param['a'] = self.a
        param['b'] = self.b
        param['it0'] = self.it0
        param['compute_gram'] = self.compute_gram
        param['resetflow'] = self.resetflow
        param['verbose'] = self.verbose
        param['pos'] = self.pos
        param['ista'] = self.ista
        param['subgrad'] = self.subgrad
        param['size_group'] = self.size_group
        param['groups'] = self.groups
        param['transpose'] = self.transpose
        param['linesearch_mode'] = self.linesearch_mode
        param['numThreads'] = self.numThreads
        return param

    def fit(self, X, y):
        import spams

        # convert data into fortran array
        if not X.flags['F_CONTIGUOUS']:
            X = np.asfortranarray(X)

        # Spam expects y to be a float
        y = y.astype(float)
        if len(y.shape) is 1:
            # SPAMS expect (n x 1) ndarray, not (n,)
            y = y[:,np.newaxis]
        if not y.flags['F_CONTIGUOUS']:
            y = np.asfortranarray(y)

        n,p = X.shape
        k = y.shape[1]

        self.n_ = n
        self.p_ = p
        self.k_ = k # <- # classes/tasks



        if self.w0 is None:
            # initial estimate
            w0 = np.zeros((p,k),order="FORTRAN")
        else:
            w0 = np.asfortranarray(self.w0)

        param = self._get_param()
        if self.return_optim_info:
            self.coef_, self.optim_info_ = spams.fistaFlat(
                y, X, w0, return_optim_info=True, **param)
        else:
            self.coef_ = spams.fistaFlat(
                y, X, w0, return_optim_info=False, **param)

        #=====================================================================#
        # convert back to C order array
        # http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html#changing-kind-of-array
        #=====================================================================#
        self.coef_ = np.ascontiguousarray(self.coef_)

        # if single-task/class, make coef_ shape as (p,), not (p,1)
        if self.coef_.shape[1] is 1:
            self.coef_ = self.coef_.ravel()

        self.nnz_ = np.count_nonzero(self.coef_)
        return self
    def predict(self, X):
        check_is_fitted(self, 'coef_')
        return np.sign(X.dot(self.coef_)).astype(int)
    def decision_function(self,X):
        check_is_fitted(self, 'coef_')
        return X.dot(self.coef_)


class LogisticGlmNet(BaseEstimator, ClassifierMixin):
    """ My wrapper class to LogisticNet object from glmnet_py

    Tested on ^1020_glmnet_predict_brute.ipynb

    GlmNet info (from class GlmNet in glmnet.py, though I modified portion for readability)
    -----------
    Glmnets are a class of predictive models. They are a regularized version
    of generalized linear models that combines the ridge (L^2) and lasso (L^1)
    penalties.  The general form of the loss function being optimized is:

    .. math::

        L(\\beta_0, \\beta_1, ..., \\beta_n) =
            Dev(\\beta_0, \\beta_1, ..., \\beta_n) +
            \\lambda * ( (1 - \\alpha)/2 * | \\beta |_2 + \\alpha * | \\beta |_1 )

    where Dev is the deviance of a classical glm, |x|_2 and |x|_1 are the L^2
    and L^1 norms, and :math:`\\lambda` and :math:`\\alpha` are tuning parameters:

      * :math:`\\lambda` controlls the overall ammount of regularization, and is usually
        tuned by cross validation.

      * :math:`\\alpha` controlls the balance between the L^1 and L^2 regularizers.

        In the extreme cases:

            - :math:`\\alpha` = 0 : Ridge Regression
            - :math:`\\alpha` = 1 : Lasso Regression

    All glmnet objects accept a value of alpha at instantiation time.  Glmnet
    defaults to fitting a full path of lambda values, from lambda_max (all
    parameters zero) to 0 (an unregularized model).  The user may also choose to
    supply a list of lambdas, in this case the default behavior is overriden and
    a glmnet is fit for each value of lambda the user supplies.

    The function Dev depends on the specific type of glmnet under
    consideration.  Different choices of Dev determine various predictive
    models in the glmnet family.  For details on the different types of
    glmnets, the reader should consult the various subclasses of GlmNet.

    Parameters
    -------------
    alpha : float (default 1.0)
        Relative weighting between the L1 and L2 regularizers.
        Value between 0. and 1. (default: 1.0)

        - alpha = 0 (ridge regression)
        - alpha = 1 (lasso)
    standardize : bool (default: False)
        standardize the predictors (I like to standardize variables explicitly,
        so default is set to False)
    threshold : (default 1e-4)
        Convergence threshold for each lambda.  For each lambda, iteration is
        stopped when imporvement is less than threshold.
    frac_lg_lambda : (default 1e-3)
        Control parameter for range of lambda values to search:

        .. math ::

            \\lambda_\\text{min} = \\text{frac_lg_lambda} *  (\\lambda_\\text{max})
        where \lambda_max is calcualted based on the data and the model type.
    n_lambdas:
        The number of lambdas to include in the grid search.

    Parameters below I don't fully understand
    ------------------------------------------
          * max_vars_largest:
              Maximum number of variables allowed in the largest model.  This
              acts as a stopping criterion.
          * max_vars_all:
              Maximum number of non-zero variables allowed in any model.  This
              controls memory alocation inside glmnet.
          * overwrite_pred_ok:
              Boolean, overwirte the memory holding the predictor when
              standardizing?
          * overwirte_targ_ok:
              Boolean, overwrite the memory holding the target when
              standardizing?
    """
    def __init__(self,alpha=1,standardize=False,threshold = 1e-4,
                 frac_lg_lambda=1e-3, n_lambdas=100):
        """ Overwrite some of the default constructor from GlmNet class"""
        self.alpha = alpha
        self.standardize = standardize
        self.threshold = threshold
        self.frac_lg_lambda = frac_lg_lambda
        self.n_lambdas = n_lambdas

        # These are GlmNet attributes I'll never touch the defaults
        self.max_vars_all = None
        self.max_vars_largest = None
        self.overwrite_pred_ok = False
        self.overwrite_targ_ok = False

    def _init_LogisticNet(self):
        """Return instance of LogisticNet object with the default constructor values"""
        from glmnet import LogisticNet
        lognet = LogisticNet(alpha = self.alpha,
                             standardize = self.standardize,
                             threshold = self.threshold,
                             frac_lg_lambda = self.frac_lg_lambda,
                             n_lambdas = self.n_lambdas,
                             max_vars_largest = self.max_vars_largest,
                             max_vars_all = self.max_vars_all,
                             overwrite_pred_ok = self.overwrite_pred_ok,
                             overwrite_targ_ok = self.overwrite_targ_ok)
        return lognet

    def fit(self,X,y,lambdas=None, weights=None, rel_penalties=None,
            excl_preds=None, box_constraints=None, offsets=None):
        """Fit a logistic or multinomial net model.

        Arguments
        -----------

          * X: The model matrix.  A n_obs * n_preds array.
          * y: The response.  This method accepts the response in two
            differnt configurations:

            - An n_obs * n_classes array.  In this case, each column in y must
              be of boolean (0, 1) type indicating whether the observation is
              or is not of a given class.
            - An n_obs array.  In this case the array must contain a discrete
              number of values, and is converted into the previous form before
              being passed to the model.

        Optional Arguments:
        --------------------
          * lambdas:
              A user supplied list of lambdas, an elastic net will be fit for
              each lambda supplied.  If no array is passed, glmnet will generate
              its own array of lambdas equally spaced on a logaritmic scale
              between \lambda_max and \lambda_min.
          * weights:
               An n_obs array. Sample weights. It is an error to pass a weights
               array to a logistic model.
          * rel_penalties:
              An n_preds array. Relative panalty weights for the covariates.  If
              none is passed, all covariates are penalized equally.  If an array
              is passed, then a zero indicates an unpenalized parameter, and a 1
              a fully penalized parameter.  Otherwise all covaraites recieve an
              equal penalty.
          * excl_preds:
              An n_preds array, used to exclude covaraites from the model. To
              exclude predictors, pass an array with a 1 in the first position,
              then a 1 in the i+1st position excludes the ith covaraite from
              model fitting.  If no array is passed, all covaraites in X are
              included in the model.
          * box_constraints:
              An array with dimension 2 * n_obs. Interval constraints on the fit
              coefficients.  The (0, i) entry is a lower bound on the ith
              covariate, and the (1, i) entry is an upper bound.  These must
              satisfy lower_bound <= 0 <= upper_bound.  If no array is passed,
              no box constraintes are allied to the parameters.
          * offsets:
              A n_preds * n_classes array. Used as initial offsets for the
              model fitting.

        Attributes returned upon fit
        -------------------------------
        coef_ : ndarray (p, n_lambdas)
            The coefficient of model matrix.  I had to wrestle around with
            ``_comp_coef`` from the original **LogisticNet fit attribute** to
            get this
        intecepts_ : ndarray of size (out_n_lambdas_,)
            A one dimensional array containing the intercept estiamtes for
            each value of lambda.
        out_lambdas_ : ndarray (out_n_lambdas_,)
            An array containing the lambda values associated with each fit model
        out_n_lambdas_ : int
            The number of lambdas associated with non-zero models (i.e.
            models with at least one none zero parameter estiamte) after
            fitting; for large enough lambda the models will become zero in
            the presense of an L1 regularizer.
        n_fit_obs_ : int
            The number of rows in the model matrix X.
        n_fit_params_ : int
            The number of columns in the model matrix X.
        n_passes_ : int
            The total number of passes over the data used to fit the model.
        null_dev_ : ndarray (out_n_lambdas_,)
           The devaince of the null (mean) model.
        exp_dev_ : ndarray (out_n_lambdas_,)
            The devaince explained by the model.
        error_flag_ : bool
            Error flag from the fortran code.

        Attributes below i have no idea what they are, but kept them just in case
        -------------
        n_comp_coef_ : ndarray (nlambdas,)
            The number of parameter estimates that are non-zero for some
            value of lambda.
        p_comp_coef_ : ndarray (p,)
            A one dimensional integer array associating the coefficients in
            _comp_coef to columns in the model matrix
            **(honestly, donno wtf this is)**
        indices_ : ndarray (size i have no idea)
            The same information as _p_comp_coef, but zero indexed to be
            compatable with numpy arrays.
        """
        n,p = X.shape

        # if labels in {-1,+1}, map to {0,1} as glmnet expects
        if np.array_equal(np.unique(y), np.array([-1,+1])):
            #warnings.warn('y is in {-1,+1}...internally mapping y to {0,1} '+
            #              'to conform with glmnet')
            y = (y+1)/2

        lognet = self._init_LogisticNet()
        #from glmnet_py import LogisticNet
        #lognet = LogisticNet(alpha=1)
        #lognet.fit(X,y)
        lognet.fit(X,y,lambdas=lambdas, weights=weights,
                   rel_penalties=rel_penalties,excl_preds=excl_preds,
                   box_constraints=box_constraints, offsets=offsets)

        nlambdas = lognet._out_n_lambdas
        #===== collect attributes after fit, but in a form that I'm more accustomed to =====#
        #| get coefficients from the _comp_coef attribute
        coef = np.zeros( (p, nlambdas) )
        for i in range(nlambdas):
            coef[:,i] = lognet.get_coefficients_from_lambda_idx(i)

        self.coef_       = coef
        self.intercepts_ = lognet._intercepts

        self.out_lambdas_ = lognet.out_lambdas
        self.out_n_lambdas_ = lognet._out_n_lambdas
        self.n_fit_obs_  = lognet._n_fit_obs
        self.n_classes      = lognet._n_classes
        self.n_fit_obs_     = lognet._n_fit_obs
        self.n_fit_params_  = lognet._n_fit_params
        self.n_passes_      = lognet._n_passes
        self.exp_dev_ = lognet.exp_dev
        self.error_flag_    = lognet._error_flag
        self.null_dev_ = lognet.null_dev

        # attributes below i have no idea what they are
        self.p_comp_coef_ = lognet._p_comp_coef
        self.n_comp_coef_   = lognet._n_comp_coef
        self.indices_     = lognet._indices

    def decision_function(self,X):
        check_is_fitted(self,'coef_')
        score = X.dot(self.coef_) + self.intercepts_
        score = -score
        return score

    def predict_proba(self,X):
        return 1./(1 + np.exp(-self.decision_function(X)))

    def predict(self, X):
        "Binary classification.  Return +1/-1 predictions"""
        ypred = (self.predict_proba(X) > 0.5).astype(int)

        # map from {0,1} to {-1, +1}
        ypred = 2*ypred - 1
        return ypred

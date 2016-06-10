# -*- coding: utf-8 -*-
"""
===============================================================================
Here I keep my *statistics* routines
===============================================================================
Created on June 6, 2016

@author: takanori
"""
import numpy as np
import pandas as pd
#%% === from tak.core.py ====
def ttest_twosample_fixnan(X,y):
    """Two sample ttest, with possible NaNs taken care of (created 11/20/2015)

    Scipy's ``sp.stats.ttest_ind`` is nice, but often yields NAN values.
    It's scary since I often presume with my analysis incognizant of the
    presence of NANs (eg, my custom gui screwed up, and it took me a long time
    that the presence of NAN in an input array was causing this).

    Here, ``tstats=NAN`` will be replaced by 0, and ``pval=NAN`` will be replaced
    by 1 (ie, least significant as possible)

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
    
    
def pvalue_ranksum(X,scores,impute = False,corr_type = 'pearson'):
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
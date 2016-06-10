# -*- coding: utf-8 -*-
"""
===============================================================================
Here I keep my ``deprecated`` codes here (for sake of backward compatibility)
===============================================================================
Created on June 6, 2016

@author: takanori
"""
from sklearn.utils import deprecated
#%%
from sklearn.utils import check_array
class DictionaryLearningOnline(TransformerMixin):
    """ Online Dictionary Learning (Createc 01/25/2016)

    Basically a wrapper class where sklearn's ``dict_learning_online`` is used
    for the ``fit`` method.  I created this since I wanted to integrate Dictionary Learning into
    scikit's **PipeLine** streamline.

    Here I inhereited the ``TransformerMixin``, which only requires me to
    define a ``fit`` and ``transform`` method.  **Important note**: I need to
    explicitly provide ``y=None`` in my ``fit`` and ``transform`` methods in
    order for the function to run without Exception when used with scikit's
    Pipeline

    Protocode: ``~/python/analysis/nmf/t_0125_e_try_onlineDL2.py``

    Parameters (from ``dict_learning_online``)
    ----------
    n_components : int,
        Number of dictionary atoms to extract.

    alpha : float,
        Sparsity controlling parameter.

    n_iter : int,
        Number of iterations to perform.

    batch_size : int,
        The number of samples to take in each batch.
    """
    def __init__(self, n_components=2, alpha=1, n_iter=100,batch_size=3):
        self.n_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self: object
            Returns the object itself
        """
        from sklearn.decomposition import dict_learning_online
        code_, dict_ = dict_learning_online(X,n_components=self.n_components,
                                            alpha=self.alpha,
                                            n_iter=self.n_iter)
        self.code_ = code_
        self.dict_ = dict_
        return self

#    def transform(self,X,y=None):
#        check_is_fitted(self, 'code_')
#        X = check_array(X)
#        code_ = check_array(self.code_)
#        return code_

    def transform(self,X,y=None):
        """ A complete clusterfuck...after headaches and debugging...seems
        like it's not a bug in my code...it was a due to some inexplicable
        fuckup in ``cross_val_score``....

        Basically the self.code_ did not reflect the transformed code on the
        test data for some reason...just refit here instead to fix this...

        Note that my **explicit** cross-validation where i iterate over the
        CV iterable object worked just fine....it's just a problem with
        ``cross_val_score``...which I sadly love to use
        """
        #check_is_fitted(self, 'code_')
        self.fit(X) #<- just refit the damn thing
#        print "---Xshape = {}".format(X.shape)
#        print "---code_.shape = {}".format(code_.shape)
        return self.code_
#%% === from tak.core.py ====
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
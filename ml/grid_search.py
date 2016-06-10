"""
===============================================================================
Old gridsearch code imported from tak.core (0608/2016)

TODO: 
- i haven't used these for a long time. ensure backward compatibility
===============================================================================
"""
import numpy as np
import time
import sys
import sklearn
from tak.core import print_time
#%% === old grid search routines (from tak.core.py; 06/08/2016) ===
"""TODO: 
- i haven't used these for a long time. ensure backward compatibility
"""
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
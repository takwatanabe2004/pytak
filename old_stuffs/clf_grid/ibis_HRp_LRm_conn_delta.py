import numpy as np
def get_gridsearch_classifier(clf_name):
    """ add docstring later
    """
    #%% "is_sparse" flag
    """note: i included this so method like Lasso, so I can obtain nnz after
             model fit.  for feature selection methods like ttest, i set this
             as False since here I know nnz prehand."""

    is_sparse = False # <- set this to True if method is sparse
    #%% ***START HUGE ELIF STATEMENT ****
    if clf_name == 'sklLogregL1':
        """ L1 logistic regression """
        np.random.seed(0) # <- needed to ensure replicability in LogReg fit model
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l1',random_state=0)
        param_grid = {'C':2.**np.arange(-8,15,1)}
        is_sparse = True
    elif clf_name == 'sklLinSvm':
        """ Linear SVM (hinge loss) """
        from sklearn.svm import LinearSVC
        clf = LinearSVC(loss='hinge')
        param_grid = {'C':2.**np.arange(-18,1,2)}
    elif clf_name == 'rbfSvm':
        """ RBF Kernel SVM """
        from tak.ml import PrecomputedRBFSVM
        clf = PrecomputedRBFSVM()
        param_grid = {'C':10.**np.arange(-1,10,1),
                      'gamma':10.**np.arange(-12,1,1)}
    elif clf_name =='ttestRbfSvm':
        from tak.ml import ttest_for_fs,PrecomputedRBFSVM
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import Pipeline
        ttest_fs = SelectKBest(score_func=ttest_for_fs)

        # setup pipeline of ttest_filter + RBF_SVM
        clf = Pipeline([('ttest', ttest_fs),('svm', PrecomputedRBFSVM())])

        # estimator parameters in a pipeline accessed as: <estimator>__<estimator>
        param_grid = {  'ttest__k':  (2**np.arange(4,11,1)).astype(int),
                          'svm__C': 10.**np.arange(-8,6,2),#^^^^^must be int, or scikit will complain
                      'svm__gamma': 10.**np.arange(-12,1,2)}
    elif clf_name == 'ttestLinSvm':
        # ttest + liblinear Pipeline (2 parameters)
        from tak.ml import ttest_for_fs
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import Pipeline
        ttest_fs = SelectKBest(score_func=ttest_for_fs)
        clf = Pipeline([('ttest', ttest_fs),
                        ('liblin', LinearSVC(loss='hinge')),])
        param_grid = {'ttest__k':  (2**np.arange(4,11.5,0.5)).astype(int), # must be int, or scikit will complain
                      'liblin__C': 2.**np.arange(-18,1,1),}
    elif clf_name == 'enetLogRegSpams':
        # Elastic-net Logistic Regression using my wrapper on SpamsToolbox (2 parameters)
        from tak.ml import SpamFistaFlatWrapper
        clf = SpamFistaFlatWrapper(loss='logistic',regul='elastic-net',max_it=400,tol=1e-3)
        param_grid = {'lambda1': 2.**np.arange(-12,1,1), # L1 penalty (lambda1 in SPAMS)
                      'lambda2': 2.**np.arange(-10,8,2),}  # L2 penalty (lambda2 in SPAMS)
        is_sparse = True
    elif clf_name == 'enetLogRegGlmNet':
        # Elastic-net Logistic Regression using my wrapper on SpamsToolbox (2 parameters)
        from tak.ml import LogisticGlmNet
        clf = LogisticGlmNet()
        param_grid = {'alpha':np.arange(0.1,1.1,0.1),
                      'lambdas':2.**np.arange(1,-11,-1)}
        is_sparse = True
    #%% === PCA stuffs...no interpretability, but see if accuracy improves ====
    elif clf_name == 'PcaLda':
        """ PCA + LDA (1 parameter) """
        from sklearn.lda import LDA
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline

        clf = Pipeline([('PCA', PCA()),
                        ('LDA', LDA(solver='lsqr',shrinkage='auto')),
                       ])
        param_grid = {'PCA__n_components':np.array([2,5, 10, 20,50, 100])}
    #=== PCA + LINSVM ===
    elif  clf_name == 'PcaLinSvm':
        from sklearn.svm import LinearSVC
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline

        clf = Pipeline([('PCA', PCA()),
                        ('SVM', LinearSVC(loss='hinge')),
                       ])
        param_grid = {'PCA__n_components':np.array([2,5, 10, 20,50, 100]),
                      'SVM__C':2.**np.arange(-14,3,2)}
    #%% PCA + RBFSVM
    elif  clf_name == 'PcaRbfSvm':
        #"""programs hangs at certain rbf bandwidth...skip"""
        #raise Exception(clf_name+" not worth running...")
        from tak.ml import PrecomputedRBFSVM
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline


        clf = Pipeline([('PCA', PCA()),
                        ('SVM', PrecomputedRBFSVM()),
                       ])
        param_grid = {'PCA__n_components':np.array([2,5, 10, 20,50, 100]),
                      'SVM__C': 10.**np.arange(-8,6,2),#^^^^^must be int, or scikit will complain
                      'SVM__gamma': 2.**np.arange(-12,1,2)}
    #%% ttest + LDA (for interpretability, I guess)
    elif clf_name == 'ttestLDA':
        from tak.ml import ttest_for_fs
        from sklearn.lda import LDA
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectKBest

        ttest_fs = SelectKBest(score_func=ttest_for_fs)

        clf = Pipeline([('ttest', ttest_fs),
                        ('LDA', LDA(solver='lsqr',shrinkage='auto')),
                       ])
        param_grid = {'ttest__k':(2**np.arange(3,10.5,0.5)).astype(int)}

    #%%______huge elif above is complete.  return ______
    return clf, param_grid, is_sparse


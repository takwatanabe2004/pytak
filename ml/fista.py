import numpy as np
from .util_ml import (
    logistic_loss,
    logistic_loss_grad,
    logistic_loss_lipschitz_constant,
    prox_l1,
    )
from scipy.linalg import norm
#import scipy as sp
from nilearn.decoding.fista import mfista
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted#, check_random_state


class LogRegElasticNetFista(BaseEstimator, ClassifierMixin):
    """ Logistic Regression Elastic-net using the mfista solver

    Minimizes the objective function::
            1 / n_samples * logistic_loss(X,y,w) +
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    Paramaters
    ------------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter.
    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    References
    ----------
    - https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/linear_model/coordinate_descent.py#L485
    - https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/linear_model/logistic.py#L925


    Mixins
    ------
    - http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    - http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
    """
    def __init__(self, alpha = 1.0, l1_ratio= 0.5, tol=1e-3,
                 verbose=0, warm_start=False):
        if not (l1_ratio >=0 and l1_ratio <=1):
            raise ValueError("l1_ratio must be between 0. and 1.")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.solver_info_ = None

    def fit(self, X, y):
        """ Fit model with mfista

        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target
        """
        n_features = X.shape[1]

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1-self.l1_ratio)

        # objective function
        obj_func = lambda w: logistic_loss(X,y,w) \
                             + l1_penalty*norm(w,1) \
                             + l2_penalty*norm(w,2)**2/2.

        # gradient of the smooth term
        f1_grad = lambda w: logistic_loss_grad(X,y,w) + l2_penalty * w

        # lipshictz contant of f1 grad
        lipschitz_constant = logistic_loss_lipschitz_constant(X) + l2_penalty

        # proximal operator for the non-smooth part
        def f2_prox(w,l, *args, **kwargs):
            return prox_l1(w,l*l1_penalty), dict(converged=True)

        if self.warm_start and self.solver_info_ is not None:
            #print 'warm start'
            coef, cost, solver_info = mfista(
                f1_grad, f2_prox, obj_func,lipschitz_constant,
                w_size = n_features,verbose=self.verbose, tol=self.tol,
                init=self.solver_info_)
        else:
            coef, cost, solver_info = mfista(
                f1_grad, f2_prox, obj_func,lipschitz_constant,
                w_size = n_features,verbose=self.verbose,tol=self.tol)

        self.coef_ = coef
        self.cost_ = cost
        self.solver_info_ = solver_info
        self.n_iter = cost.__len__()

        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        return np.sign(X.dot(self.coef_)).astype(int)

    def decision_function(self,X):
        check_is_fitted(self, 'coef_')
        return X.dot(self.coef_)


class LogRegGraphNetFista(LogRegElasticNetFista):
    def __init__(self, alpha = 1.0, l1_ratio= 0.5, tol=1e-3,verbose=0,
                 warm_start=False, C = None, C_lipschitz_squared = None):
        """

        Parameters
        -----------
        alpha : float
            Constant that multiplies the penalty terms. Defaults to 1.0

        l1_ratio : float between 0 and 1
            The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
            ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
            is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
            combination of L1 and L2.
        tol : float
            Tolerance on the (primal) cost function.
        C : sparse matrix, shape (n_edges, n_features)
            incidence matrix.
        C_lipschitz_squared : float
            squared lipschitz contant of C, ie, spectral norm of C.T.dot(C)

            note: this value is typically much smaller than the squared-lipschitz
            constant from the LogisticLoss that its impact on the Fista stepsize
            is negligible.
        """
        super(LogRegGraphNetFista, self).__init__(
            alpha=alpha, l1_ratio=l1_ratio, tol=tol,
            verbose=verbose, warm_start=warm_start)

        self.C = C

        if C_lipschitz_squared is None:
            # gradient of the smooth term
            # (note: usually this term is so small in comparison to lipschitz from
            # logistic_loss that it's impact in the mfista step size is negligible
            from scipy.sparse.linalg import svds
            C_lipschitz_squared = svds(C,k=1, return_singular_vectors=False)[0]**2

        self.C_lipschitz_squared = C_lipschitz_squared


    def fit(self, X, y):
        """ Fit model with mfista

        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target
        """
        n_features = X.shape[1]

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1-self.l1_ratio)

        # objective function
        obj_func = lambda w: logistic_loss(X,y,w) \
                             + l1_penalty*norm(w,1) \
                             + l2_penalty*norm(self.C.dot(w),2)**2/2.

        # lipshictz contant of f1 grad
        lipschitz_constant = logistic_loss_lipschitz_constant(X) + \
                             l2_penalty * self.C_lipschitz_squared

        # gradient of the smooth term
        L = self.C.T.dot(self.C) # laplacian matrix
        f1_grad = lambda w: logistic_loss_grad(X,y,w) + l2_penalty * L.dot(w)

        # proximal operator for the non-smooth part
        def f2_prox(w,l, *args, **kwargs):
            return prox_l1(w,l*l1_penalty), dict(converged=True)

        if self.warm_start and self.solver_info_ is not None:
            #print 'warm start'
            coef, cost, solver_info = mfista(
                f1_grad, f2_prox, obj_func,lipschitz_constant,
                w_size = n_features,verbose=self.verbose, tol=self.tol,
                init=self.solver_info_)
        else:
            coef, cost, solver_info = mfista(
                f1_grad, f2_prox, obj_func,lipschitz_constant,
                w_size = n_features,verbose=self.verbose,tol=self.tol)

        self.coef_ = coef
        self.cost_ = cost
        self.solver_info_ = solver_info
        self.n_iter = cost.__len__()

        return self
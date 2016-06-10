import numpy as np
#import scipy as sp
from scipy import linalg


def spectral_norm_squared(X):
    """Computes square of the operator 2-norm (spectral norm) of X

    This corresponds to the Lipschitz constant of the gradient of the
    squared-loss function:

        w -> .5 * ||y - Xw||^2

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      Design matrix.

    Returns
    -------
    lipschitz_constant : float
      The square of the spectral norm of X.

    """
    # On big matrices like those that we have in neuroimaging, svdvals
    # is faster than a power iteration (even when using arpack's)

    """tw: below is same as:
    sp.sparse.linalg.svds(Xtrz,k=1,return_singular_vectors=False)[0]**2
    """
    return linalg.svdvals(X)[0] ** 2


def logistic_loss_lipschitz_constant(X, add_intercept=False):
    """Compute the Lipschitz constant (upper bound) for the gradient of the
    logistic sum:

         w -> \sum_i log(1+exp(-y_i*(x_i*w + v)))

    """
    # N.B: we handle intercept!
    if add_intercept:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    return spectral_norm_squared(X)/4 # <- logistic loss, you can divide by 4


def squared_loss(X, y, w, compute_energy=True, compute_grad=False):
    """Compute the MSE error, and optionally, its gradient too.

    The cost / energy function is

        MSE = .5 * ||y - Xw||^2

    A (1 / n_samples) factor is applied to the MSE.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    w : ndarray shape (n_features,)
        Unmasked, ravelized weights map.

    compute_energy : bool, optional (default True)
        If set then energy is computed, otherwise only gradient is computed.

    compute_grad : bool, optional (default False)
        If set then gradient is computed, otherwise only energy is computed.

    Returns
    -------
    energy : float
        Energy (returned if `compute_energy` is set).

    gradient : ndarray, shape (n_features,)
        Gradient of energy (returned if `compute_grad` is set).

    """
    if not (compute_energy or compute_grad):
        raise RuntimeError(
            "At least one of compute_energy or compute_grad must be True.")

    residual = np.dot(X, w) - y

    # compute energy
    if compute_energy:
        energy = .5 * np.dot(residual, residual)
        if not compute_grad:
            return energy

    grad = np.dot(X.T, residual)

    if not compute_energy:
        return grad

    return energy, grad


def prox_l1(x, tau):
    """proximity operator for L1 norm (soft threshold)
    """
    y = np.sign(x)*np.maximum(0,np.abs(x) - tau)
    return y


def prox_l1_with_intercept(x, tau):
    """The same as prox_l1, but just for the n-1 components

    (from ``nilearn.decoding.proximal_operators.py``
    """
    x[:-1] = prox_l1(x[:-1], tau)
    return x


def sigmoid(t, copy=True):
    """Helper function: return 1. / (1 + np.exp(-t))"""
    if copy:
        t = np.copy(t)
    t *= -1.
    t = np.exp(t, t)
    t += 1.
    t = np.reciprocal(t, t)
    return t


def logistic_loss(X, y, w, add_intercept = False):
    """Compute the logistic function of the data: sum(sigmoid(yXw))

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector. Each entry must be +1 or -1.

    w : ndarray, shape (n_features,)
        Unmasked, ravelized input map.

    add_intercept : bool (default=False)
        Add intercept term, so w has shape (n_features+1,)

    Returns
    -------
    energy : float
        Energy contribution due to logistic data-fit term.
    """
    if add_intercept:
        z = np.dot(X, w[:-1]) + w[-1]
    else:
        z = np.dot(X, w)
    yz = y * z
    idx = yz > 0
    out = np.empty_like(yz)
    out[idx] = np.log1p(np.exp(-yz[idx]))
    out[~idx] = -yz[~idx] + np.log1p(np.exp(yz[~idx]))
    out = out.sum()
    return out


def logistic_loss_grad(X, y, w, add_intercept = False):
    """Computes the derivative of logistic"""
    if add_intercept:
        z = np.dot(X, w[:-1]) + w[-1]
    else:
        z = np.dot(X, w)
    yz = y * z
    z = sigmoid(yz, copy=False)
    z0 = (z - 1.) * y
    grad = np.empty(w.shape)

    if add_intercept:
        grad[:-1] = np.dot(X.T, z0)
        grad[-1] = np.sum(z0)
    else:
        grad = np.dot(X.T, z0)
    return grad
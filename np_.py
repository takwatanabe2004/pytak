# -*- coding: utf-8 -*-
"""
===============================================================================
Here I keep numpy/scipy/array related util functions
===============================================================================
Created on June 8, 2016

@author: takanori
"""
import numpy as np
import scipy as sp


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
#%% === image display related stuffs ===
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
#%% === Sparse related stuffs (including spectral graph theory) ===
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
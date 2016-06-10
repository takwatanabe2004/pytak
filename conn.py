# -*- coding: utf-8 -*-
"""
===============================================================================
Here I keep functions related to "connectivity" analysis.  I'll eventually
migrate all conn-type functions from ./core.py over here.
-------------------------------------------------------------------------------
===============================================================================
Created on Wed Feb 24 12:00:14 2016

@author: takanori
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


from scipy.spatial.distance import squareform as sqform

from .core import threshold_L1conc
from .plt_ import imconnmat, figure


#%%
def get_incmat_conn86(radius=50, return_coord=False):
    """Create adjacency matrix using scikit **(Created 12/05/2015)**
    """
    from sklearn.neighbors import radius_neighbors_graph
    from .np_ import adj2inc
    
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
        
#%%
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
    
    
def get_node_colorcodes():
    """ Helper for displaying connectivity in brain space.

    For now, node-color on connview represents "functional-system"
    """
    df_node = get_node_info86()

    #--- append a column indicating node-color ---#

    df_node['color'] = ''
    node_system=list(df_node['system'].unique())
    len(node_system)
    node_system

    """ Assign color-scheme, based on 'functional system'.
        Use same color scheme for system existing in Yeo-7.
        http://www.freesurfer.net/fswiki/CerebellumParcellation_Buckner2011
    """
    # yeo5: '#DCF8A4'
    color_code={'visual':'#781286', # purple: yeo1
                'motor and somatosensory':'#4682B4', # blue: somatomotor in yeo2
                'dorsal attention':'#00760E', # green yeo3
                'ventral attention':'#C63AFA', #'violet': yeo4
                'fronto-parietal':'#E69422', # 'orange': yeo6
    #            'default mode':'#CD3E4E', # red': yeo7
                'default mode':'red', # red': yeo7
                'other':'#E0E0E0', # gray
                'cingulo-opercular': '#00FFFF', # 'cyan
    #            'subcortical':'#FFFF00', # yellow
                'subcortical':'#DDDD00', # darker yellow
                'auditory':'#FF00FF' # magenta
            }

    for _system, _color in color_code.iteritems():
        idx = df_node.query('system == @_system').index
        df_node.ix[idx,'color'] = _color

    return df_node, color_code


def plt_basis_connectomes(W,idx,inv_fs=None,thresh=None,w=None):
    """ Show basis from NMF in connectome space.

    Parameters
    ----------
    W : ndarray of shape [n_features, n_basis]
        Columns represents the basis
    idx : int
        Integer of basis of interest
    inv_fs : lambda function
         Lambda function for **inverse-transform** operator to "undo"
         feature selection.  Created via:

         >>> inv_fs = lambda x: inverse_fs_transform(x, idx_fs, n_feat=p)
    thresh : float
        Threshold value (default = 0.0)

    History
    -------
    Created 02/08/2016

    Update 02/12/2016 - allowed the option of supplying string ended with % to
    show "concentration" value.

    Update 04/21/2016 - default for ``inv_fs`` is ``lambda x : x``
    """
    if inv_fs is None:
        inv_fs = lambda x:x # <- added 04/21/2016

    df_node,color_code = get_node_colorcodes()
    mni = get_mni_coord86()
    from nilearn import plotting

    figure('f')
    ax_top = plt.subplot2grid((2,3),(0,0),colspan=3)

    w_vec = inv_fs(W[:,idx])
    W_mat = sqform(w_vec)


    if thresh is None:
        thresh = 0.0
    elif isinstance(thresh,str) and thresh[-1]=='%':
        thresh = float(thresh[:-1])/100
        #print thresh
        _,thresh = threshold_L1conc(w_vec,thresh,True)
        thresh *= 1.0 # ensure float
        W_mat[W_mat < thresh] = 0
    else:
        thresh *= 1.0 # ensure float
        W_mat[W_mat < thresh] = 0

    if w is None:
        plotting.plot_connectome(W_mat, mni, node_size=80,axes=ax_top,
                                           node_color=df_node['color'].tolist(),
                                           )
    else:
        plotting.plot_connectome(w[idx]*W_mat, mni, node_size=80,axes=ax_top,
                                           node_color=df_node['color'].tolist(),
                                           )
    def remove_ticklabels():
        plt.gca().set_xticklabels('')
        plt.gca().set_yticklabels('')

    def remove_yticklabels():
        plt.gca().set_xticklabels('')

    df_node = get_node_info86()
    xtick_label = df_node['name_short'].values
    ytick_label = df_node['name_short'].values
    idx_L = range(43)
    idx_R = range(43,86)

    if w is None:
        plt.subplot(2,3,4), imconnmat(W_mat[np.ix_(idx_L,idx_L)],xtick_label[idx_L], ytick_label[idx_L],6)
        plt.title('left hemisphere')
        remove_yticklabels()
        plt.subplot(2,3,5), imconnmat(W_mat[np.ix_(idx_L,idx_R)],xtick_label[idx_L], ytick_label[idx_R],6)
        plt.title('inter hemisphere')
        remove_yticklabels()
        plt.subplot(2,3,6), imconnmat(W_mat[np.ix_(idx_R,idx_R)],xtick_label[idx_R], ytick_label[idx_R],6)
        plt.title('right hemisphere')
        remove_yticklabels()
    else:
        plt.subplot(2,3,4), imconnmat(w[idx]*W_mat[np.ix_(idx_L,idx_L)],xtick_label[idx_L], ytick_label[idx_L],6)
        plt.title('left hemisphere')
        remove_yticklabels()
        plt.subplot(2,3,5), imconnmat(w[idx]*W_mat[np.ix_(idx_L,idx_R)],xtick_label[idx_L], ytick_label[idx_R],6)
        plt.title('inter hemisphere')
        remove_yticklabels()
        plt.subplot(2,3,6), imconnmat(w[idx]*W_mat[np.ix_(idx_R,idx_R)],xtick_label[idx_R], ytick_label[idx_R],6)
        plt.title('right hemisphere')
        remove_yticklabels()


    plt.tight_layout()
    for ii, (_system, _color) in enumerate(color_code.iteritems()):
        plt.text(-2,-3-ii*3,_system,color=_color, fontsize=14)

    # concentration of weights
    wvec_thresh = w_vec.copy()
    wvec_thresh[w_vec<thresh] = 0
    conc = wvec_thresh.sum()/w_vec.sum()*100
    plt.suptitle('{:2}th basis (threshold={:.3f}, {:.1f}% of weights)'.\
        format(idx,thresh,conc),fontsize=20)


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

#%% ==== make shift ...not polished yet (02/21/2016) ===
def plt_basis_connectomes_connmat(W,idx,inv_fs=None,thresh=None,w=None):
    """ Same as plt_basis_connectomes_connmat, but show only connmat part
    """
    if inv_fs is None:
        inv_fs = lambda x:x # <- added 04/21/2016

    df_node,color_code = get_node_colorcodes()
#    mni = tw.get_mni_coord86()
#    from nilearn import plotting

    figure([   0,   26, 1600,  848])

    w_vec = inv_fs(W[:,idx])
    W_mat = sqform(w_vec)


    if thresh is None:
        thresh = 0.0
    elif isinstance(thresh,str) and thresh[-1]=='%':
        thresh = float(thresh[:-1])/100
        #print thresh
        _,thresh = threshold_L1conc(w_vec,thresh,True)
        thresh *= 1.0 # ensure float
        W_mat[W_mat < thresh] = 0
    else:
        thresh *= 1.0 # ensure float
        W_mat[W_mat < thresh] = 0

#    if w is None:
#        plotting.plot_connectome(W_mat, mni, node_size=80,axes=ax_top,
#                                           node_color=df_node['color'].tolist(),
#                                           )
#    else:
#        plotting.plot_connectome(w[idx]*W_mat, mni, node_size=80,axes=ax_top,
#                                           node_color=df_node['color'].tolist(),
#                                           )
    def remove_ticklabels():
        plt.gca().set_xticklabels('')
        plt.gca().set_yticklabels('')

    def remove_xticklabels():
        plt.gca().set_yticklabels('')

    def remove_yticklabels():
        plt.gca().set_xticklabels('')

    df_node = get_node_info86()
    xtick_label = df_node['name_short'].values
    ytick_label = df_node['name_short'].values
    xtick_label[[0,43]]='Bank of ST-Sulcus'
    ytick_label[[0,43]]='Bank of ST-Sulcus'
    idx_L = range(43)
    idx_R = range(43,86)

    fsize=7.5
    if w is None:
        plt.subplot(1,3,1), imconnmat(W_mat[np.ix_(idx_L,idx_L)],xtick_label[idx_L], ytick_label[idx_L],fsize)
        plt.title('left hemisphere')
#        remove_yticklabels()
        plt.subplot(1,3,2), imconnmat(W_mat[np.ix_(idx_L,idx_R)],xtick_label[idx_L], ytick_label[idx_R],fsize)
        plt.title('inter hemisphere')
        remove_xticklabels()
        plt.subplot(1,3,3), imconnmat(W_mat[np.ix_(idx_R,idx_R)],xtick_label[idx_R], ytick_label[idx_R],fsize)
        plt.title('right hemisphere')
#        remove_yticklabels()
    else:
        plt.subplot(1,3,1), imconnmat(w[idx]*W_mat[np.ix_(idx_L,idx_L)],xtick_label[idx_L], ytick_label[idx_L],fsize)
        plt.title('left hemisphere')
#        remove_yticklabels()
        plt.subplot(1,3,2), imconnmat(w[idx]*W_mat[np.ix_(idx_L,idx_R)],xtick_label[idx_L], ytick_label[idx_R],fsize)
        plt.title('inter hemisphere')
        remove_xticklabels()
        plt.subplot(1,3,3), imconnmat(w[idx]*W_mat[np.ix_(idx_R,idx_R)],xtick_label[idx_R], ytick_label[idx_R],fsize)
        plt.title('right hemisphere')
        remove_xticklabels()
#        remove_yticklabels()


#    plt.tight_layout()
#    for ii, (_system, _color) in enumerate(color_code.iteritems()):
#        plt.text(-2,-3-ii*3,_system,color=_color, fontsize=14)
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings

from .core import threshold_L1conc
from scipy.spatial.distance import squareform as sqform
#%% === platform dependent functions (cell began 2016-01-23) ===
from socket import gethostname
hostname = gethostname()

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
#%% === frequently used functions ===
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
        
        
def purge():
    """ My lazy close all """
    plt.close('all')

def imexp():
    """Fullscreen on my secondary monitor (10/01/2015)"""
    x,y,dx,dy=_screen_full
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(x,y,dx,dy)
#%% === statistics related plots ===
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
#%% === import from core.py ===
def plt_symm_xaxis():
    xlim_abs = max(np.abs(plt.gca().get_xlim()))
    plt.gca().set_xlim(-xlim_abs,xlim_abs)


def plt_colorbar_cmap(cmap=None, **kwargs):
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

def plt_cbar_ticks(mappable=None):
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

def plt_set_xyticks(xx,yy):
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
    
def plt_caxis_symm():
    """ symmetrize caxis in imshow (called vmin, vmax in python)

    Created 10/19/2015
    """
    vmin, vmax = plt.gci().get_clim()
    v = np.max( [np.abs(vmin), vmax] )
    plt.gci().set_clim(vmin=-v,vmax=v)

def plt_show_slices(data, x,y,z):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, 3)
   slices = [data[x,:,:],
             data[:,y,:],
             data[:,:,z]]
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
       
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
        cbar=fig.colorbar(plt.gci(), cax=cax, orientation='horizontal', ticks = plt_cbar_ticks())
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
#%% === modify figures already created (eg, using pandas/seaborn) ===
def plt_labelsize(labelsize):
    """Reminder for changing xlabel/ylabel size (06/01/2016)"""
    plt.tick_params(labelsize=20)


def plt_legend_move(xpos=1.2,ypos=0.7):
    """ Used since pandas plot's legend often hides the important part of image.

    Source
    -------
    >>> plt.legend(bbox_to_anchor=(xpos,ypos))
    """
    plt.legend(bbox_to_anchor=(xpos,ypos))


def plt_legend_hide(ax):
    """ Hide legend from axes.  Kept from mneumonic

    Source
    -------
    >>> ax.legend().set_visible(False)
    """
    ax.legend().set_visible(False)
    #ax.legend(bbox_to_anchor=(1.0, 0.5))


def plt_font_xticklabels(fs=14, rotation='vertical'):
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
        tick.label.set_rotation(rotation)

    # when rotation is applied on xtick, things look awkward as fuck.
    # apply fix.
    if rotation != 'vertical':
        plt_fix_xticklabels_rot()

    # sometimes xticklabels get truncated...so apply fix below
    #plt.gcf().subplots_adjust(bottom=0.4)


def plt_font_yticklabels(fs=14):
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)


def plt_fix_xticklabels_rot():
    """ Fix the odd looking xtick when ``rot`` is used in pandas plot

    **Created 11/18/2015**
    (see ``pnc_analyze_clf_summary_1118.py`` for usage)

    Example
    -------
    corrmat.mean().plot(kind='bar', fontsize=14, rot=30)
    pd_plot_fix_xtick_labels_rot()
    
    Ref
    ---
    - http://stackoverflow.com/questions/31859285/rotate-tick-labels-for-seaborn-barplot
    - http://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
    """
    #http://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
    plt.xticks(ha='right')
    #plt.xticks(rotation=45, ha='right',fontsize=32)
    #%% === below is old code (06/10/2016) ===
    """
    # list of text objects
    xtick_list = plt.gca().get_xticklabels()

    #text_list = [xtick.get_text() for xtick in xtick_list]
    #return text_list

    for xtick in xtick_list:
        xtick.set_ha('right')
    plt.draw()"""


#%% === pre 06/01/2016 ===
def plt_trinary_connmat(coef):
    # trinarize
    coef = threshold_L1conc(coef, 0.9)
    coef_tri=np.zeros(coef.shape)
    coef_tri[coef>0] = +1
    coef_tri[coef<0] = -1

    coef_tri = sqform(coef_tri)
    coef_tri[0,0]=2 # to adjust colormap to my taste

    imconnmat( coef_tri )
    #plt.colorbar()
    #plt.draw()


def plt_binary_connmat(coef):
    """Created (2016-01-23)"""
    # binarize
    coef = threshold_L1conc(coef, 0.9)
    coef_bin=np.zeros(coef.shape)
    coef_bin[coef!=0] = +1

    coef_bin = sqform(coef_bin)
    coef_bin[0,0]=2

    imconnmat( coef_bin )
    #plt.colorbar()
    #plt.draw()


def plt_max_fig(n_fig=5):
    """ Ensure there's only n_fig figures opened (Created 01/25/2016)

    Useful when I have to many figures opened at once
    """
    while len(plt.get_fignums()) > n_fig:
        plt.close(plt.get_fignums()[0])
#%% === gui and widgets ===
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
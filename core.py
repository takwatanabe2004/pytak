"""
===============================================================================
This will contain my "final" version utility function
-------------------------------------------------------------------------------
06/06/2016
- major code cleanup! see git commit on this date
===============================================================================
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import time
import inspect
#import re
#from pprint import pprint
import warnings
import sklearn
from sklearn.utils import deprecated

#from visualize import tw_slicer
#%% === platform dependent functions (cell began 2016-01-23) ===
from socket import gethostname
hostname = gethostname()
#%% ==== unsorted/uncategorized stuffs =====
def get_hostname():
    """ host == 'sbia-pc125' on my work computer'
    """
    import socket
    host = socket.gethostname()
    return host
    
    
def print_time(t):
    """
    Print time w.r.t. some time.time() instance

    Input
    --------
    t : time object
        Instance of the time.time() object

    Example
    ---------
    >>> import time
    >>> t = time.time()
    >>> # some code doing work
    >>> print_time(t)
    """
    print "Elapsed time: {:>5.2f} seconds".format(time.time()-t)
    sys.stdout.flush()


def reset():
    """Created 02/11/2016"""
    try:
        from IPython import get_ipython
        ipython=get_ipython()
        ipython.magic('reset -fs')
        plt.close('all')
    except Exception as inst:
        print type(inst) # the exception instance
        print "ipython function not loaded (ru running in python?)"


def get_ipython():
    """Created 04/08/2016

    More used to remind my self about IPython module
    Usage

    >>> ipython = tw.get_ipython()
    >>> ipython.magic('reset -fs')
    """
    import IPython
    return IPython.get_ipython()




def get_EDM(X):
    """ Simple wrapper to get EDM matrix.

    X : ndarray of shape [n,p]
        Data matrix

    EDM : ndarray of shape = [n,n]
        Euclidean distance matrix

    Created 02/03/2016
    """
    from sklearn.metrics.pairwise import pairwise_distances
    EDM = pairwise_distances(X,metric='euclidean')
    return EDM


def kde_1d(signal,x_grid=None):
    """ Return 1d kde of a vector signal (Created 01/24/2015)

    Todo: how are the kde's normalized?  (i want the kde to sum to 1....)

    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html

    Usage
    -----
    >>> x = np.linspace(0,1,401)
    >>> kde = tw.kde_1d(signal, x)
    >>> plt.plot(x, kde)
    >>> plt.grid('on')
    """
#    from scipy.stats.kde import gaussian_kde
#    if x is None:
#        x = np.linspace(0,1,401)
#
#    return gaussian_kde(signal)(x)
    from statsmodels.nonparametric.kde import KDEUnivariate
    kde = KDEUnivariate(signal)
    kde.fit()
    if x_grid is None:
        x_grid = np.linspace(0,1,401)
    #bin_space = x_grid[1]-x_grid[0]

    # kde estimate
    kde_est = kde.evaluate(x_grid)

    # normalize to pdf (need to come back on this....multiply by bin-spacing??)
    kde_est /= kde_est.sum()

    return kde_est, x_grid





def get_train_test_validation_index(n, ntrain, nval, ntest):
    """Added 12/15/2015"""
    assert n == (ntrain+nval+ntest)
    idx = np.random.permutation(n)
    idx_tr = np.sort( idx[:ntrain] )
    idx_vl = np.sort( idx[ntrain:ntrain+nval] )
    idx_ts =  np.sort( idx[ntrain+nval:] )

    df_index = pd.DataFrame(np.zeros((n,3)), columns=['train','validation','test'])
    for i in range(n):
        if i in idx_tr:
            df_index.iloc[i,0]=1
        elif i in idx_vl:
            df_index.iloc[i,1]=1
        elif i in idx_ts:
            df_index.iloc[i,2]=1
    return idx_tr, idx_vl, idx_ts, df_index

#def get_train_test_validation_index_df(n, ntrain, nval, ntest):
#    """Added 12/15/2015"""
#    assert n == (ntrain+nval+ntest)
#    idx_tr, idx_val, idx_ts = get_train_test_validation_index(n, ntrain, nval, ntest)
#
#    df_index = pd.DataFrame(np.zeros((n,3)), columns=['train','validation','test'])
#    for i in range(n):
#        if i in idx_tr:
#            df_index.iloc[i,0]=1
#        elif i in idx_val:
#            df_index.iloc[i,1]=1
#        elif i in idx_ts:
#            df_index.iloc[i,2]=1
#    return df_index




def unmask(w, mask):
    """Unmask an image into whole brain, with off-mask voxels set to 0.

    Parameters
    ----------
    w : ndarray, shape (n_features,)
      The image to be unmasked.

    mask : ndarray, shape (nx, ny, nz)
      The mask used in the unmasking operation. It is required that
      mask.sum() == n_features.

    Returns
    -------
    out : 3d of same shape as `mask`.
        The unmasked version of `w`
    """
    data = np.zeros(mask.shape)
    if mask.dtype != np.bool:
        mask.astype(bool)
    if np.count_nonzero(mask) != w.shape[0]:
        raise RuntimeError('Mask nnz should equal feature size!')
    data[mask.nonzero()] = w
    return data


def sort_list_str_case_insensitive(list_var):
    """ Sort list of string with case insensitivity (creatd 11/20/2015)

    Simple, but I often forget the syntax

    http://stackoverflow.com/questions/10269701/
    case-insensitive-list-sorting-without-lowercasing-the-result

    Internal
    --------
    >>> return sorted(list_var, key=lambda s: s.lower())
    """
    return sorted(list_var, key=lambda s: s.lower())

def keyboard(banner=None):
    import code
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return


def record_str_to_list(file_list_in):
    """
    Handy when dealing with a numpy record array of strings - situation
    encountered when I use ``scipy.io.matlab.loadmat`` to load a matlab
    struct variable.

    Used to be in ``tak.data_io.ndarray_string_to_list``
    """
    file_list = []
    # i can never memorize the awful code below, so made this function
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    return file_list

def threshold_L1conc(w,perc, return_threshold = False):
    """ Threshold strategy used mostly in my feature selection analysis.

    Created on 11/3/2015

    Idea:

    - I want to know where most of the "weights" in the weight vector ``w``
      is *contentrating* at.
    - This is done by computing the L1 norm of ``w``, and identify the set of
      coefficients that contribute to ``perc``% of the norm

    Paramters
    ----------
    w : ndarray
        ndarray you want to clip off
    perc : float (between 0. and 1.)
        Percentage level of what portion to cutoff at
    return_threshold : bool (default = False)
        Return the cutoff value or not

    Returns
    -------
    w_thresh : ndarray
        Thresholded version of the input ``w``, where
        norm(w_thresh)/norm(w)= ``perc``
    threshold : float (optional)
        Scalar value of threshold value returned if ``return_threshold`` is
        set to True

    Dev file
    ----------
    ``/home/takanori/work-local/tak-ace-ibis/python/analysis/__proto/proto_L1_threshold_1103.py``
    """
    w_abs = np.abs(w)

    #=== sort in descending order ===#
    idx_sort = np.argsort(w_abs)[::-1]
    w_abs_sort = w_abs[idx_sort]

    # index to get the original index-ordering after the sort takes place
    # http://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    idx_revsort = np.argsort(idx_sort)
    assert np.array_equal(w_abs, w_abs_sort[idx_revsort])

    # portion of L1 norm concentrated at each sort-points
    portion = np.cumsum(w_abs_sort)/w_abs.sum()

    # apply threshold
    w_abs_sort_thresh = w_abs_sort
    w_abs_sort_thresh[portion>perc] = 0

    # reverse back to original order
    w_abs_thresh = w_abs_sort_thresh[idx_revsort]

    # find the indices to drop off
    idx_to_keep = np.nonzero(w_abs_thresh)[0]

    # apply threshold to the original data (w/o the absolute value'ing)
    w_thresh = np.zeros( w.shape )
    w_thresh[idx_to_keep] = w[idx_to_keep]

    if return_threshold:
        threshold = np.abs(w_thresh[np.nonzero(w_thresh)[0]].min())
        return w_thresh, threshold
    else:
        return w_thresh


#%% === ipython notebook stuffs (ipynb stuffs) ===
def ipynb_print_toggle_code():
    """ print snippet for toggling code in ipynb

    References
    -------------
    - http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
    - http://stackoverflow.com/questions/9301466/python-formatting-large-text
    - http://stackoverflow.com/questions/10660435/pythonic-way-to-create-a-long-multi-line-string
    """
    import textwrap
    print textwrap.dedent("""
    from IPython.display import HTML

    HTML('''<script>
    code_show=true;
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    }
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
    """)

def ipynb_toc():
    """
    Copy and paste the following snippet as markdown cell at the top of notebook:

<h1 id="tocheading">Table of Contents</h1>
<div id="toc"></div>

    Then copy and paste the following as code cell

%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')

    See:
    https://github.com/kmahelona/ipython_notebook_goodies
    """
    print ipynb_toc.__doc__

def ipynb_print_toggle_code2():
    """ print snippet for toggling code in ipynb

    References
    -------------
    - http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
    - http://stackoverflow.com/questions/9301466/python-formatting-large-text
    - http://stackoverflow.com/questions/10660435/pythonic-way-to-create-a-long-multi-line-string
    """
    import textwrap
    str = textwrap.dedent("""
        from IPython.display import HTML

        HTML('''<script>
        code_show=true;
        function code_toggle() {
         if (code_show){
         $('div.input').hide();
         } else {
         $('div.input').show();
         }
         code_show = !code_show
        }
        $( document ).ready(code_toggle);
        </script>
        <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
        """)
    return str
#%% === General util functions ===
class Struct(object):
    """ Created 10/14/2015

    http://stackoverflow.com/questions/1305532/convert-python-dict-to-object

    >>> args = {'a': 1, 'b': 2}
    >>> s = Struct(**args)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def show_data_balance(y):
    """Print data balance info for binary label vector y

    Assumes y = +1 or y = -1

    (Created 11/06/2015)
    """
    n_pos = (y==+1).sum()
    n_neg = (y==-1).sum()
    n_ratio = 1.*n_pos/len(y)
    print("(+1) {:3} subjects, (-1) {:3} subjects ({:.2f}% is +1)".
            format(n_pos,n_neg,100*n_ratio))


def myprint(stuff):
    """

    https://docs.python.org/2/library/inspect.html
    https://docs.python.org/2/library/inspect.html#the-interpreter-stack

    Examples
    ------------
    >>> myprint(digits.keys())
    digits.keys() = ['images', 'data', 'target_names', 'DESCR', 'target']
    >>> myprint(digits.images.shape)
    digits.images.shape = (1797, 8, 8)

    About inspect stack
    ----------------------
    Each record is a tuple of six items:

    0. the frame object,
    1. the filename,
    2. the line number of the current line,
    3. the function name,
    4. a list of lines of context from the source code, and
    5. the index of the current line within that list.

    Example output
    ------------------
    >>> st = inspect.stack()[1]
    >>> for i,j in enumerate(st):
    >>>     print i,j
    0 <frame object at 0x400f2d0>
    1 <ipython-input-31-cfc6434307aa>
    2 27
    3 <module>
    4 [u'myprint(digits.images.shape']
    5 0
    """
# #     st = inspect.stack()
    st = inspect.stack()[1]
# #     print len(st)
#     for i,j in enumerate(st):
#         print i,j
# #     print(dir(stuff))
# #     print stuff.__getattribute__
    print "{} = {}".format(st[4][0][8:-2], stuff)
#    pprint("{} = {}".format(st[4][0][8:-2], stuff))


#%% === debugging related stuffs ===
def set_trace():
    """ from wes mckiness book """
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args, **kwargs):
    """ from wes mckiness book """
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)



# -*- coding: utf-8 -*-
"""
===============================================================================
Here I keep all my "pandas" related functionalities.  Function names are
prepended with "pd"

-------------------------------------------------------------------------------
TODO: migrate all pandas related functions from core.py
===============================================================================
Created on Wed Feb 10 15:42:57 2016

@author: takanori
"""
import numpy as np
import pandas as pd
import inspect

#%% === post 6/1/2016 ===
def pd_category_to_object(df):
    """ Convert categorical dtype to object.
    
    This allows me to view the dataframe in the Variable Explorer in Spyder.
    
    http://stackoverflow.com/questions/15891038/pandas-change-data-type-of-columns
    
    Created 06/08/2016
    
    Usage
    ------
    >>> from seaborn import load_dataset
    >>> df = load_dataset("tips")
    >>> df = tw.pd_category_to_object(df)
    """
    #%% here, convert dtype=category to dtype=object
    mask = (df.dtypes == 'category')

    # mask here is a Series
    for colname, is_category  in mask.iteritems():
        if is_category:
            df[colname] = df[colname].astype('object')
            
    """below doesn't work. spits out an exception when dtype is NOT category"""
#    for col_name, series  in df.iteritems():
#        if series.dtype == 'category':
#            df[col_name] = df[col_name].astype('object')
    return df

def pd_describe(df):
    """ My version of describe
    
    It prepends the column name with the corresponding index.
    Useful when using ``df.ix`` or ``df.iloc``
    """
    df_ = df.describe().T
    #%% prepend column index number
#    df_col = pd.DataFrame(df.columns.tolist())
    df_col = df.columns.tolist()
    tmp = []
    for i,name in enumerate(df_.index.tolist()):
        if name in df_col:
            idx = df_col.index(name)
            tmp.append('({}) {}'.format(idx,name))
    df_.index = tmp
    #%%
    df_.to_html('tmp.html')
    print '!google-chrome tmp.html'
    # prepend index number
#    df_.index = ['({}) {}'.format(i,str_) for i,str_ in enumerate(df_.index.tolist())]
    return df_


#%%
def pd_shift_column(df):
    """ Shift column in dataframe

    TODO: add integer argument allowing amount of shifting.  For now, just
    shift the last column circularly up front

    Help:
    http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns

    History
    -------
    Created 02/10/2016
    """
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df



def pd_add_to_first_column(df, col_name,col_val):
    """ Add new column at the first index

    Help:
    http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns

    History
    -------
    Created 02/10/2016
    """
    df[col_name] = col_val

    # move last column up front
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


def pd_prepend_col(df, col_name,col_val):
    """ Wrapper to ``pd_add_to_first_column``....which is too-looong....

    History
    -------
    Created 02/18/2016
    """
    df[col_name] = col_val

    # move last column up front
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

#%%=== old function imported from core.py ===
def pd_melt_for_sns(df, val_name=None,col_name=None):
    """ Convert DataFrame to a form amenable for sns display

    See ``pnc_analyze_clf_summary_1118.py`` for usage
    (named ``pd_mymelt`` in that file)

    """
    df = df.stack().reset_index(level=1)
    df = df.sort(columns=['level_1']) # "level_1" is default column-name from "reset_index"
    if col_name is not None:
        df = df.rename(columns={"level_1":col_name})
    if val_name is not None:
        cols = df.columns.tolist()
        cols[-1] = val_name
        df.columns = cols
    return df
    
def pd_compare_arrays(arr1,arr2):
    """ Created 11/09/2015

    I like to use this to compare two arrays, and see if equivalent.

    Example
    --------

    """
    df = pd.DataFrame([arr1,arr2,arr1==arr2],index=['arr1','arr2','equal']).T
    return df


def pd_sort_col(df):
    """Simple syntax I tend to forget too often"""
    cols = df.columns.tolist()
    cols.sort()
    return df[cols]

#%% === To review (consider dropping some...) ===
def pd_attr(obj, attrlist):
    """ Created 10/16/2015

    Add doc later.
    Use case in ^1016-try-glmnet-stability-selection.ipynb

    Example
    -------
    >>> lognet = LogisticNet(alpha=1)
    >>> lognet_dir_prefit = dir(lognet)
    >>> lognet.fit(Xz, y)
    >>> attrlist = list(set(dir(lognet)) - set(lognet_dir_prefit))
    >>> tw.pd_attr(lognet, attrlist)
    """
    # ensure list of strings are sorted
    attrlist.sort()

    #
    valuelist = []
    for x in attrlist:
        try:
            valuelist.append(getattr(obj, x))
        except Exception as err:
            valuelist.append('err: ' + str(err))

    typelist = []
    for x in attrlist:
        try:
            attr = getattr(obj, x)
            typelist.append(type(attr))
        except Exception as err:
            typelist.append('err: ' + str(err))

    sizelist = []
    for x in attrlist:
        attr = getattr(obj, x)
        try:
            if isinstance(attr, np.ndarray):
                sizelist.append(attr.shape)
            else:
                sizelist.append(len(attr))
        except Exception as err:
            sizelist.append('None')
            # sizelist.append('err: ' + str(err))

    return pd.DataFrame([attrlist, valuelist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T

def pd_fit_attr(str_list):
    """ Created 10/14/2015

    Prints attributes ending with underscore, but not beginning with underscore

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_fit_attr(clf)
    """
    #
    mylist = [x for x in dir(str_list) if not x.startswith('_') and x.endswith('_')]
    return pd.DataFrame(mylist)
    
    
def pd_fit_attr2(obj):
    """ Created 10/15/2015

    Extension of pd_fit_attr

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_fit_attr(clf)
    """
    #
    mylist = [x for x in dir(obj) if not x.startswith('_') and x.endswith('_')]
    attrlist = []
    for x in mylist:
        try:
            attrlist.append(getattr(obj, x))
        except Exception as err:
            attrlist.append('err: ' + str(err))

    typelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            typelist.append(type(attr))
        except Exception as err:
            typelist.append('err: ' + str(err))

    sizelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            if isinstance(attr, np.ndarray):
                sizelist.append(attr.shape)
            else:
                sizelist.append(len(attr))
        except Exception as err:
            sizelist.append('err: ' + str(err))

    return pd.DataFrame([mylist, attrlist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T


def pd_get_fit_attr(str_list):
    """ Created 10/14/2015

    Prints attributes ending with underscore, but not beginning with underscore.
    These are generally stuffs appended after .fit() method has taken place.

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> get_fit_attr(clf)

    See Also
    ---------
    :func:`pd_underscore`:
    """
    #
    mylist = [x for x in dir(str_list) if not x.startswith('_') and x.endswith('_')]
    return mylist

def pd_class_signature(ClassObject, index_name=None):
    """ Create DataFrame of class siganture and defaults.

    Created 10/16/2015
    Can be handy for ipython notebook.

    Example
    --------
    >>> from sklearn.linear_model import RandomizedLasso
    >>> pd_class_signature(RandomizedLasso)

    """
    argspec = inspect.getargspec(ClassObject.__init__)

    # print inspect.getargspec(pd_class_signature)
    # print inspect.getargvalues(inspect.currentframe())
    # args, _, _, _ = inspect.getargvalues(inspect.currentframe())
    # print args

    #argname = inspect.stack()[1][-2][0]
    """
    here argname will look something like this:
    >>> df1 = tw.pd_class_signature(RandomizedLasso)
        argname = 'df1 = tw.pd_class_signature(RandomizedLasso)'

    I want the shit inside the round bracket...use regex to extract that
    """
    # "group" to return the string matched by the RE
    #argname = re.search(r'\(\w+\)$', argname).group()
    # remove round bracket at beginning and end
    #argname = argname[1:-1]

    # print inspect.trace()

    # .defaults are tuples, so convert to list
    # (note: args[1:] to ignore "self" argument)
    # df = pd.DataFrame([argspec.args[1:], list(argspec.defaults)],
    #               index=['args', 'default'])
    if index_name is None:
        index_name = 'default'
    else:
        index_name = 'default ({})'.format(index_name)
    df = pd.DataFrame([list(argspec.defaults)], columns = argspec.args[1:],
                  index=[index_name])
    return df
    
    
def pd_check_pred(ytrue,ypred, err_type='clf'):
    """ Helper to check prediction error

    Created 04/10/2016

    Functionality self-explanatory from code
    """
    if err_type == 'clf':
        # classification
        err = (ytrue == ypred).astype(int)
        str_ = 'equal'
    else:
        # absolute error
        err = np.abs(ytrue-ypred)
        str_ = 'abs_err'
    return pd.DataFrame( [ytrue, ypred, err],
                          index=['ytrue','ypred',str_]).T

def pd_dir(obj, start_str='__'):
    """ Updated 04/02/2016

    Converted everything into a string!
    This way I can view everything in the variable explorer.

    Show 2column table of dir(obj)

    Input
    -----
    obj : object
        anything you can do dir(obj) on
    start_str : String (default: '_')
        filter option (useful to filter underscore ones)

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_dir(clf)
    """
    if start_str is not None:
        mylist = [x for x in dir(obj) if not x.startswith(start_str)]
    else:
        mylist = [x for x in dir(obj)]

    attrlist = []
    for x in mylist:
        try:
            attrlist.append(str(getattr(obj, x)))
        except Exception as err:
            attrlist.append('err: ' + str(err))

    typelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            typelist.append(str(type(attr)))
        except Exception as err:
            typelist.append('err: ' + str(err))

#    return pd.DataFrame([mylist, attrlist, typelist],
#                        index=['attr-name', 'attr-value', 'type']).T
    sizelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            if isinstance(attr, np.ndarray):
                sizelist.append(str(attr.shape))
            else:
                sizelist.append(str(len(attr)))
        except Exception as err:
            sizelist.append('err: ' + str(err))

    return pd.DataFrame([mylist, attrlist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T


def pd_dir_old(obj, start_str='__'):
    """ Created 10/15/2015

    Show 2column table of dir(obj)

    Input
    -----
    obj : object
        anything you can do dir(obj) on
    start_str : String (default: '_')
        filter option (useful to filter underscore ones)

    Example
    ------------
    >>> clf = svm.SVC().fit(iris.data,iris.target)
    >>> pd_dir(clf)
    """
    if start_str is not None:
        mylist = [x for x in dir(obj) if not x.startswith(start_str)]
    else:
        mylist = [x for x in dir(obj)]

    attrlist = []
    for x in mylist:
        try:
            attrlist.append(getattr(obj, x))
        except Exception as err:
            attrlist.append('err: ' + str(err))

    typelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            typelist.append(type(attr))
        except Exception as err:
            typelist.append('err: ' + str(err))

    sizelist = []
    for x in mylist:
        try:
            attr = getattr(obj, x)
            if isinstance(attr, np.ndarray):
                sizelist.append(attr.shape)
            else:
                sizelist.append(len(attr))
        except Exception as err:
            sizelist.append('err: ' + str(err))

    return pd.DataFrame([mylist, attrlist, typelist, sizelist],
                        index=['attr-name', 'attr-value', 'type', 'size']).T


def pd_fillna_taka(df):
    """Fill missing float values with mean, and "categorical" with mode

    Input
    ------
    df :
        data frame

    Output
    ------
    df :
        data frame

    fill_values :
        values filled in the nans

    """
    """Fill with mode value for dtype = "category" or "object" """
    df_out = df.copy()

    # loop through each column (may be better way, but will do)
    for colname in df_out.columns:
        cond1 = str(df_out[colname].dtype) in ['category', 'object']
        cond2 = df_out[colname].isnull().sum() != 0
        if cond1 and cond2:
            # set_trace()
            # mode()[0] since mode returns tuple
            df_out[colname].fillna(df_out[colname].mode()[0],inplace=True)
        elif cond2:
            df_out[colname].fillna(df_out[colname].mean(), inplace=True)
    return df_out

def pd_fillnan(df, p = 0.3):
    """ Fill in nan-values at random place with probability p

    Input
    -------
    df : data frame
        data frame object

    p : [0,1]
        probability of nan elements

    Output
    ------
    df: dataframe with nans

    Example
    ----------
    >>> df = pd.DataFrame(data = np.random.randn(10,6),columns=list('ABCDEF'))
    >>>
    >>> df['gender'] = pd.Series(['male']*7 + ['female']*3, dtype='category')
    >>> df['growth'] = pd.Series(['fast']*6 + ['slow']*2 + ['medium']*2)
    >>>
    >>> df = pd_fillnan(df,0.3)
    """
    df_out = df.copy() # <- create new object


    # for now, just usea loop
    for icol in xrange(df_out.shape[1]):
        #| print "({},{})".format(df.columns[icol], df.dtypes[icol])
        # insert nans at random row-indices
        mask = np.random.rand(df_out.shape[0])< p
        df_out.ix[mask, icol] = np.nan
    return df_out

def pd_fillna_mode(df):
    """Fill with mode value for dtype = "category" or "object" """
    df_out = df.copy()

    # loop through each column (may be better way, but will do)
    for colname in df_out.columns:
        cond1 = str(df_out[colname].dtype) in ['category', 'object']
        cond2 = df_out[colname].isnull().sum() != 0
        if cond1 and cond2:
            # set_trace()
            # mode()[0] since mode returns tuple
            df_out[colname].fillna(df_out[colname].mode()[0],inplace=True)
    return df_out

def pd_setdiff(df1,df2):
    """ Created 10/13/2015

    I like to use this to see what attributes are added when using the "fit"
    method in scikit.

    Example
    --------
    >>> # items prior to fitting
    >>> df_prefit = pd.DataFrame(dir(clf))
    >>>
    >>> # fit
    >>> clf.fit(Xtr, ytr)
    >>>
    >>> df_postfit = pd.DataFrame(dir(clf))
    >>>
    >>> pd_setdiff(df_postfit, df_prefit)
    """
    return pd.DataFrame(list(set(df1[0]) - set(df2[0])))


def pd_prepend_index_column(df, colname='i', col_at_end = False):
    """ Prepend a column with entries 0, .., df.shape[0]-1

    Handy for visualizing dataFrames as tables, and keeping track of which row you're on

    Usage
    ------
    >>> df = tw.pd_prepend_index_column(df, 'index_original')

    Parameters
    -----------
    df : DataFrame
        DataFrame object to prepend on
    colname : string (default = 'index_original')
        Column name to assign on the prepended column
    col_at_end : bool (default = False)
        Append column at the end (although I don't usually use this, since
        it's easier to do "sanity-checks" by having the index-column next to the
        DataFrame's Index object)
    """
    df[colname] = range(df.shape[0])

    if col_at_end:
        return df

    # bring this column up front
    # http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    return df[cols]
    

def pd_get_column_info(df):
    """ Get column info of a DataFrame...as a DataFrame!

    Migrated from ``data_io.get_df_column_info`` on 11/05/2015

    I may pad on more columns to the output DataFrame in the future
    """
    fields = pd.DataFrame(
                 {'columns':df.columns,
                  'dtypes':df.dtypes,
                  'nan_counts':df.isnull().sum(),
                  })
    fields = fields.reset_index(drop=True)
    fields['nan_rate'] = 1.*fields['nan_counts']/df.shape[0]

    return fields
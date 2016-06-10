# -*- coding: utf-8 -*-
"""
===============================================================================
Code snippets. 

WARNING: never to be loaded. i just keep it at this file-location for
convenience sake.
-------------------------------------------------------------------------------
I try my best to make the code below *self-contained* in the sense that
I can run the code block *cell-wise* in spyder. I've found it helpful to be
able to step through the code line-by-line (although being able to view
variables, especially dataframes in the spyder variable-explorer is helpful)
-------------------------------------------------------------------------------
Function defnition doen't mean anything; it just makes code-folding in spyder.
===============================================================================
Created on May 28, 2016

@author: takanori
"""
#==============================================================================
#------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import nibabel as nib

import seaborn.apionly as sns
import tak as tw
#%% === general handy things ===
def check_if_numeric():
    # http://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
    isinstance(x, (int, long, float, complex))
    
    
#%% === numpy things ===
def np_get_max_index(acc_grid):
    # Getting indices of max/min in ndarray
    #http://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurences-of-the-maximum
    print np.unravel_index(acc_grid.argmax(), acc_grid.shape)
    print acc_grid.argmax()

    """ in case you care about ties! """
    print np.argwhere(acc_grid == acc_grid.max())
#%% === machine learning ===
""" Here i'm just defining dummy variables so that editor code-analysis won't
    fire warnings"""
X = np.random.randn(500,10)
y = np.sign(np.random.randn(500))

from sklearn.svm import LinearSVC
clf = LinearSVC(C=1e-1)

#%%
def ml_plot_roc(ytrue,score):
    from sklearn.metrics import roc_curve

    #========================================================================#
    # classification summary
    #========================================================================#
    fpr,tpr,_ = roc_curve(ytrue, score)
    plt.plot(fpr,tpr,label='Linear SVM')
    plt.plot([0, 1], [0, 1], 'k--',label='random',lw=1)
    plt.legend(loc='best')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

def ml_stratified_cv():
    #from sklearn.utils import check_random_state
    #rng = check_random_state(0)
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()

    flag_scale=True

    cv = StratifiedKFold(y, n_folds = 10, shuffle=True)
    ytrue,ypred,score = [],[],[]
    for itr, its in cv:
        Xtr, ytr = X[itr], y[itr]
        Xts, yts = X[its], y[its]

        if flag_scale:
            scaler.fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xts = scaler.transform(Xts)

        clf.fit(Xtr,ytr)
        ypr = clf.predict(Xts)
        sco = clf.decision_function(Xts)

        ytrue.append(yts)
        ypred.append(ypr)
        score.append(sco)

    ytrue = np.concatenate(ytrue)
    ypred = np.concatenate(ypred)
    score = np.concatenate(score)

    print tw.clf_results_extended(ytrue,score)

def ml_train_test_split():
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()

    Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=0.3)
    flag_scale=True

    if flag_scale:
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xts = scaler.transform(Xts)

def ml_precompute_kernel_matrix(Xtr,Xts,gam):
    from sklearn.metrics.pairwise import rbf_kernel
    Ktr = rbf_kernel(Xtr,gamma=gam)
    Kts = rbf_kernel(Xts,Xtr,gamma=gam)
#%% === pandas ===
def pd_useful_functions_showing_basic_statistics(df):
    #%%
    #from seaborn import load_dataset
    import seaborn.apionly as sns
    
    titanic = sns.load_dataset("titanic")
    tips = sns.load_dataset("tips")
    iris = sns.load_dataset("iris")
    df = tw.pd_category_to_object(tips)
    #%% basics
    # shows memory usage as well here
    df.info()

    df.describe()

    # http://stackoverflow.com/questions/22128218/pandas-how-to-apply-multiple-functions-to-dataframe
    df.groupby(lambda foo: 0).agg(['count','mean','std',len,np.var]).T

    # randomly select rows
    print tips.sample(frac=1./5).describe()
    #%% little more sophisticated ones
    # creates a hierarchical index
    df_ = df.groupby(['sex','smoker']).agg(['count','mean','std'])
    print df_.T

    # pandas: different ways of boolean selection
    # (I personally like ``query`` the best)
    tips[ tips['sex'].isin(['Female'])].describe()
    tips[ tips['sex'] == 'Female' ].describe()
    tips.query('sex == "Female"').describe()
    tips.query('sex != "Male"').describe()
    tips.query('sex == "Male"').describe()
    tips.query(' (sex == "Male") & (ilevel_0 < 10) ')

    #%%
    tw.figure()
    tips.query('sex=="Female"')['total_bill'].hist(label='Female',alpha=0.4)
    tips.query('sex=="Male"')['total_bill'].hist(label='Male',alpha=0.4)
    plt.legend()
    #%%
    

def pd_indexing_and_boolean_selection_snippets():
    #%% basics
    import seaborn.apionly as sns
    tips = sns.load_dataset("tips")
    
    # filtering
    tips['sex'].isin(['Female'])
    tips['sex'].isin(['Female'])

    tips.query('sex == "Female" and tip < 3.5')
    tips.query('sex == "Female" and smoker == "Yes"')
    #%% where vs boolean (i probably won't ever use ``where``)
    # (i don't see when i'll ever use ``where``)
    idx1 = tips['sex'].where(tips['sex'] == 'Female') # <- returns a NAN
    idx2 = tips['sex'] == 'Female'                    # <- returns True/False
    #%% ``query`` vs ``isin`` (isin returns a boolean mask)
    # get all rows where columns "sex" and "smoker" have overlapping values
    tips.query('sex in smoker')
    tips[tips.sex.isin(tips.smoker)] # equivalent pythonic syntax...

    tips.query('sex not in smoker')
    tips[~tips.sex.isin(tips.smoker)] # equivalent pythonic syntax...
    #%% list-expressions also works in query
    tips.query('sex == "Male" and day in ["Sun","Sat"]')
    tips.query('sex in "Male" and day == ["Sun","Sat"]')
    tips.query('sex == "Male" and day not in ["Sun","Sat"]')
    tips.query('sex == "Male" and day != ["Sun","Sat"]')


def pd_groupby_demo(df):
    #%% http://pandas.pydata.org/pandas-docs/stable/cookbook.html
    grades = [48,99,75,80,42,80,72,68,36,78]
    
    df = pd.DataFrame( {'ID': ["x%d" % r for r in range(10)],
                        'Gender' : ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
                        'ExamYear': ['2007','2007','2007','2008','2008','2008','2008','2009','2009','2009'],
                        'Class': ['algebra', 'stats', 'bio', 'algebra', 'algebra', 'stats', 'stats', 'algebra', 'bio', 'bio'],
                        'Participated': ['yes','yes','yes','yes','no','yes','yes','yes','yes','yes'],
                        'Passed': ['yes' if x > 50 else 'no' for x in grades],
                        'Employed': [True,True,True,False,False,False,False,True,True,False],
                        'Grade': grades})
    
    print df
    
    #%%
    tmp=df.groupby('ExamYear')
    print tmp.groups
    print tmp.get_group('2007')
    #%%
    df_=df.groupby('ExamYear').agg({'Participated': lambda x: x.value_counts()['yes'],
                        'Passed': lambda x: sum(x == 'yes'),
                        'Employed' : lambda x : sum(x),
                        'Grade' : lambda x : sum(x) / len(x)})
    print df_
    #%%


def pd_fillnan(df):
    """http://pandas.pydata.org/pandas-docs/stable/missing_data.html"""
    #%% fill NANs by column-wise means
    dff = pd.DataFrame(np.random.randn(10,3), columns=list('ABC'))
    dff.iloc[3:5,0] = np.nan
    dff.iloc[4:6,1] = np.nan
    dff.iloc[5:8,2] = np.nan
    print dff
    
    print dff.mean()
    print dff.fillna(dff.mean())
    print dff.fillna(dff.mean()['B':'C'])
    
    # drop rows with nan
    print dff.dropna()
    #%%


def pd_replace_val():
    """
    http://pandas.pydata.org/pandas-docs/stable/missing_data.html#replacing-generic-values
    """
    #%% replace string value
    df = tw.pd_category_to_object(sns.load_dataset("tips"))
    df['sex'].replace('[fF]emale','girl',regex=True)
    df['sex'].replace('Female','girl')
    #%% replace numberical
    ser = pd.Series([0., 1., 2., 3., 4.])
    print ser
    
    print ser.replace(0, 5)
    print ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    print ser.replace({0: 10, 1: 100}) # <- i like this the best
    #%% numeric replace with DF
    df = pd.DataFrame(np.random.randn(10, 2))
    df[np.random.rand(df.shape[0]) > 0.5] = 1.5
    print df.replace(1.5,np.nan)
    
    print df.replace({1.5:np.nan, df.values[0,0]:'a'})
    #%%

def pd_apply_map_applymap_demos(df):
    """
    ``apply`` applies a function to a series.
    ``applymap`` is more of a ufunc.
    ``map`` is same as ``applymap``, but is for Series
    """
    #%%
    df = pd.DataFrame(np.arange(0,12).reshape(4,3),columns = list('ABC'))
    print df
    
    # dummy variable x is for Series
    print df.apply(lambda x: x.max() - x.min())
    print df.apply(lambda x: x.max() - x.min(),axis=1)
    print df.apply(lambda x: 2**x) # <- this is elementwise
    print df.apply(lambda x: x.idxmax())
    print df.apply(pd.value_counts)
    print df.apply(np.cumsum)
    #%%
    # we can also pass a Series method!
    df = pd.DataFrame(np.random.uniform(0,5.,(8,5)))
    df.ix[[2,3],[4,4]] = np.nan
    print df
    df.apply(pd.Series.interpolate)

def pd_choose_column_by_dtype():
    #%%
    from seaborn import load_dataset
    tips = load_dataset("tips")
    print tips.get_dtype_counts()
    df_categorical = tips.ix[:,tips.dtypes == 'category']     # <- pythonic
    df_categorical2= tips.select_dtypes(include=['category']) # <- more natural
    df_categorical.equals(df_categorical2) # pd.DataFrame.equals method to check equivalewcew
    #%%

def pd_cut_qcut():
    #%%
    from seaborn import load_dataset
    tips = load_dataset("tips")
    print tips.get_dtype_counts()

    # create a category field from continuous data using "cut"
    bins = range(0,101,10)
    df_categorical['tip_range'] = pd.cut(tips['tip'], bins = bins)
    df_categorical['bill_quantile'] = pd.qcut(tips["total_bill"], q=[0, 0.25,.5,.75,1])
    # create categorical data using "qcut" (quantile cut
    #%%

def pd_change_col_dtypes():
    #%%
    from seaborn import load_dataset
    tips = load_dataset("tips")
    print tips.get_dtype_counts()
    #%% here, convert dtype=category to dtype=object
    """ this way, we can view the DF in spyder variable explorer """
    #http://stackoverflow.com/questions/15891038/pandas-change-data-type-of-columns
    mask = tips.dtypes == 'category'

    # mask here is a Series
    for colname, is_category  in mask.iteritems():
        if is_category:
            tips[colname] = tips[colname].astype('object')
    #%%

def pd_combining_dataframes():
    """
    http://chrisalbon.com/python/pandas_join_merge_dataframe.html
    """
    #%% create 3 DFs
    raw_data = {
            'subject_id': ['1', '2', '3', '4', '5'],
            'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
            'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
    df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
    df_a
    
    
    raw_data = {
            'subject_id': ['4', '5', '6', '7', '8'],
            'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
            'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
    df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
    df_b

    raw_data = {
            'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
            'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
    df_c = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
    df_c 
    #%% using ``concat`` (simple appending of DF with compatible size)
    # join dataframes along rows
    print pd.concat([df_a,df_b])
    
    # join dataframes along cols
    print pd.concat([df_a,df_b],axis=1)
    #%% ``merge`` (database-style join operation by col/ind)
    df_new = pd.concat([df_a,df_b])
    print df_new
    print df_c
    
    # merge two DFs along the subject_id value
    print pd.merge(df_new,df_c,on='subject_id')
    
    # merge via outer join
    print pd.merge(df_new, df_c, on='subject_id', how='outer')
    #%%
    df_pred = df_pred.append(tw.clf_results_extended(yts,clf.predict(Xts)),
                             ignore_index=True)


def pd_merge_join_concat_append(df):
    """http://pandas.pydata.org/pandas-docs/stable/merging.html

    concat vs append (to add to current df)
    ----------------
    - ``append`` simply concatenates along the index (axis=0)
    - ``append`` is part of a DataFrame method
    
    >>> df1.append(df2) # valid
    >>> # can't do this => df.concat(df2)
    
    merge vs join (database-style merging)
    -------------
    both are for database-style merging, ````join```` uses the index as merge-key by default.
    
    """
    #%%
    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
                        
                        
    df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])

    df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])
    #%%
    print pd.concat([df1,df2,df3])
    
    # create hierarchical index
    df_new = pd.concat([df1,df2,df3],keys=['df1','y','z'])
    print df_new
    print df_new.loc['df1']
    
    # concat columns (join on index)
    print pd.concat([df1,df2,df3],axis=1)
    
    # concat columns (reset index so it'll join horizontally)
    print pd.concat([df1.reset_index(drop=True),df2.reset_index(drop=True),df3.reset_index(drop=True)],axis=1)
    #%% ``inner`` concat
    df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                        'D': ['D2', 'D3', 'D6', 'D7'],
                        'F': ['F2', 'F3', 'F6', 'F7']},index=[2, 3, 6, 7])
    print pd.concat([df1,df4]) # default is "outer" join
    print pd.concat([df1,df4],join='inner')
    #%% use of ``join_axes``
    print pd.concat([df1,df4],axis=1)
    print pd.concat([df1,df4],axis=1, join_axes=[df1.index])
    #%%
    
def pd_counting_functionalities():
    # counts over categorical stuffs over a series
    df.ix[:,10].value_counts()
    df.ix[:,10].value_counts(sort=False)

    df.brief_parent_metacog_tscore.isnull().sum() # count nulls
    df.brief_parent_metacog_tscore.isnotnull().sum() # count available
    df.brief_parent_metacog_tscore.count().sum() # count available

def pd_set_default_options():
    # keep padding on if i find other setup useful
    pd.set_option('precision', 5)
    
    
def pd_duplicates():
    """ pg 194 wesner book"""
    #%%
    df = pd.DataFrame({'k1':['one']*3+['two']*4,'k2':[1,1,2,3,3,4,4]})
    print df
    print df.duplicated()
    print df.drop_duplicates()
    print df.drop_duplicates(keep='last') # keep the last one
    #%%
    
def pd_query_vs_isin(df):
    #================query with "in" and "not" operators =====================#
    # (provides a compact syntax for calling "isin" method in dataframe)
    #-------------------------------------------------------------------------#
    # all these can be done using "isin" and "~isin", but nicer to read
    #+========================================================================#
    # get all rows where columns "a" and "b" have overlapping values

    #%%
    from seaborn import load_dataset
    df = load_dataset("tips")
    df = tw.pd_category_to_object(df)
    #%%
    df.query('sex in smoker')      # <- concise and readable
    df[df.sex.isin(df.smoker)] # equivalent pythonic syntax...
    
    df.query('sex not in smoker')
    df[~df.sex.isin(df.smoker)] # equivalent pythonic syntax...
    
    df.query('sex == "Female" and tip < 3.5')
    df.query('sex == "Female" and smoker == "Yes"')
    
    
    # list-expressions also works in query
    df.query('sex == "Male" and day in ["Sun","Sat"]')
    df.query('sex in "Male" and day == ["Sun","Sat"]')
    df.query('sex == "Male" and day not in ["Sun","Sat"]')
    df.query('sex == "Male" and day != ["Sun","Sat"]')
#%% -- row operator --
def pd_sort_by_row(df):
    # http://chrisalbon.com/python/pandas_sorting_rows_dataframe.html
    # http://stackoverflow.com/questions/17618981/how-to-sort-pandas-data-frame-using-values-from-several-columns
    df_=df_pnc.sort_values(by=['ID'])
    df_=df_pnc.sort_values(by=['ID'],ascending=False) # in descending order
#%% -- column operator ---
def pd_rename_columns(df):
    # rename age string
    df.rename(columns = {'SCAN_AGE_YEARS':'age'}, inplace=True)

def pd_remove_columns(df):
    # to delete multiple columns, need to use "drop" function
    del df['colname'] # <- this you can only do one at a time
    
    df.drop(['x','y'], axis=1)
    df.drop(df.columns[1:], axis=1)
#%% === plotting (pyplot) ===
def plt_change_ticksize():
    plt.tick_params(labelsize=7) # change xy-label tick sizes

def plt_fix_ticks():
    import seaborn as sns
    sns.set_style("white")
    plt.imshow(scores_mean, sns.cubehelix_palette(light=1, as_cmap=True))

    # ah, have to use xticks to get the label
    plt.xticks( range(gam_len), param_grid['gamma'], rotation=30 )
    plt.yticks( range(Clen), param_grid['C'])

def plt_legend_location():
    plt.legend(bbox_to_anchor=(1.25, 0.7))
    plt.gca().legend( bbox_to_anchor=(1.25, 0.7) )

#%% === os/sys ===
def os_homedir():
    from os.path import expanduser
    home = expanduser("~")
    print home

    pyname = os.path.basename(__file__)
    print "pyname = " + pyname

def os_envvars():
    os.environ.keys()
    os.environ['PYTHONPATH']
    os.getenv('PYTHONPATH')

def os_bash_command():
    #%% shell commands (os.system doesn't return output)
    # https://docs.python.org/2/library/subprocess.html#subprocess.check_output
    import subprocess

    print subprocess.check_output(['ls', '-l'])
    # https://docs.python.org/2/library/subprocess.html#subprocess.check_output
    # On Unix with shell=True, the shell defaults to /bin/sh
    print subprocess.check_output('echo ~', shell=True)
    print subprocess.check_output('ls -l', shell=True)
    print subprocess.check_output('hostname')

def os_useful_command_snippets():
    """
    
    .. code:: python
    
        os.path.dirname(os.path.realpath(__file__))
        os.getcwd()
        os.path.abspath('.')
        os.walk()
        os.listdir("some directory")
        os.system('ls -l')
        os.path.exists('junk.txt')
        os.path.isfile('junk.txt')
        os.path.isdir('junk.txt')
        os.path.basename(a)
        os.path.dirname(a)
        os.path.split() # returns both basename and dirname
        os.path.splitext(os.path.basename(a)) # splits filename and file-extension
        
        In [80]: os.path.splitext(os.path.basename(a))
        Out[80]: ('junk', '.txt')
        
        os.mkdir('junkdir')
        os.rename('junkdir', 'foodir')
        In [37]: 'junkdir' in os.listdir(os.curdir)
        Out[37]: False
        In [38]: 'foodir' in os.listdir(os.curdir)
        Out[38]: True
        
        os.rmdir('foodir')
        
        
        
        # delete a file
        In [44]: fp = open('junk.txt', 'w')
        
        In [45]: fp.close()
        
        In [46]: 'junk.txt' in os.listdir(os.curdir)
        Out[46]: True
        
        In [47]: os.remove('junk.txt')
        
        In [48]: 'junk.txt' in os.listdir(os.curdir)
        Out[48]: False
    """
    pass
#%% === sns ===



#%% === neuroimage specifics ===

def ni_data_io():
    from nibabel import load
    from nilearn import plotting

    impath = ('~/nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal'+
              '/Pitt_0050003_func_preproc.nii.gz')
    impath = os.path.expanduser(impath)
    anat_img = load(impath)

    # access imaging data and affine
    anat_data = anat_img.get_data()
    print('anat_data has shape: %s' % str(anat_data.shape))
    anat_affine = anat_img.get_affine()
    print('anat_affine:\n%s' % anat_affine)

    plotting.plot_anat(anat_img, cut_coords=(1,1,1))

def ni_plot_volume_from_path():
    #3d image display by directory
    from nilearn import plotting
    fdir = ('/home/takanori/anaconda/lib/python2.7/site-packages/'+
            'nilearn-0.1.5.dev0-py2.7.egg/nilearn/datasets/data')
    fname = 'avg152T1_brain.nii.gz'
    #fdir = '/home/takanori/nilearn_data/msdl_atlas/MSDL_rois'
    #fname = "msdl_rois.nii"
    fpath = os.path.join(fdir,fname)

    # plot glass brain
    plotting.plot_glass_brain(fpath)

    # plot anatomical slice
    plotting.plot_anat(fpath, cut_coords=(1,1,1))

    data=load(fpath).get_data()

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.io import loadmat

import os
#import sys

import tak as tw
from .util_data import get_data_path, fix_file_list
#%%
def filepath_tob():
    return os.path.join(get_data_path(),'tob')

def filepath_tob_conn():
    """ Get connmat filepath

    Created 05/23/2016
    """
    filename = 'tob_connectome_PROB_0509_2016.mat'
    return os.path.join(filepath_tob(),filename)

#%% === main routines ===
def get_tob_0523(apply_949filter=True):
    """ Get tobacco data w/o filtering subjects based on disease label
    
    Usage
    -----
    >>> import tak as tw
    >>> X, fileList, subj_id, df = tw.data.tob.get_tob_0523()
    """
    connpath = filepath_tob_conn()
    connMat = loadmat(connpath,variable_names='connMatrix')['connMatrix']
    fileList = loadmat(connpath,variable_names='fileList')['fileList']

    fileList = fix_file_list(fileList)

    assert sorted(fileList) == fileList, ' File list not sorted!'
    #return connMat, fileList

    def get_subjid(fileList):
        """First 11 char represents bblid"""
        subjid_list = []
        for i,fname in enumerate(fileList):
            subjid = fname[:11]
            #bblid = int(bblid)
            subjid_list.append(subjid)
        return subjid_list

    subj_id = get_subjid(fileList)
    #return connMat, fileList, subjid

    #=== get spreadsheet info ===#
    def get_scores_tob():
        filepath = os.path.join(get_data_path(),'tob','Tobacco_mastersheet_20160511.xlsx')
        df = pd.read_excel(filepath)
        var_list = ['SubjID','gender','dx_current','dx_lifetime']

        df1 = df[var_list]
        df2 = df.ix[:,-92:]

        df = pd.concat((df1,df2),axis=1)
        return df

    df = get_scores_tob()
    _subj_list = df['SubjID'].tolist()
    assert sorted(_subj_list) == _subj_list, ' SubjID not sorted!'


    #-- filter out subject not on disk --#
    # take the intersection between SubjID on disk and SubjID on spreadsheet
    subjects = list(set(subj_id).intersection(set( df['SubjID'])))

    # apply filter on DF
    mask = df['SubjID'].isin(subjects).values
    df = df.ix[mask,:].reset_index(drop=True)

    # apply filter on connmat on disk (ie, data on disk but no clinical info available)
    subj_mask = np.array([i for i,x in enumerate(subj_id) if x in subjects])
    connMat = connMat[:,:,subj_mask]

    # using list comprehension since ``fileList`` is not an ndarray here
    fileList = [fileList[i] for i in subj_mask]
    subj_id = [subj_id[i] for i in subj_mask]

    # reshape (nROI x nROI x nsubj) array into (nsubj x nEdges)
    X = tw.conn2design(connMat)

    return X, fileList, subj_id, df

def util_tob():
    """ Get tobacco data, selecti those with known disease labels
    
    Note: created 05/23. Old version....kept for backward compatibility
    
    Usage
    -----
    >>> import tak as tw
    >>> X,y,df = tw.data.tob.util_tob()
    """
    X, fileList, subj_id, df = get_tob_0523()

    df_aut = df.query('dx_lifetime=="AUT (Autism)" or dx_lifetime == "Aut"')
    df_tdc = df.query('dx_lifetime=="TDC"')

    idx_aut = df_aut.index
    idx_tdc = df_tdc.index

    X_aut = X[idx_aut]
    X_tdc = X[idx_tdc]

    X = np.vstack((X_aut,X_tdc))
    y = np.concatenate((
         np.ones(X_aut.shape[0]),
        -np.ones(X_tdc.shape[0])
        ))


    df_new = pd.concat([df_aut,df_tdc],axis=0)
    idx_orig = df_new.index.tolist()
    df_new = df_new.reset_index(drop=True)


    #%% change AUT to autism (added 06/01/2016)
    #http://www.regular-expressions.info/modifiers.html
    tmp = [u'AUT' if str_.upper().startswith('AUT') else str_ for str_ in df_new['dx_lifetime']]
    df_new['dx_lifetime']=tmp

    return X,y,df_new#,idx_orig



def util_tob_0610():
    """ Get tobacco data, selecti those with known disease labels
    
    Usage
    -----
    >>> import tak as tw
    >>> X,y,df = tw.data.tob.util_tob()
    """
    X, fileList, subj_id, df = get_tob_0523()

    #%% Replace all strings starting with 'AUT' (case insensitive) with AUT
    df[['dx_current','dx_lifetime']] = \
        df[['dx_current','dx_lifetime']].replace(r'(?i)aut.*', u'AUT',regex=True)
    #%%
    df_aut = df.query('dx_lifetime=="AUT"')
    df_tdc = df.query('dx_lifetime=="TDC"')

    idx_aut = df_aut.index
    idx_tdc = df_tdc.index

    X_aut = X[idx_aut]
    X_tdc = X[idx_tdc]

    X = np.vstack((X_aut,X_tdc))
    y = np.concatenate((
         np.ones(X_aut.shape[0]),
        -np.ones(X_tdc.shape[0])
        ))

    df_new = pd.concat([df_aut,df_tdc],axis=0)
    #idx_orig = df_new.index.tolist()
    df_new = df_new.reset_index(drop=True)
    return X,y.astype(int),df_new



#%%

# -*- coding: utf-8 -*-
"""
===============================================================================
Get pnc data io

Result
-------------------------------------------------------------------------------
Next up?
===============================================================================
Created on Thu May 12 12:09:45 2016

@author: takanori
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

import os
#import sys

import tak as tw
from .util_data import get_data_path, fix_file_list
#%% === path info ===
def filepath_pnc():
    return os.path.join(get_data_path(),'pnc')

def filepath_pnc_conn():
    """ Get connmat filepath

    Created 05/12/2016
    """
    filename = 'PNC_connectome_PROB_0509_2016.mat'
    return os.path.join(filepath_pnc(),filename)

def filepath_pnc_scores_old():
    """ PNC scores with the ``old`` data i/o

    (only kept this to verify consistency in scores with the updated
    spreadsheets)
    """
    filename = 'GO_Connectivity_sample_Berkan.xlsx'
    return os.path.join(filepath_pnc(),filename)


def filepath_csv_0510():
    """
    """
    csv_path = os.path.join(filepath_pnc(),'51_PNC_Demographics_0510_2016')
    return csv_path
    
def get_pnc_tob_matched_subjects_0608():
    """ Get list of *matched* PNC subjects (matched with Tobacco)
    
    
    Usage
    -----
    >>> df_pnc, df_tob = get_pnc_tob_matched_subjects_0608()
    """
    #%%
    xls_path = os.path.join(filepath_pnc(),'PNC_Tobacco_Merged_Dataset.xlsx')
    df = pd.read_excel(xls_path).iloc[:,:8] # only the first 8 columns are relevant
    
    df_tob = df[:111]
    df_pnc = df[111:]
    
    #_,_,df_ = get_pnc_0523()
    #%%
    return df_pnc, df_tob
#%% === main routines ===
def get_949_subjects():
    """ Get list of 949 subjects for our analyssi

    05/23/2016
    """
    filepath = os.path.join(filepath_csv_0510(),'list949')
    with open(filepath,'r') as file:
        subjlist = file.read()

    # convert huge string with "\n" into a list via ``splitlines``
    subjlist = sorted(subjlist.splitlines())

    def get_bblid(fileList):
        """First 6 char represents bblid"""
        bblid_list = []
        for i,fname in enumerate(fileList):
            bblid = fname[:6]
            bblid = int(bblid) # string to int
            bblid_list.append(bblid)
        return bblid_list

    # return just the bblid
    return get_bblid(subjlist)


def get_pnc_0523(apply_949filter=True):
    """
    """
    connpath = filepath_pnc_conn()
    connMat = loadmat(connpath,variable_names='connMatrix')['connMatrix']
    fileList = loadmat(connpath,variable_names='fileList')['fileList']

    fileList = fix_file_list(fileList)

    assert sorted(fileList) == fileList, ' File list not sorted!'

    def get_bblid(fileList):
        """First 6 char represents bblid"""
        bblid_list = []
        for i,fname in enumerate(fileList):
            bblid = fname[:6]
            bblid = int(bblid)
            bblid_list.append(bblid)
        return bblid_list

    bblid = get_bblid(fileList)

    subj_949 = get_949_subjects()
    # === select only the subjects in ``subjlist`` ===
    bblid_mask = np.array([i for i,x in enumerate(bblid) if x in subj_949])

    connMat  = connMat[:,:,bblid_mask]

    # using list comprehension since ``fileList`` is not an ndarray here
    fileList = [fileList[i] for i in bblid_mask]

    bblid = [bblid[i] for i in bblid_mask]

    #=== get basic demographc info ===#
    def get_demographics():
        filepath = os.path.join(filepath_csv_0510(),'51_PNC_Demographics.csv')
        df = pd.read_csv(filepath,na_values='.')

        #-- filter out subject not in 949 list --#
        mask = df['bblid'].isin(bblid).values
        df = df.ix[mask,:].reset_index(drop=True)

        # check if dataframe is sorted by bblid
        _sort_test = all(np.argsort(df['bblid'].values) == np.arange( df.shape[0]))
        assert _sort_test, ' bblid not sorted!'


        del df['siteid'] # <- this col not interesting
        df['age_at_go1_scan'] = df['age_at_go1_scan']/12 # convert age to years

        # in the original spreadsheet, y=2 is female, y=1 is male
        y=df['sex'].values
        y[y==2]=-1

        return df, y

    df,y = get_demographics()

    # reshape from (nROI,nROI,nsubj) to (nsubj,nEdges)
    X = tw.conn2design(connMat)
    return X, y, df#, fileList, bblid


def get_cnb_normed():
    filepath = os.path.join(filepath_csv_0510(),'61_PNC_CNB_Normed_Data_by_Form_and_Age.csv')
    df = pd.read_csv(filepath)

    # check if df is sorted by bblid
    _sort_test = all(np.argsort(df['bblid'].values) == np.arange( df.shape[0]))
    assert _sort_test, ' bblid not sorted!'

    #-- filter out subject not in 949 list --#
    mask = df['bblid'].isin(get_949_subjects()).values
    df = df.ix[mask,:].reset_index(drop=True)
    return df

def get_cnb_factor():
    filepath = os.path.join(filepath_csv_0510(),'57_PNC_CNB_Factor_Scores.csv')
    df = pd.read_csv(filepath)

    # check if df is sorted by bblid
    _sort_test = all(np.argsort(df['bblid'].values) == np.arange( df.shape[0]))
    assert _sort_test, ' bblid not sorted!'

    #-- filter out subject not in 949 list --#
    mask = df['bblid'].isin(get_949_subjects()).values
    df = df.ix[mask,:].reset_index(drop=True)
    return df
"""
===============================================================================
Data I/O and other relevant stuffs
===============================================================================
Created on Mon Oct 26 14:46:36 2015

@author: takanori
"""
#%% load relevant modules
#import time

import numpy as np
from scipy.io import loadmat as _loadmat

import os
#import sys as _sys
import warnings as _warnings
from sklearn.utils import deprecated as _deprecated

#%%*** get hostname ***
from socket import gethostname as _gethostname
hostname = _gethostname()
#%% load my main "tak" module
import tak as tw
reload(tw)
import pandas as pd
#%%********** K, let's dig into real stuffs below **************
def get_data_path():
    """Get root data-path (created 01/23/2016)

    Helpful for unifying codes across computer platforms.
    """
    if hostname == 'sbia-pc125':
        data_path = '/home/takanori/data'
    elif hostname in ['takanori-PC','tak-sp3']:
        data_path = "C:\\Users\\takanori\\Dropbox\\work\\data"
    else:
        # on the computer cluster
        data_path = '/cbica/home/watanabt/data'
    return data_path
#%%==== General stuffs ====
@_deprecated('Use tak.tak.record_str_to_list')
def ndarray_string_to_list(file_list_in):
    """
    # file_list comes as ndarray...kinda tricky/awkward to handle...convert to list
    """
    file_list = []
    # i can never memorize the awful code below, so made this function
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    return file_list

@_deprecated("Use pd_get_column_info from tak.tak module (11/05/2015)")
def get_df_column_info(df):
    """ Get column info of a DataFrame...as a DataFrame!

    Created 10/29/20125

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

#%%==== nifti files on disk ====
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

def get_nii_eve176_matfile():
    """ Load in .mat file of eve176 (resolution 98x116x80)

    Output
    ------
    eve176_mat : ndarray of shape [98,116,80]
        Array value integer value from [0,..,176]

    Usage
    ------
    >>> eve176_mat = get_nii_eve176_matfile()

    Warning (10/30/2015)
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
    ``proto_nii_vol_flip_fix.py``
    """
    fullpath='/home/takanori/work-local/tak-ace-ibis/EVE176_atlas.mat'
    DATA = _loadmat(fullpath, variable_names=['eve176'])

    return DATA['eve176']


def get_nii_eve176(data_only=True):
    """ Get nifti volume for eve176 volume (98 x 116 x 80)

    **11/05/2015** Data i/o updated after migrating to new git repository.


    Parameter
    --------
    data_only : bool (default=True)
        if True, return data as ndarray (ie, not the full Nifti object from nibabel

    Usage
    ------
    >>> eve176 = get_nii_eve176()
    """
    from nibabel import load
    fpath='/home/takanori/data/misc/eve176'
    fname='EVE_1mm_to_DTIAtlas_v06v12v24_ROIs.nii.gz'
    fullpath = os.path.join(fpath,fname)
    img = load(fullpath)

    if data_only:
        return img.get_data() # return ndarray
    else:
        return img

def get_nii_ibis_mean_volume(modality='FA', data_only=False):
    """
    Parameter
    --------
    modality : string (default='FA')
        string specifying valid modality ('FA', 'TR', 'RD', 'AX')
    data_only : bool (default=False)
        if True, return data as ndarray (ie, not the full Nifti object from nibabel

    Usage
    ------
    >>> eve176 = get_nii_eve176()
    """
    modality = modality.upper() # for case insensitivity
    filepath = '/home/takanori/work-local/tak-ace-ibis'
    if modality == 'FA':
        filename = 'mean_volume_FA_0807.nii.gz'
    elif modality == 'TR':
        filename = 'mean_volume_FA_0807.nii.gz'
    elif modality == 'RD':
        filename = 'mean_volume_FA_0807.nii.gz'
    elif modality == 'AX':
        filename = 'mean_volume_FA_0807.nii.gz'
    else:
        print "modality = {} specified not recognized.  Exit program.".format(modality)
        return

    from nibabel import load
    fullpath = os.path.join(filepath,filename)
    img = load(fullpath)

    if data_only:
        return img.get_data() # return ndarray
    else:
        return img

#%%==== filepath functions ====#
""" 11/11/2015: decided to create these since it'd be easier for me to keep
track of what part of my code to modify when I move data around my computer"""
#%%--- IBIS ---
def filepath_ibis_conn():
    """ Last update: 2016-01-23"""
    filepath = os.path.join(get_data_path(),'ibis')
    filename = 'IBIS_connectome_PROB_1105_2015.mat'
    return os.path.join(filepath,filename)


def filepath_ibis_scores_basic():
    """ Last update: 2016-01-23"""
    filepath = os.path.join(get_data_path(),'ibis')
    #filename = 'IBIS_subjs_at_cbica.xlsx'
    #^^^^^^^^ reading excel file is slow....i've converted the sheets into csv files
    #         (see tw_IBIS_clinical_scores_to_csv.py)
    filename = 'IBIS_subjs_at_cbica_basic.csv'
    return os.path.join(filepath,filename)


def filepath_ibis_scores_master():
    """ Last update: 2016-01-23"""
    filepath = os.path.join(get_data_path(),'ibis')
    filename = 'IBIS_subjs_at_cbica_all.csv'
    return os.path.join(filepath,filename)


def filepath_ibis_176rois():
    """ Last update: 2016-01-23"""
    filepath = os.path.join(get_data_path(),'ibis')
    filename = 'IBIS_ROIStats_All.csv'
    return os.path.join(filepath,filename)
#%%--- PNC ---
def filepath_pnc_conn():
    filepath = os.path.join(get_data_path(),'pnc')
    filename = 'PNC_connectome_PROB_1105_2015.mat'
    return os.path.join(filepath,filename)

def filepath_pnc_scores():
    filepath = os.path.join(get_data_path(),'pnc')
    filename = 'GO_Connectivity_sample_Berkan.xlsx'
    return os.path.join(filepath,filename)
#%%--- TBI ---
def filepath_tbi_conn():
    """Get full path to the tbi connectome .mat file

    Last updated (2016-01-23)
    """
#    if hostname == 'sbia-pc125':
#        filepath = '/home/takanori/data/tbi/'
#    elif hostname in ['takanori-PC','tak-sp3']:
#        filepath = "C:\\Users\\takanori\\Dropbox\\work\\data\\tbi"
#    else:
#        # on the computer cluster
#        filepath = '/cbica/home/watanabt/data/tbi/'
    filepath = os.path.join(get_data_path(),'tbi')
    #filename = 'tbi_connectome_PROB_1027_2015.mat'
    filename = 'tbi_connectome_PROB_1207_2015.mat'
    return os.path.join(filepath,filename)

def filepath_tbi_scores():
    """Get full path to the tbi-score spreadsheet

    Last updated (2016-01-23)
    """
    filepath = os.path.join(get_data_path(),'tbi')
    filename = 'TBI_Data_Consolidated_v2.xlsx'
    return os.path.join(filepath,filename)
#%%--- Tobacco ----
def filepath_tob_hardi():
    """ Get full path to hardi spreadsheet

    Last update: 2016-01-23"""
    filepath = os.path.join(get_data_path(),'tob')
    filename = 'Hardi_invariants_roi_stats_for_Tak.csv'
    return os.path.join(filepath,filename)


def filepath_tob_conn():
    """ Get full path to connectivity matfile

    Last update: 2016-01-23

    Last update: 2016-05-09 - data points to the new 100 streamlines
    """
    filepath = os.path.join(get_data_path(),'tob')
    #filename = 'tob_connectome_PROB_1105_2015.mat'
    filename = 'tob_connectome_PROB_0509_2016.mat'
    return os.path.join(filepath,filename)


def filepath_tob_scores():
    """ Get full path to clinicasl scores sheet

    Last update: 2016-01-23"""
    filepath = os.path.join(get_data_path(),'tob')
    filename = 'Tobacco_DTI_Connectivity_sample_20151023.csv'
    return os.path.join(filepath,filename)
#%%---- misc ----
def filepath_176roi_labels():
    if hostname == 'sbia-pc125':
        filepath = '/home/takanori/data/misc/eve176'
    else:
        # on the computer cluster
        filepath = '/cbica/home/watanabt/data/misc/eve176'
    filename = 'JHU_labels_1.csv'
    return os.path.join(filepath,filename)
#%%==== IBIS stuffs ====
def get_ibis_connectomes(return_as_design=True, return_dropped=False,
                         return_all_scores=False):
    """ Get the connectome data for IBIS

    **11/05/2015** Data i/o updated after migrating to new git repository.

    Usage (all cases)
    ------
    The default (the most typical usecase for me)

    >>> X, df_basic = twio.get_ibis_connectomes()

    Get dropped rows (added this for potential sanity checks I may want to do in the future)

    >>> X, df_basic, df_dropped = twio.get_ibis_connectomes(return_dropped=True)

    Get all clinical scores

    >>> X, df_basic, df_master = twio.get_ibis_connectomes(return_all_scores=True)

    Get all clinical scores + dropped rows

    >>> X, df_basic, df_master, df_basic_dropped = twio.get_ibis_connectomes(return_all_scores=True,return_dropped=True)

    Handy snippets for cross-sectoinal analysis
    --------------
    Extract DataFrame by Sessions

    >>> # note that we are dropping columns with all NANS
    >>> df_v06 = df_master.groupby('Session').get_group('V06').dropna(axis=1,how='all')
    >>> df_v12 = df_master.groupby('Session').get_group('V12').dropna(axis=1,how='all')
    >>> df_v24 = df_master.groupby('Session').get_group('V24').dropna(axis=1,how='all')
    >>>  # corresponding connectome data can be accessed by the above DataFrames' "index" attribute
    >>> X06 = X[df_v06.index]
    >>> X12 = X[df_v12.index]
    >>> X24 = X[df_v24.index]

    Equivalent method via ``query`` (I find this more intuitive)

    >>> df_v06 = df_master.query('Session == "V06"').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)
    >>> df_v12 = df_master.query('Session == "V12"').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)
    >>> df_v24 = df_master.query('Session == "V24"').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)

    Get summary of column infos as dict

    >>> col_info = {'v06':twio.get_df_column_info(df_v06),
    ...             'v12':twio.get_df_column_info(df_v12),
    ...             'v24':twio.get_df_column_info(df_v24)}

    Pandas Snippets
    ---------
    For refreshers, go to https://github.com/takwatanabe2004/snippet_book/blob/master/python/snippet-python.rst#pandas-snippets-condensed

    The ``groupby`` and ``get_group`` method is really handy for selecting subset of DataFrames

    >>> df_basic.groupby('Gender').get_group('Male')
    >>> # this index can be used to extract corresponding subjects
    >>> df_basic.groupby('Gender').get_group('Male').index

    More ``groupby`` stuffs

    >>> df_basic.groupby('Session').mean()
    >>> df_basic.groupby('Session').describe()

    Parameters
    -----------
    return_as_design : bool (default: True)
        How to return the connectomes

        - If True, return the connectomes as an (n x p) 2d array (p = # edges)
        - If False, return connectomes as (nROI x nROI x n) 3d array

    return_dropped : bool (default: False)
        decides whether to return a DataFrame of the dropped indices as output
    return_all_scores : bool (default: False)
        Output DataFrame of all clinical scores

    Remarks
    -------
    With the IBIS data, we don't really have a definite "y" label information in
    the sense that there are so many possible scores to use as labels, such as:

    - HR+ vs HR-
    - ASD vs Healthy
    - Male vs Female
    - Male (LR) vs Female (LR)
    - etc..

    Thus unlike the other ``get_<project>_connectomes`` script, here I won't
    return any "label" vector y

    However, similar to ``get_tbi_connectomes``, I'll return the clinical scores
    """
    #%%===== first load in the connectome matrix ========#
    var_names = ('connMatrix', 'fileList')

    DATA = _loadmat(filepath_ibis_conn(),variable_names=var_names)

    if return_as_design:
        X = tw.conn2design(DATA['connMatrix'])
    else:
        X = DATA['connMatrix']

    file_list_in = DATA['fileList']

    # file_list comes as ndarray...kinda tricky/awkward to handle...convert to list
    file_list = []
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    # grab the subject id from the filename substring (upper just in case...)
    file_id = [xx[:10].upper() for xx in file_list]

    # create a DataFrame of "ID" and "file_name" (to be used for "join" later)
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original')
    #%% === read in spreadsheet of "basic" clinical scores ===
    df_basic = _ibis_helper_get_basic_scores()
    #%% === "join" df_file_list with df_basic['ID'] ===
    """
    ===========================================================================
    DROP CRITERIA
    - ID = NAN after outer-join: 2 possible reasons for this:
        (1) file-scan on disk was not on the spreadsheet
        (2) the ID on spreadsheet does not have connectome data no disk

    - after some thoughts, decided to make above the ONLY drop criteria...
    - there are other "drop" i may consider, such as NAN in DX, or GROUP
      column...but these may still have "Gender" and "Risk" label available...
    # TODO: include snippet in the docstring of how to drop rows with NANS,
      depending on the type of analysis i'm interested in
    ===========================================================================
    """
    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_basic = pd.merge(df_file_id,df_basic,how='outer',on='ID')

    # if "file_name" value is nan, then connectome data not available....so drop these rows
    idx_to_drop = df_basic['file_name'].isnull()

    df_basic_dropped = df_basic.ix[idx_to_drop,:]
    df_basic = df_basic.ix[~idx_to_drop,:]

    # ensure no redundant data exists (here check #rows remain unchanged after "drop_duplicates" method
    assert (df_basic.drop_duplicates(['ID']).shape[0] == df_basic.shape[0]),  'oopsie'

    """ Silly sanity check
    ===========================================================================
    Turns out after the above "dropping", the order of the X rows (subjects)
    appears to not have changed.
    Verify this by asserting the DataFrame "ID" and the original file_list
    string agrees with each other
    ===========================================================================
    # argh, ``assert`` is a statement, not a function. I've been using it like
    # a function, in which case Warning message of "assertion is always true"
    # always kepts popping out
    """
    for id1,id2 in zip(file_id, df_basic['ID'].values.tolist()):
        #assert(id1==id2, 'oh noes!')
        #^^^^^ wrong syntax
        assert (id1==id2), 'oh noes!'

    checker = pd.DataFrame(dict(id1=file_id, id2 = df_basic['ID'].values.tolist()))
    checker['check'] = checker['id1']==checker['id2']
    assert np.all(checker['check']), 'oh no'
    #%% == Create a column with a key indicating session availability ==
    # here i'm using an helper function i created
    df_basic = _ibis_helper_add_key(df_basic)

    #%% reorder columns to my taste
    cols = df_basic.columns.tolist()

    # columns I'd like to show up front
    cols_front = ['ID', 'key', 'Session', 'Subject','Group','DX','Risk',
                  'Gender','Age','file_name']

    # remove the list items from above list using pop.
    for colname in cols_front:
        idx = cols.index(colname)
        cols.pop(idx)

    # append the remaining columns at the tails of the list
    cols = cols_front + cols

    # apply reordering
    df_basic = df_basic[cols]
    #%%******* 1st BLOCK DONE ******
    """========================================================================
       OK BLOCK1 of code is done....if "basic" scores is our only interest,
       return here
       ========================================================================
    """
    if not return_all_scores:
        if return_dropped:
            return X, df_basic, df_basic_dropped
        else:
            return X, df_basic # <- the default output
    #%% ==== read-in ALL scores=====
    #%% now read in sheet with all clinical scores
    df_master = _ibis_helper_get_master_scores(df_basic)

    # ensure indices/row-order remains unchanged
    assert np.array_equal(df_master['ID'].values, df_basic['ID']), 'Assertion fail'

    if return_dropped:
        return X, df_basic, df_master, df_basic_dropped
    else:
        return X, df_basic, df_master


def get_ibis_full_volumes(modality='FA', session='V24', return_dropped=False,return_all_scores=False):
    """ Get design matrix for IBIS diffusion volume at full resolution of 98 x 116 x 80

    **(Created 12/03/2015)**

    Important remark (difference between downsampled version)
    ----------------
    - Here we must specify ``session`` as parameter.  I made this choice since
      the filesize got too bulky for saving the entire data in a single file.
    - I dropped the ``key`` idea here, as things got tricky due to all files
      not being loaded

    Remarks about row/col major indexing
    ------------------------------------
    - features from the brain volumes now extracted using **row-major** indexing,
      which follows python/C convention.
    - Old version extracted features from brain-mask using Matlab, which
      follows **column-major** indexing; book-keeping this difference in Python
      was a headache and was just asking for a bug to creep in...
    - thus I decided to extract features from brain-mask using Python, and
      saving the result as an ``*.npz`` file
    - see ``save_ibis_fullvolume_rowmajor_order_1203.py``

    Usage
    ------
    >>> X, df_basic = twio.get_ibis_downsampled_volumes('FA', 'V24')
    >>> X, df_basic, df_dropped = twio.get_ibis_downsampled_volumes('AX', 'V06',return_dropped=True)
    >>> X, df_basic, df_master, df_dropped = twio.get_ibis_downsampled_volumes('RD', 'V12',return_dropped=True, return_all_scores=True)

    Parameters
    ----------
    modality : string (default='FA')
        string specifying valid modality (``'FA', 'TR', 'RD', 'AX'``)
    session : string (default='V24')
        string specifying session (``'V06', 'V12', 'V24'``)
    return_dropped : bool (default: False)
        decides whether to return a DataFrame of the dropped indices as output
    return_all_scores : bool (default: False)
        Output DataFrame of all clinical scores


    Remark on design matrix
    -----------------
    - The #rows in the design matrix ``X`` is smaller than what is stored in the
      ``.mat`` file (goes from 978 to 968)
    - This is because for 10 subjects, the ``gender`` and ``dx`` entry had NAN values
    - These are the 10 subjects I am speaking of:

    ::

        210765_V06
        268987_V06
        512998_V06
        593785_V06
        593785_V12
        649639_V06
        649639_V12
        753513_V06
        793001_V06
        793001_V12

    - Set ``return_dropped = True`` to see what's going on
    """
    #%%
    if hostname == 'sbia-pc125':
        filepath = '/home/takanori/data/ibis/'
    else:
        # on the computer cluster
        filepath = '/cbica/home/watanabt/data/ibis/'
    #%%
    #=== select modality to read ===#
    modality = modality.upper() # for case insensitivity
    filename = 'IBIS_fullvolume_rowmajor_'+modality+'_'+session+'_1203_2015.npz'

    DATA = np.load(os.path.join(filepath,filename))
    X = DATA['design_matrix']

    # convert numpy array of string to list
    file_list = [fname_ for fname_ in DATA['fileList']]

    # grab the subject id from the filename substring (upper just in case...)
    file_id = [xx[:10].upper() for xx in file_list]

    # create a DataFrame of "ID" and "file_name" (to be used for "join" later)
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original')
    #%% === read in spreadsheet of "basic" clinical scores ===
    df_basic = _ibis_helper_get_basic_scores()
#    return X,df_basic,df_file_id, file_id
    #%% === "join" df_file_list with df_basic['ID'] ===
    """
    ===========================================================================
    DROP CRITERIA
    - ID = NAN after outer-join: 2 possible reasons for this:
        (1) file-scan on disk was not on the spreadsheet
        (2) the subject's "Gender" or "Risk" is missing (turns out these coincide
            anyways in the original spreadsheet, but just for clarify in the code)
    ===========================================================================
    """
    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_basic = pd.merge(df_file_id,df_basic,how='outer',on='ID')

    # get indices to drop using above drop criteria
    idx_to_drop1 = df_basic['file_name'].isnull()
    idx_to_drop2 = df_basic['Risk'].isnull()
    idx_to_drop3 = df_basic['Gender'].isnull()

    idx_to_drop = (idx_to_drop1 | idx_to_drop2 | idx_to_drop3)

    df_basic_dropped = df_basic.ix[idx_to_drop,:]
    df_basic = df_basic.ix[~idx_to_drop,:]

    # ensure no redundant data exists (here check #rows remain unchanged after "drop_duplicates" method
    assert (df_basic.drop_duplicates(['ID']).shape[0] == df_basic.shape[0]),  'oopsie'
    #%% some scans on disk has NAN for 'Gender' and 'Risk'...drop them
    X = X[df_basic['index_data_original']]
    #%% reorder columns to my taste
    cols = df_basic.columns.tolist()

    # columns I'd like to show up front
    cols_front = ['ID', 'Session', 'Subject','Group','DX','Risk',
                  'Gender','Age','file_name']

    # remove the list items from above list using pop.
    for colname in cols_front:
        idx = cols.index(colname)
        cols.pop(idx)

    # append the remaining columns at the tails of the list
    cols = cols_front + cols

    # apply reordering
    df_basic = df_basic[cols]
    #%%******* 1st BLOCK DONE ******
    """========================================================================
       OK BLOCK1 of code is done....if "basic" scores is our only interest,
       return here
       ========================================================================
    """
    if not return_all_scores:
        if return_dropped:
            return X, df_basic, df_basic_dropped
        else:
            return X, df_basic # <- the default output
    #%% ==== read-in ALL scores=====
    #%% now read in sheet with all clinical scores
    df_master = _ibis_helper_get_master_scores(df_basic)

    # ensure indices/row-order remains unchanged
    assert np.array_equal(df_master['ID'].values, df_basic['ID']), 'Assertion fail'

    if return_dropped:
        return X, df_basic, df_master, df_basic_dropped
    else:
        return X, df_basic, df_master

def get_ibis_downsampled_volumes(modality='FA', return_dropped=False,return_all_scores=False):
    """ Get design matrix for IBIS diffusion volume, downsampled by factor of 2.

    Update 11/09/2015
    ------------------
    - features from the brain volumes now extracted using **row-major** indexing,
      which follows python/C convention.
    - Old version extracted features from brain-mask using Matlab, which
      follows **column-major** indexing; book-keeping this difference in Python
      was a headache and was just asking for a bug to creep in...
    - thus I decided to extract features from brain-mask using Python, and
      saving the result as an ``*.npz`` file
    - seee ``save_ibis_downsampled_volume_rowmajor_order_1109.py``

    - Original resolution:    98 x 116 x 80
    - Downsampled resolution: 49 x  58 x 40

    Usage
    ------
    >>> X, df_basic = twio.get_ibis_downsampled_volumes()
    >>> X, df_basic = twio.get_ibis_downsampled_volumes('ALL')
    >>> X, df_basic = twio.get_ibis_downsampled_volumes('TR')
    >>> X, df_basic, df_dropped = twio.get_ibis_downsampled_volumes('AX', return_dropped=True)
    >>> X, df_basic, df_master, df_dropped = twio.get_ibis_downsampled_volumes('RD', return_dropped=True, return_all_scores=True)

    Parameters
    ----------
    modality : string (default='FA')
        string specifying valid modality (``'FA', 'TR', 'RD', 'AX','ALL'``)
        ('ALL' will create a huge-ass design matrix with the columns
        from the 4 modalities concatenated together)
    return_dropped : bool (default: False)
        decides whether to return a DataFrame of the dropped indices as output
    return_all_scores : bool (default: False)
        Output DataFrame of all clinical scores

    Handy snippets for cross-sectoinal analysis
    --------------
    Extract DataFrame by Sessions

    >>> df_v06 = df_master.query('Session == "V06"').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)
    >>> df_v12 = df_master.query('Session == "V12"').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)
    >>> df_v24 = df_master.query('Session == "V24"').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)
    >>>  # corresponding diffusion data can be accessed by the above DataFrames' "index" attribute
    >>> X06 = X[df_v06.index]
    >>> X12 = X[df_v12.index]
    >>> X24 = X[df_v24.index]

    Remark on design matrix
    -----------------
    - The #rows in the design matrix ``X`` is smaller than what is stored in the
      ``.mat`` file (goes from 978 to 968)
    - This is because for 10 subjects, the ``gender`` and ``dx`` entry had NAN values
    - These are the 10 subjects I am speaking of:

    ::

        210765_V06
        268987_V06
        512998_V06
        593785_V06
        593785_V12
        649639_V06
        649639_V12
        753513_V06
        793001_V06
        793001_V12

    - Set ``return_dropped = True`` to see what's going on
    """
    if hostname == 'sbia-pc125':
        filepath = '/home/takanori/data/ibis/'
    else:
        # on the computer cluster
        filepath = '/cbica/home/watanabt/data/ibis/'

    #=== select modality to read ===#
    modality = modality.upper() # for case insensitivity
    if modality in ['FA','TR','RD','AX']:
        filename='IBIS_downsampled_volume_rowmajor_'+modality+'_1108_2015.npz'
    elif modality == 'ALL':
        filename_list=['IBIS_downsampled_volume_rowmajor_'+mod+'_1108_2015.npz'
                       for mod in ['FA','TR','RD','AX']]
    else:
        print "modality = {} not recognized.  Exit program.".format(modality)
        return

    if modality == 'ALL':
        X = []
        for filename in filename_list:
            DATA = np.load(os.path.join(filepath,filename))
            X.append(DATA['design_matrix'])
        # convert columns into ndarray
        X = np.hstack(X)
    else:
        DATA = np.load(os.path.join(filepath,filename))
        X = DATA['design_matrix']

    file_list = tw.record_str_to_list(DATA['fileList'])

    # grab the subject id from the filename substring (upper just in case...)
    file_id = [xx[:10].upper() for xx in file_list]

    # create a DataFrame of "ID" and "file_name" (to be used for "join" later)
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original')
    #%% === read in spreadsheet of "basic" clinical scores ===
    df_basic = _ibis_helper_get_basic_scores()
    #%% === "join" df_file_list with df_basic['ID'] ===
    """
    ===========================================================================
    DROP CRITERIA
    - ID = NAN after outer-join: 2 possible reasons for this:
        (1) file-scan on disk was not on the spreadsheet
        (2) the subject's "Gender" or "Risk" is missing (turns out these coincide
            anyways in the original spreadsheet, but just for clarify in the code)
    ===========================================================================
    """
    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_basic = pd.merge(df_file_id,df_basic,how='outer',on='ID')

    # get indices to drop using above drop criteria
    idx_to_drop1 = df_basic['file_name'].isnull()
    idx_to_drop2 = df_basic['Risk'].isnull()
    idx_to_drop3 = df_basic['Gender'].isnull()

    idx_to_drop = (idx_to_drop1 | idx_to_drop2 | idx_to_drop3)

    df_basic_dropped = df_basic.ix[idx_to_drop,:]
    df_basic = df_basic.ix[~idx_to_drop,:]

    # ensure no redundant data exists (here check #rows remain unchanged after "drop_duplicates" method
    assert (df_basic.drop_duplicates(['ID']).shape[0] == df_basic.shape[0]),  'oopsie'
    #%% some scans on disk has NAN for 'Gender' and 'Risk'...drop them
    X = X[df_basic['index_data_original']]
    #%% == Create a column with a key indicating session availability ==
    # here i'm using an helper function i created
    df_basic = _ibis_helper_add_key(df_basic)
    #%% reorder columns to my taste
    cols = df_basic.columns.tolist()

    # columns I'd like to show up front
    cols_front = ['ID', 'key', 'Session', 'Subject','Group','DX','Risk',
                  'Gender','Age','file_name']

    # remove the list items from above list using pop.
    for colname in cols_front:
        idx = cols.index(colname)
        cols.pop(idx)

    # append the remaining columns at the tails of the list
    cols = cols_front + cols

    # apply reordering
    df_basic = df_basic[cols]
    #%%******* 1st BLOCK DONE ******
    """========================================================================
       OK BLOCK1 of code is done....if "basic" scores is our only interest,
       return here
       ========================================================================
    """
    if not return_all_scores:
        if return_dropped:
            return X, df_basic, df_basic_dropped
        else:
            return X, df_basic # <- the default output
    #%% ==== read-in ALL scores=====
    #%% now read in sheet with all clinical scores
    df_master = _ibis_helper_get_master_scores(df_basic)

    # ensure indices/row-order remains unchanged
    assert np.array_equal(df_master['ID'].values, df_basic['ID']), 'Assertion fail'

    if return_dropped:
        return X, df_basic, df_master, df_basic_dropped
    else:
        return X, df_basic, df_master


@_deprecated('This one obsolete!  Uses row-major indexing, which makes indexing nightmare in python')
def _get_ibis_vol_dsamped(modality='FA', get_dropped_rows=False,get_all_scores=False):
    """ Get design matrix for IBIS diffusion volume, downsampled by factor of 2.

    - Original resolution:    98 x 116 x 80
    - Downsampled resolution: 49 x  58 x 40

    Usage
    ------
    >>> X, df_basic, extra_info = twio.get_ibis_vol_dsamped()
    >>> X, df_basic, extra_info = twio.get_ibis_vol_dsamped('ALL')
    >>> X, df_basic, extra_info = twio.get_ibis_vol_dsamped('TR')
    >>> X, df_basic, df_dropped, extra_info = twio.get_ibis_vol_dsamped('AX', get_dropped_rows=True)
    >>> X, df_basic, df_master, df_dropped, extra_info = twio.get_ibis_vol_dsamped('RD', get_dropped_rows=True, get_all_scores=True)

    Parameters
    ----------
    modality : string (default='FA')
        string specifying valid modality ('FA', 'TR', 'RD', 'AX','ALL')
        ('ALL' will create a huge-ass design matrix with the columns
        from the 4 modalities concatenated together)
    get_dropped_rows : bool (default: False)
        decides whether to return a DataFrame of the dropped indices as output
    get_all_scores : bool (default: False)
        Output DataFrame of all clinical scores

    Readme (written 10/30/2015)
    -------------------
    Data path of original nii files:

    - ``/home/takanori/data/IBIS_DTI_MAP``


    Matlab Code pipeline for the mat file loaded
    -------------------
    **The order of the pipeline:**

    (1) ``save_IBIS_volume_brainmask_0809.m`` - save brain-mask

        Output: **IBIS_volume_mask_0809.mat**

        - ``IBIS_volume_mask_0809.nii.gz`` and ``IBIS_volume_mask_0809.mat``
          containing variable ``brain_mask``
        - **NOTE: this brain mask is in the original resolution of (98x116x80)**

    (2) ``save_IBIS_downsampled_volume_0809.m`` - downsample volumes from all subjects

        Output: **IBIS_<FA/TR/RD/AX>_downsampled_volumes_0809_2015.mat**

        - ``fileList`` and ``vol_dsamp_ibis`` are the main variable contained
        - note: these are **(dx,dy,dz,nsubject) = (49x58x40x968) 4d array**
        - note this .mat file is currently in ``/home/takanori/data/IBIS_DTI_MAP``
          for storage reason
        - Note: this read and downsamples full-volume nifti volumes individually.


    (3) ``save_IBIS_designMatrix_dsamp_vol_0815.m``: Save designMatrix by taking
          the "vectorized" brain volumes for all subjects (downsampled volume)

        Output: **IBIS_<FA/TR/AX/RD>_designMatrix_dsamped_0815_2015.mat**

        - ``designMatrix``: **(968 x 18722 double array)**
        - ``brain_mask``: **(49x58x40 logical array)** downsampled version of brain mask saved from (1)
        - ``eve176``: ``(49x58x40 uint8 array)** 176 roi label (uint8 suffices to cover 176)
        - ``meta_info``
        - ``graph_info``

    (4) ``save_mean_<FA/TR/AX/RD>_0807.m``: (optional) saves mean diffusion volume at the
         **original resolution of (98x116x80)** (both as ``.mat`` and ``.nii`` file)

        - ``mean_volume_AX_0807.<nii.gz/mat>``
        - ``mean_volume_FA_0807.<nii.gz/mat>``
        - ``mean_volume_RD_0807.<nii.gz/mat>``
        - ``mean_volume_TR_0807.<nii.gz/mat>``

    MAJOR (10/30/2015)
    -------------------
    I replaced 3rd step with ``save_IBIS_designMatrix_dsamp_vol_1030.m`` where
    I saved the data without using **struct**
    """
    if hostname == 'sbia-pc125':
        filepath = '/home/takanori/work-local/tak-ace-ibis'
    else:
        # on the computer cluster
        filepath = '/cbica/home/watanabt/data/ibis/'

    #=== select modality to read ===#
    modality = modality.upper() # for case insensitivity
    if modality == 'FA':
        filename = 'IBIS_FA_designMatrix_dsamped_1030_2015.mat'
    elif modality == 'TR':
        filename = 'IBIS_TR_designMatrix_dsamped_1030_2015.mat'
    elif modality == 'RD':
        filename = 'IBIS_RD_designMatrix_dsamped_1030_2015.mat'
    elif modality == 'AX':
        filename = 'IBIS_AX_designMatrix_dsamped_1030_2015.mat'
    elif modality == 'ALL':
        filename_list = ['IBIS_'+mod+'_designMatrix_dsamped_1030_2015.mat'
                            for mod in ['FA','TR','RD','AX']]
    else:
        print "modality = {} specified not recognized.  Exit program.".format(modality)
        return

    if modality == 'ALL':
        X = []
        for filename in filename_list:
            DATA = _loadmat(os.path.join(filepath,filename),
                            variable_names=['designMatrix'])
            X.append(DATA['designMatrix'])
        # convert columns into ndarray
        X = np.hstack(X)
    else:
        DATA = _loadmat(os.path.join(filepath,filename),
                        variable_names=['designMatrix'])
        X = DATA['designMatrix']
    #=== readin other relevant info ===#
    var_names = ('eve176', 'brain_mask', 'fileList',
                 'graph_augmat','graph_adjmat','graph_incmat',
                 'graph_idx_kept','volume_size','dropped_files',
                 'dropped_files_message')


    DATA = _loadmat(os.path.join(filepath,filename),variable_names=var_names)

    graph_info = {'augmat': DATA['graph_augmat'], # augmentation matrix
                  'adjmat': DATA['graph_adjmat'], # adjacency matrix
                  'incmat': DATA['graph_incmat'], # incidence matrix
                  'idx_kept': DATA['graph_idx_kept'], # can get handy
                  }
    dim = DATA['volume_size']

    # make "dim" a tuple
    x_,y_,z_ = dim[0][0], dim[0][1], dim[0][2]
    dim = (x_,y_,z_)

    # prepend these with "mat" since these volumes got "flipped" in matlab...
    # see ``nii_correct_coord_mat`` for details
    eve176_mat = DATA['eve176']
    brainmask_mat = DATA['brain_mask']

    dropped_files = ndarray_string_to_list(DATA['dropped_files'])
    dropped_files_message = DATA['dropped_files_message']

    #=== to make function-call simple, bundle misc info as single dict
    extra_info = dict(eve176_mat=eve176_mat,
                      brainmask_mat=brainmask_mat,
                      graph_info=graph_info,
                      dropped_files=dropped_files,
                      dropped_files_message=dropped_files_message,
                      fulldatapath=os.path.join(filepath,filename),
                      modality=modality,
                      dim=dim,)
    #return X, extra_info
    #%%************************************************************************
    #%% EVERYTHING BELOW MIRRORS CONNECTOME AND 176ROI SCRIPT
    #%%************************************************************************
    file_list = ndarray_string_to_list(DATA['fileList'])

    # grab the subject id from the filename substring (upper just in case...)
    file_id = [xx[:10].upper() for xx in file_list]

    # create a DataFrame of "ID" and "file_name" (to be used for "join" later)
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original')
    #%% === read in spreadsheet of "basic" clinical scores ===
    df_basic = _ibis_helper_get_basic_scores()

    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_basic = pd.merge(df_file_id,df_basic,how='outer',on='ID')

    # if "file_name" value is nan, then connectome data not available....so drop these rows
    idx_to_drop = df_basic['file_name'].isnull()

    df_basic_dropped = df_basic.ix[idx_to_drop,:]
    df_basic = df_basic.ix[~idx_to_drop,:]

    # ensure no redundant data exists (here check #rows remain unchanged after "drop_duplicates" method
    assert (df_basic.drop_duplicates(['ID']).shape[0] == df_basic.shape[0]),  'fucksie'
    #%% == Create a column with a key indicating session availability ==
    # here i'm using an helper function i created
    df_basic = _ibis_helper_add_key(df_basic)
    #%%******* 1st BLOCK DONE ******
    """========================================================================
       OK BLOCK1 of code is done....if "basic" scores is our only interest,
       return here
       ========================================================================
    """
    if not get_all_scores:
        if get_dropped_rows:
            return X, df_basic, df_basic_dropped, extra_info
        else:
            return X, df_basic, extra_info # <- the default output
    #%% ==== read-in ALL scores=====
    #%% now read in sheet with all clinical scores
    df_master = _ibis_helper_get_master_scores(df_basic)

    # ensure indices/row-order remains unchanged
    assert np.array_equal(df_master['ID'].values, df_basic['ID']), 'Assertion fail'

    if get_dropped_rows:
        return X, df_basic, df_master, df_basic_dropped, extra_info
    else:
        return X, df_basic, df_master, extra_info

def get_ibis_176_rois(multi_index=True,return_dropped=False,return_all_scores=False):
    """ Get 176 ROI diffusion measures for IBIS, FA, TR, AX, RD)

    **11/05/2015** Data i/o updated after migrating to new git repository.

    The structure of the I/O is identical to ``get_ibis_connectomes``, except
    the data matrix is returned as a **Pandas DataFrame** (with hierarchical index,
    where top level index indicate image modality)

    Usage (all cases)
    ------
    The default (the most typical usecase for me)

    >>> df_data, df_basic = twio.get_ibis_176_rois()

    Get dropped rows (added this for potential sanity checks I may want to do in the future)

    >>> df_data, df_basic, df_dropped = twio.get_ibis_176_rois(return_dropped=True)

    Get all clinical scores

    >>> df_data, df_basic, df_master = twio.get_ibis_176_rois(return_all_scores=True)

    Get all clinical scores + dropped rows

    >>> df_data, df_basic, df_master, df_basic_dropped = twio.get_ibis_176_rois(return_all_scores=True,return_dropped=True)

    Handy snippets for cross-sectoinal analysis
    --------------
    >>> df_data, df_basic = get_ibis_176_rois()

    With hierarchical columns, we can access each diffusion measures conviniently

    >>> df_FA = df_data[('FA')]
    >>> df_TR = df_data[('TR')]
    >>> df_AX = df_data[('AX')]
    >>> df_RD = df_data[('RD')]

    Getting data of specific group of interest is made easy

    >>> df_06 = df_master.groupby('Session').get_group('V06').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v12, v24 here)
    >>> df_12 = df_master.groupby('Session').get_group('V12').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v06, v24 here)
    >>> df_24 = df_master.groupby('Session').get_group('V24').dropna(axis=1,how='all') # drop columns with all NANS (corresponds to v06, v12 here)
    >>>  # corresponding connectome data can be accessed by the above DataFrames' "index" attribute
    >>> df_data06 = df_data.ix[df_06.index,:]
    >>> df_data12 = df_data.ix[df_12.index,:]
    >>> df_data24 = df_data.ix[df_24.index,:]

    Get summary of column infos as dict

    >>> col_info = {'v06':twio.get_df_column_info(df_06),
    ...             'v12':twio.get_df_column_info(df_12),
    ...             'v24':twio.get_df_column_info(df_24)}

    Pandas Snippets
    ---------
    For refreshers, go to https://github.com/takwatanabe2004/snippet_book/blob/master/python/snippet-python.rst#pandas-snippets-condensed

    The ``groupby`` and ``get_group`` method is really handy for selecting subset of DataFrames

    >>> df_basic.groupby('Gender').get_group('Male')
    >>> # this index can be used to extract corresponding subjects
    >>> df_basic.groupby('Gender').get_group('Male').index

    More ``groupby`` stuffs

    >>> df_basic.groupby('Session').mean()
    >>> df_basic.groupby('Session').describe()

    Parameters
    ----------
    multi_index : Bool (default = True)
        return DataFrame ``df_data`` with hierarchical columns
    return_dropped : bool (default: False)
        decides whether to return a DataFrame of the dropped indices as output
    return_all_scores : bool (default: False)
        Output DataFrame of all clinical scores

    Returns
    --------
    df_data : DataFrame
        shape [nsubj, 176*4] dataFrame representing the 4 diffusion measures of the eve176 map

    Dev
    ----
    ``/home/takanori/work-local/tak-ace-ibis/python/analysis/__proto/proto_ibis_eve176_data_io.py``
    ``proto_ibis_eve176_data_io.html``
    """
    df = pd.read_csv(filepath_ibis_176rois())

    # extract columns corresponding to the columns
    df_data = df.ix[:,7:]

    if multi_index:
        # create a multi-index using the procedure described in:
        # http://pandas.pydata.org/pandas-docs/stable/advanced.html#creating-a-multiindex-hierarchical-index-object
        arrays = [ ['FA']*176 + ['TR']*176 + ['AX']*176 + ['RD']*176,
                    df_data.columns.tolist()]

        tuples = list(zip(*arrays))

        #multi_index = pd.MultiIndex.from_tuples(tuples)
        multi_index = pd.MultiIndex.from_tuples(tuples, names=['diffusion_type','brain_region'])

        # this creates a multi-level column
        df_data.columns = multi_index
        #df_data = pd.DataFrame(df_data.values, columns=multi_index) #<-equivalent way

    #=== sort out other "meta"info from current sheet ===#
    # extract First 3 rows that'll be used for "join" on master score-sheet
    df_meta = df.ix[:,0:3] # first 3 columns = "ID", "Session", "Lookup"

    # I like Subject ID to be string-type (so it won't go through "mean" and "std"
    # when applying .mean() and .describe() and such
    df_meta['ID'] = df_meta['ID'].astype(str)

    """ establish consistency in the column name with ``get_ibis_connectomes``
    Rename: ID -> Subject
    Rename: Lookup -> ID
    """
    df_meta.rename(columns={'ID':'Subject'},inplace=True)
    df_meta.rename(columns={'Lookup':'ID'},inplace=True)

    # replace strings indicating "Missing" value with nans
    df_meta = df_meta.replace('Missing Data',np.nan)

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_meta = tw.pd_prepend_index_column(df_meta, 'index_data_original')

    #%%=== now "join" with clinical scores sheets ===#
    df_basic = _ibis_helper_get_basic_scores()

    #=== "join" between df_meta (corresponding to original data) and df_basic (spreadsheet of scores)
    merge_keys = ['ID', 'Subject', 'Session']
    df_basic = pd.merge(df_meta, df_basic, how='outer', on=merge_keys)

    # sanity check that index/row has not been shuffled via the "join" operation
    n = df_data.shape[0]
    assert np.array_equal(df_basic.index.values[:n], df_meta.index.values), 'fuck'

    #=== data dropping ====#
    # if "index_data_original" value is nan, then 176ROI data not available....
    # so drop these rows
    idx_to_drop = df_basic['index_data_original'].isnull()

    df_basic_dropped = df_basic.ix[idx_to_drop,:]
    df_basic = df_basic.ix[~idx_to_drop,:]

    # I like Subject ID to be string-type (so it won't go through "mean" and "std"
    # when applying .mean() and .describe() and such
    df_basic['Subject'] = df_basic['Subject'].astype(str)

    #%% Create a column with a key indicating session availability
    # here i'm using an helper function i created
    df_basic = _ibis_helper_add_key(df_basic)
    #%%******* 1st BLOCK DONE ******
    """========================================================================
       OK BLOCK1 of code is done....if "basic" scores is our only interest,
       return here
       ========================================================================
    """
    if not return_all_scores:
        if return_dropped:
            return df_data, df_basic, df_basic_dropped
        else:
            return df_data, df_basic # <- the default output
    #%% ==== read-in ALL scores=====
    #%% now read in sheet with all clinical scores
    #%% now read in sheet with all clinical scores
    df_master = _ibis_helper_get_master_scores(df_basic)

    # ensure indices/row-order remains unchanged
    assert np.array_equal(df_master['ID'].values, df_basic['ID']), 'Assertion fail'

    if return_dropped:
        return df_data, df_basic, df_master, df_basic_dropped
    else:
        return df_data, df_basic, df_master


def get_ibis_brainmask(data_only=True):
    """ Get brain_mask for the fullvolume IBIS diffusion volumes (created 12/03/2015)

    Returns
    ---------
    brain_mask : binary ndarray of shape [49,58,40]
        Brain mask
    """
    if hostname == 'sbia-pc125':
        filepath = '/home/takanori/data/ibis/misc'
    else:
        # on the computer cluster
        filepath = '/cbica/home/watanabt/data/ibis/misc'
    filename = 'IBIS_volume_mask_1108_2015.nii.gz'

    from nibabel import load
    brain_mask = load(os.path.join(filepath,filename))

    if data_only:
        return brain_mask.get_data() # return ndarray
    else:
        return brain_mask


def get_ibis_downsampled_brainmask():
    """ Get brain_mask for the downsampled IBIS diffusion volumes (created 11/09/2015)

    Returns
    ---------
    brain_mask : binary ndarray of shape [49,58,40]
        Brain mask
    """
    if hostname == 'sbia-pc125':
        filepath = '/home/takanori/data/ibis/misc'
    else:
        # on the computer cluster
        filepath = '/cbica/home/watanabt/data/ibis/misc'
    filename = 'IBIS_downsampled_volume_mask_1108_2015.mat'

    brain_mask = _loadmat(os.path.join(filepath,filename),
                          variable_names='brain_mask')['brain_mask']
    return brain_mask

def recon_volume():
    """Just return the lambda function to reconstruct volume from flattened volume"""
    return get_ibis_graph_info_downsampled_volume()['get_vol']


def get_graph_info(brain_mask):
    """ Created **12/03/2015**"""
    import scipy.sparse as sparse
    #| xyz coordinates of the voxels used as features
    #| (note: these are in index space, not real spatial coordinate space)
    xyz = np.vstack(brain_mask.nonzero()).T

    """Create adjacency matrix using scikit"""
    from sklearn.neighbors import radius_neighbors_graph
    A = radius_neighbors_graph(xyz,radius=1,metric='manhattan',include_self=False)

    # incidence matrix
    C = tw.adj2inc(A)

    # Laplacian matrix
    L = C.T.dot(C)

    # degree vector
    #deg = L.diagonal()
    deg = np.array(A.sum(axis=1)).ravel()
    #^^^ here i'm converting a matrix into array, and flattening it)

    """Augmentation matrix"""
    idx_kept = np.nonzero(brain_mask.ravel())[0]
    augmat = sparse.eye(brain_mask.size,format='csr')
    augmat = augmat[:,idx_kept]

    # create lambda function to reconstruct brain volume using augmat
    get_vol = lambda x: (augmat*x).reshape(brain_mask.shape)

    """Return everything as a dict"""
    graph_info = dict(A=A, C=C, L=L, deg=deg, get_vol=get_vol)
    return graph_info


def get_ibis_graph_info_downsampled_volume():
    """ Created **11/09/2015**"""
    import scipy.sparse as sparse
    brain_mask = get_ibis_downsampled_brainmask()

    #| xyz coordinates of the voxels used as features
    #| (note: these are in index space, not real spatial coordinate space)
    xyz = np.vstack(brain_mask.nonzero()).T

    """Create adjacency matrix using scikit"""
    from sklearn.neighbors import radius_neighbors_graph
    A = radius_neighbors_graph(xyz,radius=1,metric='manhattan',include_self=False)

    # incidence matrix
    C = tw.adj2inc(A)

    # Laplacian matrix
    L = C.T.dot(C)

    # degree vector
    #deg = L.diagonal()
    deg = np.array(A.sum(axis=1)).ravel()
    #^^^ here i'm converting a matrix into array, and flattening it)

    """Augmentation matrix"""
    idx_kept = np.nonzero(brain_mask.ravel())[0]
    augmat = sparse.eye(brain_mask.size,format='csr')
    augmat = augmat[:,idx_kept]

    # create lambda function to reconstruct brain volume using augmat
    get_vol = lambda x: (augmat*x).reshape(brain_mask.shape)

    """Return everything as a dict"""
    graph_info = dict(A=A, C=C, L=L, deg=deg, get_vol=get_vol)
    return graph_info


def unmask_ibis_vol(w,mask):
    """ Convert *flattened* [p,] version of the feature vector into volume form of shape [dx,dy,dz]

    **Created 11/24/2015**

    Parameters
    ----------
    w : ndarray of shape [p,]
        "Flattened" version of the feature vector from a diffusion volume
    mask : ndarray of shape [dx,dy,dz]
        Volume mask in volume space.  ``nnz(mask) = p`` required.
    """
    data = np.zeros(mask.shape)
    if mask.dtype != np.bool:
        mask.astype(bool)
    if np.count_nonzero(mask) != w.shape[0]:
        raise RuntimeError('Mask nnz should equal feature size!')
    data[mask.nonzero()] = w
    return data

def _ibis_helper_get_basic_scores():
    """ A helper function for IBIS dataset.

    **11/05/2015** Data i/o updated after migrating to new git repository.

    Currently used in ``get_ibis_connectomes()`` and ``get_ibis_176_rois()``,
    as the stuffs going on are identical, so for the sake of code modularity
    thought it'd be better to create a helper function for this.

    This helper function is only to be used for the above two functions.

    What does this do?
    -------------------
    - Returns the "basic" clinical scores,handling the data/input here.
    - Helps code modularity in the above two functions

    Usage
    ------
    >>> df_basic = _ibis_helper_get_basic_scores()
    """
    #%% collect "basic" scores from first sheet
    """
    [0 'ID',
     1 'file_name',
     2 'Subject',
     3 'Session',
     4 'Risk',
     5 'DX',
     6 'Group',
     7 'Gender',
     8 'Age']"""
    df_basic = pd.read_csv(filepath_ibis_scores_basic())
    #%%
    # replace odd value of 'IBIS2 High Risk' with 'HR'
    df_basic.replace('IBIS2 High Risk','HR',inplace=True)

    # convert "Subject" to strings (I like Subject ID to be string-type,
    # so it won't go through "mean" and "std" when applying .mean()
    # and .describe() and such
    df_basic['Subject'] = df_basic['Subject'].astype(str)

    return df_basic


def ibis_helper_get_key_diffusion(modality='FA'):
    """ Get data availability **"key"** for IBIS diffusion volume **(Created 12/03/2015)

    Create a list of "key" on representing data availability (checked via ``glob``)
    **(note: the "key" below are strings, not int)**

    Key table

    .. csv-table ::
        :header: key (strings\, not int), data availability
        :widths: 2,5
        :delim: |

          '1' | V06 only
          '2' | V12 only
          '3' | V24 only
         '12' | V06 and V12 only
         '13' | V06 and V24 only
         '23' | V12 and V24 only
        '123' | all sessions available
    """
    from glob import glob
    file_path = '/home/takanori/data/ibis/original_data/dti_volumes/'+modality
    file_list_fullpath = glob(file_path+'/*.nii.gz')

    # get list of filenames
    fileList = [os.path.basename(filename_) for filename_ in file_list_fullpath]

    # sort list by ID (ID comes first, so this is easy to handle)
    fileList = sorted(fileList)

    # first 10 chars represent scan ID
    ID = [file[:10] for file in fileList]

    # first 6 chars represent subject ID (w/o session info)
    subject_ID = [file[:6] for file in fileList]

    # the last 3 chars represent session
    session = [file[-3:] for file in ID]

    df = pd.DataFrame([fileList, ID,subject_ID,session],
                        index=['filenames', 'ID','subject_ID','session']).T
    #%% create "keys" for each subjects
    # get unique subject list
    subjects_uniq = sorted(list(set(subject_ID)))
    key_list = []

    # loop over unique subjects
    for i,subj in enumerate(subjects_uniq):
        # extract list of scans corresponding to a subject
        # (eg, ['106211_V06', '106211_V24'])
        subj_scans = [scan for scan in ID if subj in scan]

        key = ''
        for scan in subj_scans:
            # loop over scans from this particular subject, and append key info
            # if scan exists (last 3 chars correspond to scan session)
            if scan[-3:] == 'V06':
                key = key + '1'
            elif scan[-3:] == 'V12':
                key = key + '2'
            elif scan[-3:] == 'V24':
                key = key + '3'
        key_list.append(key)

    df_key = pd.DataFrame([subjects_uniq, key_list],
                          index=['subject_ID','key']).T

    # join key info and return
    return pd.merge(df, df_key, on = ['subject_ID'])


def _ibis_helper_add_key(df_basic):
    """ A helper function for IBIS dataset.

    Currently used in ``get_ibis_connectomes()`` and ``get_ibis_176_rois()``,
    as the stuffs going on are identical, so for the sake of code modularity
    thought it'd be better to create a helper function for this.

    This helper function is only to be used for the above two functions.

    What does this do?
    -------------------
    Takes the IBIS DataFrame, and appends a column of "key" on the DataFrame
    representing data availability **(note: the "key" below are strings, not int)**

    Key table

    .. csv-table ::
        :header: key (strings\, not int), data availability
        :widths: 2,5
        :delim: |

          '1' | V06 only
          '2' | V12 only
          '3' | V24 only
         '12' | V06 and V12 only
         '13' | V06 and V24 only
         '23' | V12 and V24 only
        '123' | all sessions available
    """

    # == turns out subjects are already sorted by ID, creating the key is easy ==
    #np.array_equal(np.sort(df_merge['Subject'].values), df_merge['Subject'].values)
    subjects = list(np.unique(df_basic['Subject']))
    key_list = []
    for i,subj in enumerate(subjects):
        #print i, subj
        session_list = df_basic.query('Subject == @subj')['Session'].values
        #print session_list
        key = ''
        for sess in session_list:
            if sess == 'V06':
                key = key + '1'
            elif sess == 'V12':
                key = key + '2'
            elif sess == 'V24':
                key = key + '3'
        key_list.append(key)

    # df_keys: a 2 column DataFrame of Subject ID and keys, for joining with df_basic
    df_keys = pd.DataFrame(dict(Subject=subjects, key=key_list))

    # join
    df_basic = pd.merge(df_basic, df_keys, how='outer', on=['Subject'])

    return df_basic


def _ibis_helper_get_master_scores(df_basic):
    """ A helper function for IBIS dataset.

    **11/05/2015** Data i/o updated after migrating to new git repository.

    Currently used in ``get_ibis_connectomes()`` and ``get_ibis_176_rois()``,
    as the stuffs going on are identical, so for the sake of code modularity
    thought it'd be better to create a helper function for this.

    This helper function is only to be used for the above two functions.

    What does this do?
    -------------------
    - Returns the "master" clinical scores,handling the data/input here.
    - Helps code modularity in the above two functions

    Usage
    ------
    >>> df_master = _ibis_helper_get_basic_scores(df_basic)

    """
    df_all_scores = pd.read_csv(filepath_ibis_scores_master())

    # again, convert 'Subject' column to string
    df_all_scores['Subject'] = df_all_scores['Subject'].astype(str)

    #== find out which scores corresponds to which session (06,12,24) ==
    #tmp = df_all_scores.columns.str[:3].tolist()
    #v06_score_names = pd.DataFrame([[idx,str_] for idx,str_ in enumerate(df_all_scores.columns) if str_[:3]=='V06'], columns=['idx','V06_scores'])
    #v12_score_names = pd.DataFrame([[idx,str_] for idx,str_ in enumerate(df_all_scores.columns) if str_[:3]=='V12'], columns=['idx','V12_scores'])
    #v24_score_names = pd.DataFrame([[idx,str_] for idx,str_ in enumerate(df_all_scores.columns) if str_[:3]=='V24'], columns=['idx','V24_scores'])
    v06_idx = [0,1,2]+[idx for idx,str_ in enumerate(df_all_scores.columns) if str_[:3]=='V06']
    v12_idx = [0,1,2]+[idx for idx,str_ in enumerate(df_all_scores.columns) if str_[:3]=='V12']
    v24_idx = [0,1,2]+[idx for idx,str_ in enumerate(df_all_scores.columns) if str_[:3]=='V24']

    v06_scores = df_all_scores.ix[:,v06_idx]
    v06_scores.loc[:,'Session'] = ['V06']*v06_scores.shape[0] # <- create column of 'Session', which can be used as a join key
    #         ^^^^^^^^^^^^^^^^^ <- this loc approach is to suppress pandas warning
    # (see http://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas)
    v12_scores = df_all_scores.ix[:,v12_idx]
    v12_scores.loc[:,'Session'] = ['V12']*v12_scores.shape[0] # <- create column of 'Session', which can be used as a join key

    v24_scores = df_all_scores.ix[:,v24_idx]
    v24_scores.loc[:,'Session'] = ['V24']*v24_scores.shape[0] # <- create column of 'Session', which can be used as a join key
    #%% join
    merge_key = ['Subject','Group','Gender','Session']
    df_master = pd.merge(df_basic, v06_scores, how='left', on=merge_key).merge(
                                   v12_scores, how='left', on=merge_key).merge(
                                   v24_scores, how='left', on=merge_key)
    #%% reorder columns to my taste
#    cols = df_master.columns.tolist()
#
#    # columns I'd like to show up front
#    cols_front = ['ID', 'key', 'Session', 'Subject','Group','DX','Risk',
#                  'Gender','Age']
#
#    # remove the list items from above list using pop.
#    for colname in cols_front:
#        idx = cols.index(colname)
#        cols.pop(idx)
#
#    # append the remaining columns at the tails of the list
#    cols = cols_front + cols
#
#    # apply reordering
#    df_master = df_master[cols]
    #%%
    return df_master

#%% --- utility functions for our analysis ----
def util_ibis_gender(modality, session, group, all_sessions=False):
    """ Util function for gender analysis with the IBIS dataset.

    **Created 11/10/2015**

    Parameters
    ----------
    modality : string
        Choices: ``'conn', 'FA', 'TR', 'RD', 'AX', 'ALL'``

        - ``conn``: PROB connectomes
        - ``FA``: FA volume (downsampled)
        - ``TR``: TR volume (downsampled)
        - ``RD``: RD volume (downsampled)
        - ``AX``: AX volume (downsampled)
        - ``ALL``: FA+TR+RD+AX volumes concatenated together (downsampled)
    session : string
        Choices: ``'V06', 'V12', 'V24'``
    group : string
        Choices : ``'LR-', 'HR-', 'ALL'``
    all_sessions : (optional) boolean (default=False) (**Added 11/12/2015**)
        Only include subjects whose scan is available at all 3 sessions

    Returns
    --------
    **X, y, df**

    Example
    -------
    >>> setup = dict(modality='FA', session='V06', group='LR-')
    >>> X, y, df = twio.util_ibis_gender_dvol(**setup)
    """
    if modality.upper() in ['FA','TR','AX','RD','ALL']:
        Xfull, df_full = get_ibis_downsampled_volumes(modality=modality)
    elif modality.lower() in ['conn']:
        Xfull, df_full = get_ibis_connectomes()

    # drop these; it's potentially confusing (I originally added this to
    # emphasize that 10 subjects were dropped in the original desing matrix
    # since their labels were NAN in the scores spreadsheet)
    #del df_full['index_data_original']

    #%%== extract data corresponding to the analysis requested ===
    # create "query" string
    query  = 'Session == @session'

    if group in ['LR-','HR-']:
        query += ' and Group == @group'
    elif group == 'ALL':
        query += ' and (Group == "LR-" or Group == "HR-")'

    if all_sessions:
        query += ' and key == "123"'

    df = df_full.query(query)

    X = Xfull[df.index]

    # create label vector of gender (+1 = male, -1 = female)
    y = (df['Gender']=='Male').values.astype(int)-(df['Gender']=='Female').values.astype(int)

    # reset index for convenience and to avoid confusion
    df.reset_index(inplace=True)

    # rename the index column that arised after the "reset_index" into a more meaningful name
    df.rename(columns={'index':'index_original'},inplace=True)

    # for ease of sanity check, place a column of y labels in the DataFrame
    df['y_gender'] = y
    #%% reorder columns
    # columns I'd like to show up front
    cols_front = ['index_original','Gender','y_gender']

    # filter away cols_front
    cols = cols_front + [_ for _ in df.columns.tolist() if _ not in cols_front]

    # apply desired reordering
    df = df[cols]
    return X, y, df


def util_ibis_gender_delta(modality, key, group, all_sessions=False):
    """ Util function for gender analysis with the "delta" feature for IBIS.

    **Created 11/10/2015**

    Parameters
    ----------
    modality : string
        Choices: ``'conn', 'FA', 'TR', 'RD', 'AX', 'ALL'``

        - ``conn``: PROB connectomes
        - ``FA``: FA volume (downsampled)
        - ``TR``: TR volume (downsampled)
        - ``RD``: RD volume (downsampled)
        - ``AX``: AX volume (downsampled)
        - ``ALL``: FA+TR+RD+AX volumes concatenated together (downsampled)
    key : string
        Choices: ``'12', '13', '23'``

        - ``'12'``: take **delta** between V12 and V06 volumes
        - ``'13'``: take **delta** between V24 and V06 volumes
        - ``'23'``: take **delta** between V24 and V12 volumes
    group : string
        Choices : ``'LR-', 'HR-', 'ALL'``
    all_sessions : (optional) boolean (default=False) (**Added 11/12/2015**)
        Only include subjects whose scan is available at all 3 sessions

    Returns
    --------
    **X, y, df**

    Example
    -------
    >>> setup = dict(modality='FA', session='V06', group='LR-')
    >>> X, y, df = twio.util_ibis_gender_dvol(**setup)
    """
    if key == '12':
        sess1 = 'V06'
        sess2 = 'V12'
    elif key == '13':
        sess1 = 'V06'
        sess2 = 'V24'
    elif key == '23':
        sess1 = 'V12'
        sess2 = 'V24'

    if modality.upper() in ['FA','TR','AX','RD','ALL']:
        Xfull, df_full = get_ibis_downsampled_volumes(modality=modality)
    elif modality.lower() in ['conn']:
        Xfull, df_full = get_ibis_connectomes()

    # drop these; it's potentially confusing (I originally added this to
    # emphasize that 10 subjects were dropped in the original desing matrix
    # since their labels were NAN in the scores spreadsheet)
    del df_full['index_data_original']

    # now update it with the actual original indexing we have
    df_full = tw.pd_prepend_index_column(df_full, colname='index_original')
    #%% get DataFrames at the 2 requested sessions
    if all_sessions:
        # only select subjects with all 3 scans
        query1 = 'key == "123" and Session == @sess1'
        query2 = 'key == "123" and Session == @sess2'
    else:
        query1 = 'key in [@key, "123"] and Session == @sess1'
        query2 = 'key in [@key, "123"] and Session == @sess2'
    df1 = df_full.query(query1)
    df2 = df_full.query(query2)

    # sanity check that we have the same subjects here
    assert np.all(df1.Subject == df2.Subject)

    # compute "delta" volumes
    Xdelta = Xfull[df2.index] - Xfull[df1.index]
    #%% merge data-frames from two sessions
    # delete these info, as these are redundant
    del df1['Session'], df2['Session']

    df_delta = pd.merge(df1,df2, how='outer',
                  on=['Subject','key', 'DX', 'Risk','Gender','Group'],
                  suffixes=('_'+sess1,'_'+sess2))
    df_delta = tw.pd_sort_col(df_delta)
    #return Xdelta, df_delta
    #%% extract data corresponding to the analysis requested
    if group in ['LR-','HR-']:
        df = df_delta.query('Group == @group')
    elif group == 'ALL':
        df = df_delta.query('Group == "LR-" or Group == "HR-"')
    #%% create label vector of gender (+1 = male, -1 = female)
    y = (df['Gender']=='Male').values.astype(int) \
       -(df['Gender']=='Female').values.astype(int)

    # extract indices corresponding to above grouops
    X = Xdelta[df.index]

    # reset index for convenience and to avoid confusion
    df.reset_index(drop=True,inplace=True)
    return X,y,df


def util_ibis_HRp_LRm(modality, session, male_only = False):
    """ Util function for HR+ vs LR- analysis with the IBIS dataset.

    **Created 11/10/2015**

    Parameters
    ----------
    modality : string
        Choices: ``'conn', 'FA', 'TR', 'RD', 'AX', 'ALL'``

        - ``conn``: PROB connectomes
        - ``FA``: FA volume (downsampled)
        - ``TR``: TR volume (downsampled)
        - ``RD``: RD volume (downsampled)
        - ``AX``: AX volume (downsampled)
        - ``ALL``: FA+TR+RD+AX volumes concatenated together (downsampled)
    session : string
        Choices: ``'V06', 'V12', 'V24'``
    male_only : bool (default=False)
        Option to restrict analysis to male.

        Why?

        - HR+ consist mostly of male, creating a problem when doing group-comparisons
          with the LR- group (which is much more balanced in Male/Female)
        - thus it may be useful to drop all females for this analysis.

    Returns
    --------
    **X, y, df**

    Example
    -------
    >>> setup = dict(modality='TR', session='V24',male_only=True)
    >>> X, y, df = twio.util_ibis_HRp_LRm_dvol(**setup)
    """
    if modality.upper() in ['FA','TR','AX','RD','ALL']:
        Xfull, df_full = get_ibis_downsampled_volumes(modality=modality)
    elif modality.lower() in ['conn']:
        Xfull, df_full = get_ibis_connectomes()

    # drop these; it's potentially confusing (I originally added this to
    # emphasize that 10 subjects were dropped in the original desing matrix
    # since their labels were NAN in the scores spreadsheet)
    #del df_full['index_data_original']

    # now update it with the actual original indexing we have
    df_full = tw.pd_prepend_index_column(df_full, colname='index_original')

    if male_only:
        query1 = 'Session == @session and Gender == "Male" and Group == "HR+"'
        query2 = 'Session == @session and Gender == "Male" and Group == "LR-"'
        df_hrp = df_full.query(query1)
        df_lrm = df_full.query(query2)
    else:
        df_hrp = df_full.query('Session == @session and Group == "HR+"')
        df_lrm = df_full.query('Session == @session and Group == "LR-"')

    # extract indices corresponding to above grouops
    X = np.concatenate([Xfull[df_hrp.index],Xfull[df_lrm.index]])

    # create label vector (+1 = HR+, -1 = LR-)
    y = np.concatenate([+np.ones(df_hrp.shape[0]),-np.ones(df_lrm.shape[0])])
    df = pd.concat([df_hrp,df_lrm])

    # reset DF index to allow indexing by position
    df.reset_index(drop=True,inplace=True)

    # for ease of sanity check, place a column of y labels in the DataFrame
    df['y_group'] = y
    #%% reorder columns
    # columns I'd like to show up front
    cols_front = ['index_original','Group','y_group']

    # filter away cols_front
    cols = cols_front + [_ for _ in df.columns.tolist() if _ not in cols_front]

    # apply desired reordering
    df = df[cols]
    return X, y, df


def util_ibis_HRp_LRm_delta(modality, key, male_only=False):
    """ Util function for HR+ vs LR- analysis with the "delta" feature for IBIS.

    **Created 11/10/2015**

    Parameters
    ----------
    modality : string
        Choices: ``'conn', 'FA', 'TR', 'RD', 'AX', 'ALL'``

        - ``conn``: PROB connectomes
        - ``FA``: FA volume (downsampled)
        - ``TR``: TR volume (downsampled)
        - ``RD``: RD volume (downsampled)
        - ``AX``: AX volume (downsampled)
        - ``ALL``: FA+TR+RD+AX volumes concatenated together (downsampled)
    key : string
        Choices: ``'12', '13', '23'``

        - ``'12'``: take **delta** between V12 and V06 volumes
        - ``'13'``: take **delta** between V24 and V06 volumes
        - ``'23'``: take **delta** between V24 and V12 volumes
    male_only : bool (default=False)
        Option to restrict analysis to male.

        Why?

        - HR+ consist mostly of male, creating a problem when doing group-comparisons
          with the LR- group (which is much more balanced in Male/Female)
        - thus it may be useful to drop all females for this analysis.

    Returns
    --------
    **X, y, df**

    Example
    -------
    >>> setup = dict(modality='TR', session='V24',male_only=True)
    >>> X, y, df = twio.util_ibis_HRp_LRm_dvol(**setup)
    """
    if key == '12':
        sess1 = 'V06'
        sess2 = 'V12'
    elif key == '13':
        sess1 = 'V06'
        sess2 = 'V24'
    elif key == '23':
        sess1 = 'V12'
        sess2 = 'V24'

    if modality.upper() in ['FA','TR','AX','RD','ALL']:
        Xfull, df_full = get_ibis_downsampled_volumes(modality=modality)
    elif modality.lower() in ['conn']:
        Xfull, df_full = get_ibis_connectomes()

    # drop these; it's potentially confusing (I originally added this to
    # emphasize that 10 subjects were dropped in the original desing matrix
    # since their labels were NAN in the scores spreadsheet)
    del df_full['index_data_original']

    # now update it with the actual original indexing we have
    df_full = tw.pd_prepend_index_column(df_full, colname='index_original')
    #%% get DataFrames at the 2 requested sessions
    query1 = 'key in [@key, "123"] and Session == @sess1'
    query2 = 'key in [@key, "123"] and Session == @sess2'
    df1 = df_full.query(query1).dropna(subset=['Group'])
    df2 = df_full.query(query2).dropna(subset=['Group'])

    # sanity check that we have the same subjects here
    assert np.all(df1.Subject == df2.Subject)

    # compute "delta" volumes
    Xdelta = Xfull[df2.index] - Xfull[df1.index]
    #%% merge data-frames from two sessions
    # delete these info, as these are redundant
    del df1['Session'], df2['Session']

    df_delta = pd.merge(df1,df2, how='outer',
                  on=['Subject','key', 'DX', 'Risk','Gender','Group'],
                  suffixes=('_'+sess1,'_'+sess2))
    df_delta = tw.pd_sort_col(df_delta)
    #return Xdelta, df_delta
    #%% next, take only subjects corresponding to HR+ or LR-
    if male_only:
        df_hrp = df_delta.query('Gender == "Male" and Group == "HR+"')
        df_lrm = df_delta.query('Gender == "Male" and Group == "LR-"')
    else:
        df_hrp = df_delta.query('Group == "HR+"')
        df_lrm = df_delta.query('Group == "LR-"')

    # extract indices corresponding to above grouops
    X = np.concatenate([Xdelta[df_hrp.index],Xdelta[df_lrm.index]])

    # create label vector (+1 = HR+, -1 = LR-)
    y = np.concatenate([+np.ones(df_hrp.shape[0]),-np.ones(df_lrm.shape[0])])
    df = pd.concat([df_hrp,df_lrm])

    # reset DF index to allow indexing by position
    df.reset_index(drop=True,inplace=True)

    # for ease of sanity check, place a column of y labels in the DataFrame
    df['y_group'] = y

    #%% reorder columns
    # columns I'd like to show up front
    cols_front = ['Group','y_group','Gender',]
#                  'index_original_'+sess1,'index_original_'+sess2,]

    # filter away cols_front
    cols = cols_front + [_ for _ in df.columns.tolist() if _ not in cols_front]

    # apply desired reordering
    df = df[cols]
    return X, y, df
#%%==== PNC stuffs ====
def get_pnc_connectomes(return_as_design = True, return_all_scores=False):
    """ Get connectomes and clinical scores for the pnc data

    **11/05/2015** Data i/o updated after migrating to new git repository.

    Returns
    ------
    X : ndarray
        (n x p) design matrix
    y : ndarray
        (n,) label array (+1 = male, -1 = female)
    df_scores : DataFrame with n rows
        DataFrame containing clinical scores of the n subjects.
        Also contains subject category quantized over 3 age group
        (labeled ``q1, q2, q3``)

    Parameters
    -----------
    return_as_design : bool (default: True)
        How to return the connectomes

        - If True, return the connectomes as an (n x p) 2d array (p = # edges)
        - If False, return connectomes as (nROI x nROI x n) 3d array

    return_all_scores : bool (default : False)
        Return all clinical scores available in the original clinical scoresheeet
        (default is False, which returns basic subject info such as age, gender, ID)


    Usage
    --------
    **Most typical usecase**: X = [n,p] design matrix, y = [n,] label array of gender (+1 = male, -1 = female)

    >>> X, y, df_scores = get_pnc_connectomes()

    Get all scores available (fatter ``df_scores`` DataFrame)

    >>> X, y, df_scores = get_pnc_connectomes(return_all_scores=True)

    Return X as 3d connArray of dimension [nROI x nROI x nSubjects]

    >>> X, y, df_scores = get_pnc_connectomes(return_as_design=False)

    Extract "age_bins" (couldn't think of a better way at the moment; return
    the first instance of the query catch)

    >>> age_bins = [df_scores.query('age_group == "q1"')['age_bins'].tolist()[0],
    ...             df_scores.query('age_group == "q2"')['age_bins'].tolist()[0],
    ...             df_scores.query('age_group == "q3"')['age_bins'].tolist()[0]]
    Out[336]: ['[8.167, 13.333]', '(13.333, 17.083]', '(17.083, 21.833]']

    Snippets
    ---------
    >>> X, y, df_scores = twio.get_pnc_connectomes(return_all_scores=True)

    Break data up over 3 age quantized age-group

    >>> df_scores_q1 = tw.pd_prepend_index_column(df_scores.groupby('age_group').get_group('q1'))
    >>> df_scores_q2 = tw.pd_prepend_index_column(df_scores.groupby('age_group').get_group('q2'))
    >>> df_scores_q3 = tw.pd_prepend_index_column(df_scores.groupby('age_group').get_group('q3'))
    >>> idx1 = df_scores_q1.index.values
    >>> idx2 = df_scores_q2.index.values
    >>> idx3 = df_scores_q3.index.values
    >>> X_q1,y_q1 = X[idx1],y[idx1]
    >>> X_q2,y_q2 = X[idx2],y[idx2]
    >>> X_q3,y_q3 = X[idx3],y[idx3]

    Note
    ------
    - no subjects meet the "drop" requirement for this PNC data
      (all files connectomes disk has entry in the clinical scores spreadsheet)
    - therefore, no ``return_dropped`` flag option is included for this function,
      unlike with other datasets

    Dev file
    --------
    - ``proto_pnc_data_io_correted_1101.py``

    Warning
    --------
    11/02/2015 - renamed columns ['age-bins'] and ['age-group'] to ['age_bins']
    and ['age_group'] (ie, replaced dash with underscore)

    NEVER use dash in column name, as it will complicate things when using ``query`` method
    """
    #%%=== first get data matrix of connectomes from matlab ====#
    var_names = ('connMatrix','fileList')

    DATA = _loadmat(filepath_pnc_conn(),variable_names=var_names)

    if return_as_design:
        X = tw.conn2design(DATA['connMatrix'])
    else:
        X = DATA['connMatrix']

    file_list_in = DATA['fileList']

    # file_list comes as ndarray...kinda tricky/awkward to handle...convert to list
    file_list = []
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    # grab the subject id from the filename substring (upper just in case...)
    file_id = [xx[:13].upper() for xx in file_list]

    # create a DataFrame of "ID" and "file_name" (to be used for "join" later)
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original')
    #return X, df_file_id
    #%% === read in spreadsheet of clinical scores ===
    df_scores = pd.read_excel(filepath_pnc_scores())
    #return X, df_file_id, df_scores
    #%% === "join" df_file_list with df_basic['ID'] ===
    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_scores = pd.merge(df_file_id,df_scores,how='outer',on='ID')
    #%% everything below is mostly sanity checks (verified)
#    # if "file_name" value is nan, then connectome data not available....so drop these rows
#    idx_to_drop = df_scores['file_name'].isnull()
#
#    """TURNS OUT NO SUBJECTS GET DROPPED IN PNC!!! WE'RE GOOD!"""
#    #df_scores_dropped = df_scores.ix[idx_to_drop,:]
#    df_scores = df_scores.ix[~idx_to_drop,:]
#
#    # ensure no redundant data exists (here check #rows remain unchanged after "drop_duplicates" method
#    assert (df_scores.drop_duplicates(['ID']).shape[0] == df_scores.shape[0]),  'huh?'
#
#    """ Silly sanity check
#    ===========================================================================
#    Turns out after the above "dropping", the order of the X rows (subjects)
#    appears to not have changed.
#    Verify this by asserting the DataFrame "ID" and the original file_list
#    string agrees with each other
#    ===========================================================================
#    # argh, ``assert`` is a statement, not a function. I've been using it like
#    # a function, in which case Warning message of "assertion is always true"
#    # always kepts popping out
#    """
#    for id1,id2 in zip(file_id, df_scores['ID'].values.tolist()):
#        #assert(id1==id2, 'oh noes!')
#        #^^^^^ wrong syntax
#        assert (id1==id2), 'oh noes!'
#
#    checker = pd.DataFrame(dict(id1=file_id, id2 = df_scores['ID'].values.tolist()))
#    checker['check'] = checker['id1']==checker['id2']
#    assert np.all(checker['check']), 'oh no'
    #%% return label vectors of gender
    # gender: +1 = male, -1 = female
    idx_male = np.array([_ == 'Male' for _ in df_scores['GENDER']])
    idx_fema = np.array([_ == 'Female' for _ in df_scores['GENDER']])
    y_gender = np.zeros(df_scores.shape[0],dtype=int)
    y_gender[idx_male] = +1
    y_gender[idx_fema] = -1
    #%% add a "quantized" age row (quantized into 3 blocks)
    """Here, code looks odd, but the pandas 'qcut' object is somewhat
    awkward to handle.  For one, the Variable Explorer pandas does not like it,
    as it is apparently a special data structure with attributes/methods i
    propery will never care about.

    So here, I'm going to apply "qcut" twice: once to get the quantized labels,
    and second time with my own mnemonic ['q1','q2','q3'] so I can easily
    access them when coding (else, i can't memorize shit like (12.77,21.833]
    for accessing a column of a DataFrame"""
    df_scores['age_group'] = pd.qcut(df_scores['SCAN_AGE_YEARS'],q=3,
                                     labels=['q1','q2','q3'])
    df_scores['age_bins'], age_bin = pd.qcut(df_scores['SCAN_AGE_YEARS'],q=3,
            labels=None, retbins=True)

    # convert above columns into string-type, a form i'm much more comfortable handling
    df_scores['age_group'] = df_scores['age_group'].astype(str)
    df_scores['age_bins']  = df_scores['age_bins'].astype(str)
    #%%=== ALMOST DONE!  BEFORE OUPUTTING, change column names and order =====
    """This step not critical but it makes it easier for me to handle the DataFrame"""
    # rename "SCAN_AGE_YEARS" and "GENDER" to simply "Age" and "Gender"
    df_scores.rename(columns=dict(SCAN_AGE_YEARS='Age',GENDER='Gender'),inplace=True)

    # just for kicks (well, for sanity check), add column of binarized gender-label
    df_scores['y_gender'] = y_gender

    #%% reorder columns to my taste
    cols = df_scores.columns.tolist()

    # columns I'd like to show up front
    cols_front = ['index_data_original','age_group','age_bins','Age','Gender','y_gender']

    # remove the list items from above list using pop.
    for colname in cols_front:
        idx = cols.index(colname)
        cols.pop(idx)

    # append the remaining columns at the tails of the list
    cols = cols_front + cols

    # apply reordering
    df_scores = df_scores[cols]
    #%%___ ALL DONE!  OUTPUT
    if return_all_scores:
        return X, y_gender, df_scores#, age_bin
    else:
        # return only the basic/major scores (the one i normally deal with)
        return X, y_gender, df_scores.ix[:,:8]


def util_pnc(age_group):
    """ Util function for gender analysis with the PNC dataset.

    **Created 11/11/2015**

    Parameters
    ----------
    modality : string
        Choices: ``'q1', 'q2', 'q3', 'all'``

        - ``q1``: select age-group1 [8.167, 13.333] years
        - ``q2``: select age-group2 (13.333, 17.083] years
        - ``q3``: select age-group3 (17.083, 21.833] years
        - ``all``: data

    Returns
    --------
    **X, y, df**

    Example
    -------
    >>> X, y, df = twio.util_pnc(age_group='q1')

    History
    -------
    - 02/01/2016: updated to return all scores.  also added ``RuntimeError`` exception.
    """
    X, y, df = get_pnc_connectomes(return_all_scores=True)

    if age_group == 'all':
        # do nothing - use all data
        pass
        #print "Use all PNC data"
    elif age_group in ['q1','q2','q3']:
        idx = df.query('age_group == @age_group').index.values
        X, y, df = X[idx], y[idx], df.ix[idx,:]
        #print "Select age-group:{} --- age-bins = {}".format(age_group, age_bin)
        #print "num_males = {:3}, num_females = {:3}".format( (y==+1).sum(), (y==-1).sum())
    else:
        raise RuntimeError("Invalid argument.  Input should be either: 'q1','q2','q3'")
    return X,y,df




#%% ==== TBI stuffs ====
def get_tbi_connectomes(return_as_design = True, return_dropped=False,
                        return_all_scores=False):
    """ Get connectome data for the TBI data

    Subjects with ``dx`` or ``gender`` label missing are dropped.

    **11/05/2015** Data i/o updated after migrating to new git repository.

    **11/06/2015 Updates**

    - Added column ``index_data_original`` to output df (useful for sanity checking on data alignment)
    - Added option ``return_all_scores``

    **12/07/2015 Updates**

    - Added column ``key`` indicating scan availability (important for TBI subjects)
    - renamed column ``Subject ID`` to ``Subject_ID`` (allows the use of ``query`` method)

    Parameters
    -----------
    return_as_design : bool (default: True)
        How to return the connectomes

        - If True, return the connectomes as an (n x p) 2d array (p = # edges)
        - If False, return connectomes as (nROI x nROI x n) 3d array

    return_dropped : (default: False)
        decides whether to return a DataFrame of the dropped indices as 4th output)
    return_all_scores : bool (default : False)
        Return all clinical scores available in the original clinical scoresheeet
        (default is False, which returns basic subject info such as age, gender, ID)

    Usage
    -----
    **Most typical usecase**: X = [n,p] design matrix, y = [n,] label array of TBI status (+1 = TBI)

    >>> X,y,df_scores = twio.get_tbi_connectomes()

    Return X as 3d connArray of dimension [nROI x nROI x nSubjects]

    >>> connAarray, y, df_scores = twio.get_tbi_connectomes(return_as_design=False)

    Also return the dropped indices (rows)

    >>> X,y,df_scores,df_dropped=twio.get_tbi_connectomes(return_dropped=True)

    Get all clinical scores available

    >>> X,y,df_master,df_master_dropped=twio.get_tbi_connectomes(return_dropped=True,return_all_scores=True)

    Extract by Scan session (note: HC has single scan, so they are included in all cases below)

    >>> df1 = df_scores.query('y_dx == -1 or (Scan == 1 and y_dx == 1)')
    >>> X1, y1 = X[df1.index], y[df1.index]
    >>> df2 = df_scores.query('y_dx == -1 or (Scan == 2 and y_dx == 1)')
    >>> X2, y2 = X[df2.index], y[df2.index]
    >>> df3 = df_scores.query('y_dx == -1 or (Scan == 3 and y_dx == 1)')
    >>> X3, y3 = X[df3.index], y[df3.index]

    Can also play around with hierarchical indexing on the TBI subjects

    >>> df_tbi = df_scores.query('y_dx == 1')
    >>> # try hierarchical indexing ()
    >>> df_tbi_hi = df_tbi.set_index('Scan', append=True,drop=False).swaplevel(0,1) # set drop=False since i want the "Scan" column to remain there for sanity check
    >>> df_tbi_hi.sortlevel(level=0, inplace=True) # let's sort rows according to Scan
    >>> df_tbi_hi['idx']=range(df_tbi_hi.shape[0]) # i find it helpful to have range(#rows) for indexing the rows...add this as 1st column
    >>> cols = df_tbi_hi.columns.tolist()
    >>> df_tbi_hi = df_tbi_hi[ cols[-1:] + cols[:-1] ]
    >>> # cross-section is your friend in HierArchical Indexing
    >>> df_tbi_hi.xs(1,level='Scan').head()
    >>> df_tbi_hi.xs(2,level='Scan').head()
    >>> df_tbi_hi.xs(3,level='Scan').head()

    Development
    -----------
    See ``~/tak-ace-ibis/python/tbi/proto/proto_tbi_data_io.py``
    """
    var_names = ('connMatrix','fileList')
    DATA = _loadmat(filepath_tbi_conn(),variable_names=var_names)

    if return_as_design:
        X = tw.conn2design(DATA['connMatrix'])
    else:
        X = DATA['connMatrix']

    file_list_in = DATA['fileList']

    # file_list comes as ndarray...kinda tricky/awkward to handle...convert to list
    file_list = []
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    # grab the subject id from the filename substring
    file_id = [xx[:7] for xx in file_list]

    # create a dataframe of "ID" and "file_name"
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original')
    #%% get basics scores ('age','gender','dx')
    df_scores = pd.read_excel(filepath_tbi_scores())

    # replace entries with period "." with nans
    df_scores = df_scores.replace('.',np.nan).copy()

    # rename column 'sbiaID' to 'ID'
    df_scores.rename(columns={'sbiaID':'ID'},inplace=True)

    # (12/07/2015) rename column 'Subject ID' to 'Subject_ID' (allows use of query)
    df_scores.rename(columns={'Subject ID':'Subject_ID'},inplace=True)
    #%% === "join" ====
    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_merge = pd.merge(df_file_id,df_scores,how='outer',on='ID')

    # if "file_name" value is nan, then connectome data not available....so drop these rows
    idx_to_drop = df_merge['file_name'].isnull()

    df_dropped = df_merge.ix[idx_to_drop,:]
    df_merge = df_merge.ix[~idx_to_drop,:]

    #df_merge.dropna(axis=0,subset=['file_name'],inplace=True) #<-same as above
    #%% sanity check that the order of data remains unchanged
    #==========================================================================
    # clusterfuck with array equality assertion...
    #==========================================================================
    #==== HOLY FUCK, I DONT KNOW WHY THE ASSERTION BELOW RAISES AN EXCEPTION
    # THE TEST RETURNS TRUE! DAFUCK????
    # - just use an if statement...god fuck...
    if not np.all(df_file_id['ID'].values == df_merge['ID'].values):
        raise Exception('oh noes...')
    #assert(np.all(df_file_id['ID'].values == df_merge['ID'].values), 'fuck...data/row order fucked up...')
    #^^^^^^ this assertion raises an exception....WHY PYTHON WHY???
    # oh so is this the reason?
    # http://stackoverflow.com/questions/3302949/whats-the-best-way-to-assert-for-numpy-array-equality
    np.testing.assert_equal(df_file_id['ID'].values, df_merge['ID'].values, 'fuck')
    #==========================================================================
    #%% add a binary column indicating gender and TBI status
    # disease: +1 = BI (brain injury), -1 = HC
    idx_BI = (df_merge['ID'].str[0] == 'p').values # first char in ID = 'p'
    idx_HC = (df_merge['ID'].str[0] == 'c').values # first char in ID = 'c'
    y_dx = np.zeros(df_merge.shape[0],dtype=int)
    y_dx[idx_BI] = +1
    y_dx[idx_HC] = -1

    # gender: +1 = male, -1 = female
    idx_male = np.array([_ == 'M' for _ in df_merge['Gender']])
    idx_fema = np.array([_ == 'F' for _ in df_merge['Gender']])
    y_gender = np.zeros(df_merge.shape[0],dtype=int)
    y_gender[idx_male] = +1
    y_gender[idx_fema] = -1

    # add above labels to DataFrame
    df_merge['y_dx']     = y_dx
    df_merge['y_gender'] = y_gender
    #return df_merge, df_dropped
    #%% 12/07/2015 add "key" indicating subject availability
    subjects = sorted(list(set(df_merge['Subject_ID'])))
    key_list = []
    for i,subj in enumerate(subjects):
        #print i, subj
        session_list = df_merge.query('Subject_ID == @subj')['Scan'].values
        #print session_list
        key = ''
        for sess in session_list:
            if sess == 1:
                key = key + '1'
            elif sess == 2:
                key = key + '2'
            elif sess == 3:
                key = key + '3'
        key_list.append(key)

    df_key = pd.DataFrame([subjects, key_list],
                          index=['Subject_ID','key']).T

    df_merge = pd.merge(df_merge, df_key, on = ['Subject_ID'])
    #%% reorder columns to my taste
    cols = df_merge.columns.tolist()

    # columns I'd like to show up front
    cols_front = ['index_data_original','ID', 'Subject_ID', 'Scan','key','y_dx',
                  'file_name','Age','Gender','y_gender']

    # remove the list items from above list using pop.
    for colname in cols_front:
        idx = cols.index(colname)
        cols.pop(idx)

    # append the remaining columns at the tails of the list
    cols = cols_front + cols

    # apply reordering
    df_merge = df_merge[cols]

    if not return_all_scores:
        # only take a subset of the DataFrame columns
        df_merge = df_merge[cols_front]
    #%%=== finally done; return results ===
    if return_dropped:
        return X, y_dx, df_merge, df_dropped
    else:
        return X, y_dx, df_merge

def get_tbi_key_info():
    """ Get the key info of the clinical scores of TBI

    **11/05/2015** Data i/o updated after migrating to new git repository.
    """
    df_con = pd.read_excel(filepath_tbi_scores(), sheetname='Key_controls')
    df_pat = pd.read_excel(filepath_tbi_scores(), sheetname='Key_patients')
    df_key_info = pd.merge(df_con,df_pat,how='outer',on='Variable',
                           suffixes=('_Controls', '_Patients'))
    # check if the definitions agree (add a boolean column)
    df_key_info['def_agrees'] = df_key_info['Definition_Controls'].values == \
                                df_key_info['Definition_Patients'].values

    return df_key_info


def get_tbi_dropped_rows():
    """ Return DataFrame of dropped rows in  ``get_tob_invariants_batch``

    **note**: I dropped duplicate scans from same subject, taking the first scan in the DataFrame

    See ``get_tob_invariants_batch``
    """
    _,_,_,df_dropped = get_tbi_connectomes(return_dropped=True)
    return df_dropped


def util_tbi(scan):
    """ Util function for classification study with the tbi dataset.

    **Created 11/11/2015**

    Parameters
    ----------
    scan : integer
        1, 2, or 3

    Returns
    --------
    **X, y, df**

    Example
    -------
    >>> X,y,df = twio.util_tbi(scan=1)

    -----
    **Update 11/15/2015**

    When argv's are inputted from shell, we need to convert the num2str
    """
    if not isinstance(scan,int):
        # needed when taking argv from shell
        scan = int(scan)

    X, y, df = get_tbi_connectomes(return_all_scores=True)

    df = df.query('y_dx == -1 or (Scan == @scan and y_dx == 1)')
    X, y = X[df.index], y[df.index]
    df.reset_index(drop=True,inplace=True)
    return X, y, df
#%% ==== Tobacco stuffs =====
def get_tob_connectomes(return_as_design = True, return_dropped=False,
                        return_all_scores=False):
    """ Get connectome data for the tobacco autism data

    **05/09/2016** Now using 100 streamline connectomes

    **11/05/2015** Data i/o updated after migrating to new git repository.

    **11/06/2015 update** added option ``return_all_scores``

    Subjects with ``dx`` or ``gender`` label missing are dropped.

    Parameters
    -----------
    return_as_design : bool (default: True)
        How to return the connectomes

        - If True, return the connectomes as an (n x p) 2d array (p = # edges)
        - If False, return connectomes as (nROI x nROI x n) 3d array

    return_dropped : (default: False)
        decides whether to return a DataFrame of the dropped indices as 4th output)
    return_all_scores : bool (default : False) (**added 11/05/2015**)
        Return all clinical scores available in the original clinical scoresheeet
        (default is False, which returns basic subject info such as age, gender, ID)

    Usage
    -----
    Most frequenly used case (y = autism label, +1 = ASD, -1 = TDC)

    >>> X, y, df_basic = get_tob_connectomes()

    Returns DataFrame rows that were dropped as well

    >>> X, y, df_basic, df_dropped = twio.get_tob_connectomes(return_dropped=True)

    Return ALL clinical scores, together with the dropped subjects

    >>> X, y, df_master, df_master_dropped = twio.get_tob_connectomes(return_all_scores=True, return_dropped=True)

    Return as 3d connArray

    >>> connAarray, y_gender,df_basic, df_dropped = get_tob_connectomes(return_as_design=False,return_dropped=True)




    Important remark
    -----------------
    - The #rows in the design matrix ``X`` is smaller than what is stored in the
      ``.mat`` file (goes from 286 to 280)
    - This is because for 6 subjects, the ``gender`` and ``dx`` entry had NAN values
    - These are the 6 subjects I am speaking of:

    ::

        R0029_V0066_connmat_symm.txt
        R0037_V0081_connmat_symm.txt
        R0064_V0128_connmat_symm.txt
        R0086_V0172_connmat_symm.txt
        R0353_V0783_connmat_symm.txt
        R0398_V0966_connmat_symm.txt

    - Set ``return_dropped = True`` to see what's going on

    Development
    -----------
    See ``~/tak-ace-ibis/python/tobacco/proto/proto_tobacco_data_io_scores.py``
    """
    var_names = ('connMatrix','fileList')

    DATA = _loadmat(filepath_tob_conn(),variable_names=var_names)

    if return_as_design:
        X = tw.conn2design(DATA['connMatrix'])
    else:
        X = DATA['connMatrix']

    file_list_in = DATA['fileList']

    # file_list comes as ndarray...kinda tricky/awkward to handle...convert to list
    file_list = []
    for i in xrange(len(file_list_in)):
        file_list.append(str(file_list_in[i][0][0]))

    # grab the subject id from the filename substring
    file_id = [xx[:11] for xx in file_list]

    # create a dataframe of "ID" and "file_name"
    df_file_id = pd.DataFrame({'ID':file_id, 'file_name':file_list})

    """ For sanity-checks, prepend a column indicating original column indexing
        of the data-dataframe (note: df_meta indexes agree with df_data)"""
    df_file_id = tw.pd_prepend_index_column(df_file_id, 'index_data_original', col_at_end=True)

    #%% get basics scores ('age','gender','dx')
    df_scores = pd.read_csv(filepath_tob_scores())

    # replace entries with period "." with nans
    df_scores = df_scores.replace('.',np.nan).copy()

    # convert column dtypes (as it stands, all dtypes of "object", which is too generic)
    # (eg, cant even use plots with "object" dtypes)
    # http://stackoverflow.com/questions/21197774/assign-pandas-dataframe-column-dtypes
    df_scores = df_scores.convert_objects(convert_numeric=True)

    #return X, df_scores, df_file_id
    #%% === "join" ===
    #=========================================================================#
    # join based on ID (outer join)
    #-------------------------------------------------------------------------#
    # http://pandas.pydata.org/pandas-docs/stable/merging.html#database-style-dataframe-joining-merging
    #=========================================================================#
    # apply "outer-join" (logical OR), then drop records with "NAN" on "file-name"
    df_merge = pd.merge(df_file_id, df_scores, how='outer',on=['ID'])
    #return X, df_scores, df_file_id, df_merge

    #=========================================================================#
    # criteria for drop:
    # - either "file_name", "gender", or "dx" entry is nan
    #=========================================================================#
    idx_to_drop1 = df_merge['file_name'].isnull().values
    idx_to_drop2 = df_merge['gender'].isnull().values
    idx_to_drop3 = df_merge['dx'].isnull().values

    # if "file_name" value is nan, then connectome data not available....so drop these rows
    idx_to_drop = (idx_to_drop1 | idx_to_drop2 | idx_to_drop3)

    # drop rows
    df_dropped = df_merge.ix[idx_to_drop,:]
    df_merge   = df_merge.ix[~idx_to_drop,:]
    # now drop subjects with missing gender/dx (turns out they overlap)
    # # http://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
    #df_merge = df_merge[df_merge['gender'].notnull()]
    #df_merge = df_merge[df_merge['dx'].notnull()]
    #return X, df_scores, df_file_id,df_merge
    #%% create a column of label indicating gender and disease status
    # disease: +1 = ASD, -1 = TDC
    idx_TDC = np.array([_ == 'TDC' for _ in df_merge['dx']])
    idx_ASD = np.array([_ == 'ASD' for _ in df_merge['dx']])
    y_dx = np.zeros(df_merge.shape[0],dtype=int)
    y_dx[idx_ASD] = +1
    y_dx[idx_TDC] = -1

    # gender: +1 = male, -1 = female
    idx_male = np.array([_ == 'M' for _ in df_merge['gender']])
    idx_fema = np.array([_ == 'F' for _ in df_merge['gender']])
    y_gender = np.zeros(df_merge.shape[0],dtype=int)
    y_gender[idx_male] = +1
    y_gender[idx_fema] = -1

    df_merge['y_dx']     = y_dx
    df_merge['y_gender'] = y_gender
    #%% reorder columns to my taste
    cols = df_merge.columns.tolist()

    # columns I'd like to show up front
    cols_front = ['ID','dx','y_dx','gender','y_gender',
                  'age','file_name','index_data_original']

    # remove the list items from above list using pop.
    for colname in cols_front:
        idx = cols.index(colname)
        cols.pop(idx)

    # append the remaining columns at the tails of the list
    cols = cols_front + cols

    # apply reordering
    df_merge = df_merge[cols]

    #%%
    if return_all_scores:
        pass
    else:
        # select only the basic scores that i typically need
        df_merge = df_merge[cols_front]

    # reset index so that i get nice [0,1,...,n-1] without weird jumps in the index
    df_merge.reset_index(drop=True,inplace=True)
    #%% some scans on disk has NAN for 'gender' and 'DX'...drop them
    # select only the data where 'dx' and 'gender' is available
    if return_as_design:
        X = X[df_merge['index_data_original']]
    else:
        X = X[:,:,df_merge['index_data_original']]
    #%%=== finally done; return results ===
    # the ``.values`` attribute critical here; sending a Pandas Series to classifiers screws things up
    y = df_merge['y_dx'].values
    if return_dropped:
        return X, y, df_merge, df_dropped
    else:
        return X, y, df_merge


def get_tob_invariants_batch(sort_by_id=True, return_dropped=False):
    """ Return the content of the tobacco HARDI invariant as dataFrame as a single DataFrame

    **11/05/2015** Data i/o updated after migrating to new git repository.

    Parameters
    ----------
    sort_by_id : bool (default: True)
        sort index/row of dataframe by subject-id
    return_dropped : bool (default: False)
        return DataFrame of the dropped indices

    Notes
    ------
    - some subjects have multiple scans

    Example

    .. code:: python

        R0023_V0055
        R0023_V1142
        R0025
        R0025_V0389
        R0031_V0070
        R0031_V1122

    For these scans, I'll take the first occurence of the duplicates
    (so in the above example, ``R0023_V0055``, ``R0025``, and ``R0031_V0070``
    get selected)

    Usage
    ------
    >>> df = get_tob_invariants_batch()

    >>> df, df_dropped = get_tob_invariants_batch(return_dropped=True)

    """
    df = pd.read_csv(filepath_tob_hardi())
    # removing the first 10 chars "invariants_" since it's redundant info
    #df.columns[4:] = [_[10:] for _ in df.columns[4:]] # <-oops, this won't work
    #^^^^^^^^^^^^ raises TypeError: Indexes does not support mutable operation

    # so i have to rename the entire thing at once...just append the column-names
    # that i don't want to modify
    df.columns = list(df.columns[:4])+[_[10:] for _ in df.columns[4:]]

    # sort by subject id
    df.sort(columns=['Subject'],inplace=True)

    # create a column of subject ID (the first 5 characters in the "Subject" col)
    df['ID'] = df['Subject'].str[:5]

    # bring this "ID" column into the front
    # http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # remove subjects with seemingly multiple scans
    # examples: R0023_V0055 R0025
    #           R0023_V1142 R0025_V0389
    # (here, take the first occurence)
    idx_drop = df.duplicated(subset='ID', take_last=False)
    df_dropped = df.ix[idx_drop,:]
    df = df.drop_duplicates(subset='ID', take_last=False)

    # reset index
    df.reset_index(inplace=True)
    # rename the original index
    df.rename(columns = {'index':'original_index_in_spreadsheet'},inplace=True)

    # reorder column to my flavor
    cols = df.columns.tolist()
    #cols = [cols[1]]+[cols[2]]+[cols[0]]+cols[3:]
    cols = cols[1:6]+[cols[0]]+cols[6:]
    df = df[cols]

    if return_dropped:
        return df, df_dropped
    else:
        return df

def get_tob_invariants():
    """ Return HARDI invariant data, where invariant measures are returned as lists

    Returns
    -------
    y : ndarray [nsubj]
        Label vector (+1 = ASD, -1 = TDC)
    df_meta : pandas DataFrame of shape [n,6]
        contains: ``(ID, Subject, Sex, Age, DX, original index in DataFrame)``
    HARDI_mean_list : list of length 12
        Each list elements contains ndarray of shape [n,176] representing mean invariants values
    HARDI_std_list : list of length 12
        Each list elements contains ndarray of shape [n,176] representing std-dev invariants values

    Usage
    -----
    >>> y, df_meta, HARDI_mean_list, HARDI_std_list = get_tob_invariants()

    To create a huge feature matrix, use ``np.hstack``

    >>> # This create an (n x 176*12) feature matrix of all HARDI mean invariants
    >>> X_nzmean = np.hstack(HARDI_mean_list)
    >>> X_stddev = np.hstack(HARDI_std_list)

    >>> # This create a (n x 176*24) feature matrix of all HARDI mean&std invariants (so all info possible)
    >>> X_all = np.hstack(HARDI_mean_list+HARDI_std_list)
    """
    df_all = get_tob_invariants_batch(return_dropped=False)
    #%%======== clear out "meta_info" ============
    # meta_info (ID, Subject, Sex, Age, DX, original index in spreadsheet)
    df_meta = df_all.ix[:,:6]
    #df_data = df_all.ix[:,6:]
    #%% create a column of label indicating gender and disease status
    # disease: +1 = ASD, -1 = TDC
    idx_TDC = np.array([_ == 'TDC' for _ in df_meta['DX']])
    idx_ASD = np.array([_ == 'ASD' for _ in df_meta['DX']])
    y_dx = np.zeros(df_meta.shape[0],dtype=int)
    y_dx[idx_ASD] = +1
    y_dx[idx_TDC] = -1

    # gender: +1 = male, -1 = female
    idx_male = np.array([_ == 'M' for _ in df_meta['Sex']])
    idx_fema = np.array([_ == 'F' for _ in df_meta['Sex']])
    y_gender = np.zeros(df_meta.shape[0],dtype=int)
    y_gender[idx_male] = +1
    y_gender[idx_fema] = -1

    df_meta['y_dx']     = y_dx
    df_meta['y_gender'] = y_gender

    # reorder columns
    df_meta = df_meta[['ID','Subject','DX','y_dx','Sex','y_gender','Age','original_index_in_spreadsheet']]
    #%%======== sort out HARDI feature values ============
    # instead of dealing with dataFrames, just deal with ndarrays
    # (fix the sluggishness encountered when i slice through DF's)
    hardi_invariants = df_all.ix[:,6:].values

    nROI = 176
    n_invariants = 12
    idx_range = np.arange(nROI,dtype=int)
    HARDI_mean_list = []
    HARDI_std_list  = []
    for i in range(n_invariants):
        # extract mean invariants
        HARDI_mean_list.append(hardi_invariants[:,idx_range])

        # translate indices to next invariant measures
        idx_range += nROI

        # extract std-dev
        HARDI_std_list.append(hardi_invariants[:,idx_range])

        # translate indices to next invariant measures
        idx_range += nROI

    return y_dx, df_meta, HARDI_mean_list, HARDI_std_list

def get_tob_invariants_dropped():
    """ Return DataFrame of dropped rows in  ``get_tob_invariants_batch``

    **note**: I dropped duplicate scans from same subject, taking the first scan in the DataFrame

    See ``get_tob_invariants_batch``
    """
    _, df_dropped = get_tob_invariants_batch(return_dropped=True)

    return df_dropped

def get_176roi_labels():
    """ Get the abbreviated labels for the 176 rois

    """
    #%% old crap
#    fpath = filepath_tob_hardi()
#    df_data = pd.read_csv(fpath, nrows=1,header=None).T
#    labels = [str_[20:] for str_ in df_data[0].tolist()
#                if str_.startswith('Invariant_00_nzmean_')]
    #%%
    df = pd.read_csv(filepath_176roi_labels())

    # i only care about these colums
    df = df.ix[:,1:4]

    # rename columes
    df.rename(columns={'__Volume':'volume',
               '__Description':'name',
               '__Short':'short'}, inplace=True)
    # i like float for this...
    df['volume'] = df['volume'].astype(float)
    return df
#%% --- utility functions for our analysis ----
def util_tobacco(modality, hardi_index=None, male_only=False):
    """ Util function for ASD vs TDC analysis with the Tobacco project.

    **Created 11/11/2015**

    Parameters
    ----------
    modality : string
        Choices:

        - ``conn``: PROB connectomes
        - ``hardi_mean``: HARDI mean invariant.  There are 12 choices
          (indexed from [0-11]) you can query via ``hardi_index`` parameter.
        - ``hardi_std``: HARDI std-deviation invariant.  There are 12 choices
          (indexed from [0-11]) you can query via ``hardi_index`` parameter.
        - ``hardi_both``: combination of HARDI mean and std-dev invariants.
          You can specify which invariants to use via supplying a 2-d list to
          the ``hardi_index`` parameter
        - ``hardi_mean_all``: concatenate all the 12 HARDI-mean invariants to
          obtain feature of size **176*12=2112**
        - ``hardi_std_all``: concatenate all the 12 HARDI-std-dev invariants to
          obtain feature of size **176*12=2112**
        - ``hardi_all``: concatenate all the 12 HARDI-mean invariants and
          12 HARDI-std-dev invariants to to obtain feature of size
          **176*12 + 176*12=4224**
    hardi_index : list of HARDI indices (irrelevant for ``modality=conn``)
        List of indices querying which HARDI invariant to use.
        Only relevant if **modality** = ``hardi_mean``, ``hardi_std``, or ``hardi_both``

        - If **modality** = ``hardi_mean`` or ``hardi_std``, **hardi_index**
          will be a 1-D list giving indices beween [0-11], corresponding to
          the 12 HARDI mean/std invariants
        - If **modality** = ``hardi_both``, **hardi_index** shall be a 2-d
          list, where the first dimension is for querying for HARDI-mean
          invariant, and the second dim for querying HARDI-std invariant.

    male_only : bool (default=False)
        Option to restrict analysis to male.


    Returns
    --------
    **X, y, df**

    Example
    -------
    Extract prob-tract connectome, restricting to male only

    >>> X, y, df = twio.util_tobacco(modality='conn',male_only=True)

    Extract HARDI-nzmean0 invariant

    >>> X, y, df = twio.util_tobacco(modality='hardi_mean', hardi_index = 0)

    Extract and concatenate HARDI-nzstd{0,5,11} to create feature of size 176*3

    >>> X, y, df = twio.util_tobacco(modality='hardi_std', hardi_index = [0,5,11])

    Extract and concatenate HARDI-nzmean{0,1,5,11} and HARDI-std{1,9} to create
    a feature matrix of size **4*176+2*176 = 1056**.  Note the use of a 2d list here.

    >>> hardi_index = [ [0,1,5, 11], [1, 9] ]
    >>> X, y, df = twio.util_tobacco(modality='hardi_both', hardi_index = hardi_index)

    Extract all 12 HARDI-mean invariant to create feature of size 176*12, and also
    restrict data to Male only

    >>> X, y, df = twio.util_tobacco(modality='hardi_mean_all',male_only=True)

    Extract all 12 HARDI-std invariant to create feature of size 176*12 = 2112,
    and also restrict data to Male only

    >>> X, y, df = twio.util_tobacco(modality='hardi_std_all',male_only=True)

    Extract all 12 HARDI-mean and 12 HARDI-std invariants to create feature of
    size 176*12 + 176*12 = 4224

    >>> X, y, df = twio.util_tobacco(modality='hardi_all')


    Protocode
    ---------
    ``__done__proto_tob_data_io_1111.py``

    Remarks about the clinical scores DataFrame for tobacco
    -----------------------
    ::

        Note: I couldn't do a reliable 'join' with the master-score
        DataFrame with the spreadsheet containing HARDI-invariants...the age
        information seem to be somewhat off, and the V_NUM (eg V1226) info
        is not present in the HARDI spreadsheet...thus I don't know how to join
        the two DataFrames.  Thus I have two separate DataFrames to work with
        for the tobacco project
    """
    if isinstance(hardi_index,str):
        hardi_index = int(hardi_index)

    if isinstance(hardi_index, int):
        # if a scalar is provided, convert to list, as my list comprehension
        # syntax below relies on this variable being iterable
        hardi_index = [hardi_index]
    if modality == 'conn':
        # 05/09/2016 - set return all scores to true
        X, y, df = get_tob_connectomes(return_all_scores=True)
    elif modality[:5].lower() == 'hardi':
        y, df, HARDI_mean_list, HARDI_std_list = get_tob_invariants()

        # for consistency with the DataFrame with the connectomes, rename "Sex" to "gender"
        """Note: I couldn't do a reliable 'join' with the master-score
           DataFrame with the spreadsheet containing HARDI-invariants...the age
           information seem to be somewhat off, and the V_NUM (eg V1226) info
           is not present in the HARDI spreadsheet...thus I don't know how to join
           the two DataFrames.  Thus I have two separate DataFrames to work with
           for the tobacco project"""
        df.rename(columns={'Sex':'gender'},inplace=True)

        """ Below constructs the design matrix X by concatenating 176ROI features
            of requested invariant numbers"""
        if modality == 'hardi_mean_all':
            # create (n x 176*12) feature matrix of all HARDI mean invariants
            X = np.hstack(HARDI_mean_list)
        elif modality == 'hardi_std_all':
            # create (n x 176*12) feature matrix of all HARDI stddev invariants
            X = np.hstack(HARDI_std_list)
        elif modality == 'hardi_all':
            # create (n x 176*24) feature matrix of all HARDI mean&std invariants
            X = np.hstack(HARDI_mean_list+HARDI_std_list)
        elif modality == 'hardi_mean':
            X = np.hstack([HARDI_mean_list[idx_] for idx_ in hardi_index])
        elif modality == 'hardi_std':
            X = np.hstack([HARDI_std_list[idx_] for idx_ in hardi_index])
        elif modality == 'hardi_both':
            query_mean = [HARDI_mean_list[idx_] for idx_ in hardi_index[:][0]]
            query_std  = [HARDI_std_list[idx_]  for idx_ in hardi_index[:][1]]
            X = np.hstack(query_mean + query_std)


    if male_only:
        df = tw.pd_prepend_index_column(df,'index_original')
        df = df.query('gender == "M"')
        y = y[df.index]
        X = X[df.index]
        df.reset_index(drop=True,inplace=True)

    return X, y, df

#%% ******** TEST BLOCK ********** __name__ == "__main__":
if __name__ == "__main__":
    pass
    #%% sanity check
    #df_scores1, age_bins1, idx_list1 = get_pnc_scores_age_qcut()
    #import pnc_data_io
    #df_scores2, age_bins2, idx_list2 = pnc_data_io.get_pnc_scores_age_qcut()
    #np.all((df_scores1 == df_scores2).values)
    #np.all(age_bins1 == age_bins2)
    #for i1, i2 in zip(idx_list1, idx_list2):
    #    print np.all(i1 == i2)
    #%% sanity check on get_pnc_scores() function
    #df_sc1, f1, m1 = get_pnc_scores()
    #import pnc_data_io
    #df_sc2, f2, m2 = pnc_data_io.get_pnc_scores2()
    #np.all((df_sc1 == df_sc2).values)
    #%%
    #X_tob = get_tob_connectomes()
    #X_pnc,y,file_names = get_pnc_connectome()
    #print tw.get_hostname()
    #print tw.get_hostname()
# -*- coding: utf-8 -*-
"""
===============================================================================
Maintain "quick and dirty" functions that I may find useful at the moment.
===============================================================================
Created on Fri Nov 13 14:48:37 2015

@author: takanori
"""
#% module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

import scipy as sp
import inspect
import time
from pprint import pprint

import tak as tw
import data_io as twio
import ml as twml
from sklearn.utils import deprecated
#%% === plotting utilities that have not been fully refined yet ====


def show_cv_gridsearch2d(acc_grid, param_grid, iouter=-101,clf_name='',showmax=True):
    """ Display outercv gridsearch result (creatd 10/31/2015)

    Details
    -------
    - Before launching my "official" run of the script, I like to get an idea
      on what parameter values are within reasonable values of search range
    - seeing ``flag_show_plot`` below will give me an plot of the couter-cv
      accuracy...
    - when content with the "search-range", set flag to False, and launch script

    Usecase
    -------
    See for example:
    ``~/tak-ace-ibis/python/analysis/tbi/nested_cv_conn/##try_outercv_tbi.py``
    """
    #plt.clf()
    tw.figure('f')
    plt.title(clf_name+' (CV={:2}, max={:.3f})'.format(iouter+1,acc_grid.max()))
    ylabel = param_grid.keys()[0]
    xlabel = param_grid.keys()[1]
    ytick_label = param_grid.values()[0]
    xtick_label = param_grid.values()[1]
#        imgridsearch(acc_grid,xtick_label=xtick_label,ytick_label=ytick_label,
#                        show_max=True,fontsize=14,show_cbar=True)
    tw.imgridsearch(acc_grid,xtick_label=xtick_label,ytick_label=ytick_label,
                    vmin=0.55, vmax=acc_grid.max(),show_max=True,fontsize=14,show_cbar=True)
    plt.gca().set_xlabel(xlabel)
    plt.gca().set_ylabel(ylabel)

    # need these for the plots to update
    plt.draw()
    plt.pause(0.5)
    #plt.waitforbuttonpress()


def imconnmat_subplot_fix_xticks(fsize=8,rot=50):
    node_info = tw.get_node_info86()
    axs = plt.gcf().axes[:3] # first 3 are subplots
    for n, ax in enumerate(axs):
        ax.set_xticklabels(node_info['name_short'], rotation=rot, ha='right',fontsize=fsize-1.5)
        if n == 0:
            ax.set_yticklabels(node_info['name_short'].values,fontsize=fsize)
        elif n == 1:
            ax.set_yticklabels(node_info['loabes'].values,fontsize=fsize)
        elif n==2:
            ax.yaxis.tick_right()
            ax.set_yticklabels(node_info['name_short'].values,fontsize=fsize)
#%% ==== for project specific stuffs ====
def get_data_1116(project, **setup):
    if project == 'pnc_gender_conn':
        X, y, df = twio.util_pnc(**setup)
    #=== TBI: conn ====#
    elif project == 'tbi_conn':
        X, y, df = twio.util_tbi(**setup)
    #=== IBIS gender ===#
    elif project == 'ibis_gender_conn':
        X, y, df = twio.util_ibis_gender(**setup)
    elif project == 'ibis_gender_conn_delta':
        X, y, df = twio.util_ibis_gender_delta(**setup)
    #elif project in ['ibis_gender_dvol']:
    #    X, y, df = twio.util_ibis_gender(**setup)
    #elif project in 'ibis_gender_dvol_delta':
    #    X, y, df = twio.util_ibis_gender_delta(**setup)
    #=== IBIS HR+ vs LR- ===#
    elif project == 'ibis_HRp_LRm_conn':
        X, y, df = twio.util_ibis_HRp_LRm(**setup)
    elif project == 'ibis_HRp_LRm_conn_delta':
        X, y, df = twio.util_ibis_HRp_LRm_delta(**setup)
    #elif project == 'ibis_HRp_LRm_dvol':
    #    X, y, df = twio.util_ibis_HRp_LRm(**setup)
    #elif project == 'ibis_HRp_LRm_dvol_delta':
    #    X, y, df = twio.util_ibis_HRp_LRm_delta(**setup)
    #==== Tobacco: conn =====
    elif project == 'tob_conn':
        X, y, df = twio.util_tobacco(**setup)
    #==== Tobacco: HARDI =====
    elif project == 'tob_HARDI':
        X, y, df = twio.util_tobacco(**setup)
    #==== Tobacco: HARDI all =====
    elif project == 'tob_HARDI_all':
        X, y, df = twio.util_tobacco(**setup)
    return X,y,df


def get_pklpath_ncv10_1115(project, setup, clf_name, flag_balance_data=False):
    """ Get path to pkl-files outputted from ``nested_10foldCV_batchsubmit_1115.py``

    Codes here is made by modifying parts from ``nested_10foldCV_batchsubmit_1115.py``

    After running on SGE on the server, I moved the results to
    ``/home/takanori/data/<proj>/results``, where ``<proj>`` = **<ibis|pnc|tbi|tob>**

    Parameters
    -------
    **project** : string

    - pnc_gender_conn
    - tbi_conn
    - tob_conn
    - tob_HARDI
    - tob_HARDI_all
    - ibis_gender_conn
    - ibis_gender_conn_delta
    - ibis_HRp_LRm_conn
    - ibis_HRp_LRm_conn_delta

    **setup** : dict

    The key-value of the dict here depends on choice the ``project``

    - ``pnc_gender_conn``
        - age_group = 'q1', 'q2', 'q3'
    - ``tbi_conn``
        - scan = 1, 2, 3
    - ``tob_conn``
        - modality = 'conn'
        - male_only = <True | False>
    - ``tob_HARDI``
        - modality = <'hardi_mean' | 'hardi_std'>
        - hardi_index = list (or single int) of intger containing values from 0...11
        - male_only = <True | False>
    - ``tob_HARDI_all``
        - modality = <'hardi_mean_all' | 'hardi_std_all' | 'hardi_all'>
        - male_only = <True | False>

    **clf_name** : string

    - clf_name = ``'sklLogregL1'``
    - clf_name = ``'sklLinSvm'``
    - clf_name = ``'rbfSvm'``
    - clf_name = ``'ttestRbfSvm' # skip for hardi``
    - clf_name = ``'ttestLinSvm'``
    - clf_name = ``'enetLogRegSpams'``
    - clf_name = ``'enetLogRegGlmNet'``
    - clf_name = ``'PcaLda'``
    - clf_name = ``'PcaLinSvm'``
    - clf_name = ``'PcaRbfSvm'# <- skip for hardi``
    - clf_name = ``'ttestLDA'``

    **flag_balance_data** : bool (default=False)
    """
    datadir = '/home/takanori/data'

    proj     = project.split('_',1)[0] # proj = (ibis|pnc|tbi|tob)
    analysis = project.split('_',1)[1] # eg, "HRp_LRm_conn_delta"

    outfolder = os.path.join(datadir, proj, "results","random_nested10cv",analysis)

    if flag_balance_data:
        outfolder = os.path.join(outfolder, 'balanced_labels')

    #=== parse <setup> ===#
    setup_folder = ''
    for key in sorted(setup.keys()):
        if key in ['male_only','all_sessions']:
            # thesee are boolean variables.  If True, create folder for it
            if setup[key] == True:
                # create a directory for these
                outfolder = os.path.join(outfolder,key)
        else:
            setup_folder += '_'+str(setup[key])

    # remove underscore at very front
    setup_folder = setup_folder[1:]

    outfolder = os.path.join(outfolder,setup_folder,clf_name)
    return outfolder
    #=== glob list of pkl outfiles ====
#    glob('*.pkl')
#    outfile   = 'random_state'+str(random_state)+'.pkl'

#    outfile_fullpath = os.path.join(outfolder, outfile)

#    return outfile_fullpath
#%% ==== fista related stuffs ====

#%% everything in this block is from ``nilearn.decoding``
from scipy import linalg




""" Below will "blow-up" to infinity....use nilearn's version
"""
@deprecated('this version may blow up to infinity.  use nilearn\'s version')
def logistic_function_tw(X,y,w):
#    yX=(X.T*y).T # same as np.diag(y).dot(X), but faster
#    yXw = yX.dot(w)

    yXw = y*(X.dot(w))
    def logis(t):
        return np.log(1+np.exp(-t))

    return np.sum(logis(yXw))

def logistic_grad_tw(X,y,w):
    """Created this just as a sanity check that nilearn's funcion is doing
    what i think it's doing.  I'll use nilearn's version for real coding"""
    yXw = y*(X.dot(w))

    def sigm(t):
        return 1./(1+np.exp(-t))

    return -(X.T.dot(y*sigm(-yXw)))


def add_intercept_X(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import inspect
from tak.tak import set_trace,debug
import tak.tak as tw
import os
import sys
from pprint import pprint

if __name__ == "__main__":
    filepath = '/home/takanori/work-local/tak-ace-ibis/data_local/86'
    filename = 'tak_CoM_master.xlsx'
    df = pd.read_excel(os.path.join(filepath,filename),header=None)
    
    # load xyz coord from another sheet
    df_coord = pd.read_excel(os.path.join(filepath,filename),header=None,
                             sheetname='coords_unrotated',names=['x','y','z'])
    
    #%% drop 1st and last column
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(df.columns[-1], axis=1, inplace=True)
    
    df2=df.rename(columns={1:'label', 2:'name_full', 3:'lobes', 4:'name_short',
                           5:'system'})
    
    #%% the cerebellum has some wonky things going on ... apply adhoc fix
    # add L/R info on these items for consistency
    df2.ix[34,'lobes'] = 'L '+df2.ix[34,'lobes']
    df2.ix[77,'lobes'] = 'R '+df2.ix[77,'lobes']
    
    # "cerebellum" set to NaN....annoying so set to "other"
    df2.ix[[34,77],'system'] = 'other'
    
    # concatenate coord info
    df3 = pd.concat([df2,df_coord], axis=1)
    
    # add column of hemisphere info (can be handy)
    df3['hemisphere'] = ['L']*43+['R']*43
    #%% read-in mni coordinates
    filepath='/home/takanori/work-local/tak-ace-ibis/python/data_post_oct2015/'
    filename='coords_desikan_86_mni.csv'
    df_mni = pd.read_csv(os.path.join(filepath,filename),header=None,
                         names=['xmni','ymni','zmni'])
    df_out = pd.concat([df3,df_mni],axis=1)
    #%% write to disk
    #http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
    outpath='/home/takanori/work-local/tak-ace-ibis/python/data_post_oct2015/'
    df_out.to_csv(outpath+'tw_node_info_86.csv')
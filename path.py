# -*- coding: utf-8 -*-
"""
===============================================================================
Utility functions related to "path" and "directory"

Also decided to include stuffs relating to "time"
-------------------------------------------------------------------------------
===============================================================================
Created on Tue Apr 12 11:09:09 2016

@author: takanori
"""
#% module
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import sklearn
#import scipy as sp

import os
import sys
import inspect
import time

#%%
from socket import gethostname
hostname = gethostname()
#%% === post 04/12/2016 functions ===
def get_project_root():
    """ Get directory ~/sbia_work/python

    Created 04/12/2016
    """
    root = os.path.expanduser('~')
    if hostname == 'sbia-pc125':
        return os.path.join(root,'Dropbox','work','sbia_work','python')
    elif hostname in ['takanori-PC','tak-sp3']:
        return os.path.join(root,'Dropbox','work','sbia_work','python')
    else:
        # on the computer cluster
        return os.path.join(root,'sbia_work','python')

#%% === migrated from core.py (4/12/2016)...may remove some in near future  ===
def gethost():
    """Old script migrated from core.py (04/12/2016)"""
    import subprocess
    return subprocess.check_output('hostname')[:-1] # [:-1] to remove '\n'

def get_hostname():
    """Created 04/12/2016"""
    return gethostname()

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


def homedir():
    """ Get home directory ``"~"``

    Usage
    -----
    >>> tw.homedir()
    Out[5]: '/home/takanori

    Source code
    ------------
    .. code :: python

        from os.path import expanduser
        home = expanduser("~")
        return home

    Ref
    -----
    http://stackoverflow.com/questions/4028904/how-to-get-the-home-directory-in-python
    """
    from os.path import expanduser
    home = expanduser("~")
    return home

def filepath():
    """ Return full path of current function

    Output
    -------
    filepath : string
        full filepath of the caller script

    Usage
    -----
    >>> tw.filepath()
    /home/takanori/work-local/tak-ace-ibis/python/analysis/__proto/proto_3d_slice_viewer_1028.py

    Use ``os.path.split()`` to get directory name and filename separately

    >>> fpath, fname = os.path.split(tw.filepath())

    See also
    --------
    ``filename``

    Ref
    ---
    - http://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python
    - http://stackoverflow.com/questions/3711184/how-to-use-inspect-to-get-the-callers-info-from-callee-in-python
    """
    import inspect

    frame,fullpath,line_number,function_name,lines,index = inspect.stack()[1]
    #module = inspect.getmodule(frame[0])

    #directory, filename = os.path.split(fullpath)
    #return os.path.basename(__file__)
    return fullpath


def filename():
    """ Return the filenname of current function

    Output
    -------
    filename : string
        filename of the caller script

    Usage
    -----
    >>> tw.filename()
    proto_3d_slice_viewer_1028.py

    See also
    --------
    ``filename``

    Ref
    ---
    - http://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python
    - http://stackoverflow.com/questions/3711184/how-to-use-inspect-to-get-the-callers-info-from-callee-in-python
    """
    import inspect
    import os
    frame,fullpath,line_number,function_name,lines,index = inspect.stack()[1]
    #module = inspect.getmodule(frame[0])
    directory, filename = os.path.split(fullpath)
    #return os.path.basename(__file__)
    return filename



def path_get_dir_pyname():
    """Create directory with the same script name

    http://stackoverflow.com/questions/16305867/determine-where-a-function-was-executed
    """
    caller_frame = inspect.stack()[1]
    pydir, pyname = os.path.split(caller_frame[0].f_globals.get('__file__', None))

    # remove ".py" extension
    pyname = pyname[:-3]
    folder_path = os.path.join(pydir,pyname)
    print folder_path

    # create a folder with the python script name
    if not os.path.exists(folder_path):
        message = 'The following path does not exit: {}\n'.format(folder_path)
        message += '\nCreate directory? [Enter "y" for yes]'
        prompt = raw_input(message)
        if prompt == 'y':
            os.makedirs(folder_path)
        else:
            sys.exit('Exiting code...')

    return folder_path

def path_get_current_script_path():
    """Get fullpath to the python script calling this function

    http://stackoverflow.com/questions/16305867/determine-where-a-function-was-executed
    """
    caller_frame = inspect.stack()[1]
    return caller_frame[0].f_globals.get('__file__', None)


#%% -- time related stuffs --
def get_timestamp():
    """ Get time stamp

    Usage
    ------
    >>> tw.get_timestamp()
    Out[32]: '2015-10-27 14:21:22

    """
    from datetime import datetime
    timeStamp = str(datetime.now())[0:19]
    return timeStamp

def timestamp():
    """ Get time stamp

    Usage
    ------
    >>> tw.get_timestamp()
    Out[32]: '2015-10-27 14:21:22

    """
    from datetime import datetime
    timeStamp = str(datetime.now())[0:19]
    return timeStamp

def print_time_dec(func):
    """ Used as decorator to evaluate runtime

    For help on decorator, see http://stackoverflow.com/questions/739654/how-can-i-make-a-chain-of-function-decorators-in-python

    Example
    --------
    >>> @print_time
    >>> def looptest(A):
    >>>     for i in xrange(10):
    >>>         np.dot(A,A)


    Reminder on how decorator with any arguments
    ---------------------------------------------
    >>> def a_decorator_passing_arbitrary_arguments(function_to_decorate):
    >>>     # The wrapper accepts any arguments
    >>>     def a_wrapper_accepting_arbitrary_arguments(*args, **kwargs):
    >>>         print "Do I have args?:"
    >>>         print args
    >>>         print kwargs
    >>>         # Then you unpack the arguments, here *args, **kwargs
    >>>         # If you are not familiar with unpacking, check:
    >>>         # http://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
    >>>         function_to_decorate(*args, **kwargs)
    >>>     return a_wrapper_accepting_arbitrary_arguments
    """

    def wrapper(*args, **kwargs):
        t=time.time()
        func(*args, **kwargs)
        print "Elapsed time: {:>5.2f} seconds".format(time.time()-t)
    return wrapper


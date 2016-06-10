# -*- coding: utf-8 -*-
from socket import gethostname
hostname = gethostname()

def fix_file_list(fileList):
    """Convert fileList, which is in recarray form, to list

    The awkward recarray form occurs when reading array of strings from
    *.mat file

    05/23/2016
    """
    file_list = []
    for i in xrange(len(fileList)):
        file_name = str(fileList[i][0][0])
        file_list.append(file_name)
    return file_list
    
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
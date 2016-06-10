


#%% ***** BELOW IS OLD STUFFS!!!! DELETE IF I THINK THEY"RE NO LONGER NEEDED***
#%% ==== define functions ======
def tw_dashed_message(message):
    dash = "#" + '=' *78 + '#'
    print '\n\n'+dash +'\n'+ message + '\n' + dash

def tw_get_timestamp():
    from datetime import datetime
    timeStamp = str(datetime.now())
    return timeStamp

def tw_gethost():
    import subprocess
    return subprocess.check_output('hostname')[:-1] # [:-1] to remove '\n' at the end
#%% ======= directory related stuffs ==============
def tw_homedir():
    """ Get home directory "~"

    Ref: http://stackoverflow.com/questions/4028904/how-to-get-the-home-directory-in-python
    """
    from os.path import expanduser
    home = expanduser("~")
    return home

def tw_filepath():
    """
    http://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python
    http://stackoverflow.com/questions/3711184/how-to-use-inspect-to-get-the-callers-info-from-callee-in-python
    """
    import inspect,os
    frame,fullpath,line_number,function_name,lines,index = inspect.stack()[1]
#    module = inspect.getmodule(frame[0])
#    directory, filename = os.path.split(fullpath)
    return fullpath
#    return os.path.basename(__file__)

def tw_filename():
    """
    http://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python
    http://stackoverflow.com/questions/3711184/how-to-use-inspect-to-get-the-callers-info-from-callee-in-python
    """
    import inspect,os
    frame,fullpath,line_number,function_name,lines,index = inspect.stack()[1]
#    module = inspect.getmodule(frame[0])
    directory, filename = os.path.split(fullpath)
    return filename
#    return os.path.basename(__file__)

if __name__ == "__main__":
    pass
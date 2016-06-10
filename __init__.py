"""
===============================================================================
The import/reload below is often frowned upon, but most of my work involves
interactive module design, as my research involves heavy prototype coding.
-------------------------------------------------------------------------------
 http://stackoverflow.com/questions/1739924/python-reload-component-y-imported-with-from-x-import-y
 http://stackoverflow.com/questions/5516783/how-to-reload-python-module-imported-using-from-module-import
===============================================================================
"""
#%% load subpackages
import tak.data
import tak.ml

reload(tak.data)
reload(tak.ml)

#from .data import *
#from .ml import *
#%% load submodules
import tak.conn
import tak.core
import tak.gt
import tak.np_
import tak.path
import tak.pd_
import tak.plt_
import tak.stats

reload(tak.conn)
reload(tak.core)
reload(tak.gt)

reload(tak.np_)
reload(tak.path)
reload(tak.pd_)
reload(tak.plt_)
reload(tak.stats)

from .conn import *
from .core import *
from .gt import *

from .np_ import *
from .path import *
from .pd_ import *
from .plt_ import *
from .stats import *

from inspect import getsourcefile # <- to get where a function was defined
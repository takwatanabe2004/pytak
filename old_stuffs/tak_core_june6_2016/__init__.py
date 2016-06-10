"""
===============================================================================
The import/reload below is often frowned upon, but most of my work invovles
interctive module design, as my research involves heavy prototype coding.
-------------------------------------------------------------------------------
 http://stackoverflow.com/questions/1739924/python-reload-component-y-imported-with-from-x-import-y
 http://stackoverflow.com/questions/5516783/how-to-reload-python-module-imported-using-from-module-import
===============================================================================
"""
import tak.core
import tak.conn
import tak.plot
import tak.pd
import tak.tw_cross_val
import tak.path
import tak.gt

reload(tak.core)
reload(tak.conn)
reload(tak.plot)
reload(tak.pd)
reload(tak.path)
reload(tak.gt)
reload(tak.tw_cross_val)

from .core import *
from .conn import *
from .plot import *
from .pd import *
from .path import *
from .tw_cross_val import *
from .gt import *

from inspect import getsourcefile # <- to get where a function was defined
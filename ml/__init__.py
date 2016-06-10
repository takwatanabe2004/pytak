"""
===============================================================================
This submodule contains my machine-learning related functionalities
===============================================================================
Created on Tue Jun  7 21:28:58 2016

@author: takanori
"""

from . import core_ml as core
from . import util_ml as util
from . import nmf

#from tak.ml import core_ml as core
#from tak.ml import util_ml as util
#from tak.ml import nmf

reload(core)
reload(util)
reload(nmf)


from .core_ml import *
from .util_ml import *
#from .nmf import *

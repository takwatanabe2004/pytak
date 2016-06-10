"""
===============================================================================
The import/reload below is often frowned upon, but most of my work invovles
interctive module design, as my research involves heavy prototype coding.
-------------------------------------------------------------------------------
 http://stackoverflow.com/questions/1739924/python-reload-component-y-imported-with-from-x-import-y
 http://stackoverflow.com/questions/5516783/how-to-reload-python-module-imported-using-from-module-import
===============================================================================
"""
from . import core_nnet as core
from . import proto
from . import util_nnet as util
#from tak.nnet import core_nnet
#from tak.nnet import proto
#from tak.nnet import util_nnet
reload(core)
reload(proto)
reload(util)

#import tak.nnet.util
#reload(tak.nnet.util)
#import tak.nnet.util

#reload(tak.nnet)
#reload(tak.nnet.util)

from .core_nnet import *
#from .nnet_proto import *
from .util_nnet import *

#from . import nnet

#__all__ = ['util','nnet']
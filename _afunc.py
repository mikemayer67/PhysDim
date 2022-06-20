"""Numpy function overrides for PhysDim.Array

A handful of numpy functions invoke univeral functions under the covers.
For some (all?) of these cases, it is necessary to cast PhysDim.Array
inputs to numpy.ndarray and numpy.ndarray outputs back to PhyDim.Array.
"""

import numpy as np

from ._funcs import output_pdim
from ._funcs import same_pdim
from ._funcs import assert_same_pdim

from .exceptions import IncompatibleDimensions

io_mapping = {
    'amin' : 'io_map_minmax',
    'amax' : 'io_map_minmax',
    }


def io_map_minmax(func,obj,args):
    args = args[0]
    for arg in args:
        print(arg)
        import pdb; pdb.set_trace()

        if not same_pdim(obj,arg):
            raise IncompatibleDimensions(obj,arg)
    return output_pdim(obj.pdim)


def io_map(func,obj,args):
    fname = func.__name__
    io_map_name = io_mapping.get(fname,f"io_map_{fname}")
    io_map_func = globals().get(io_map_name,None)
    return io_map_func

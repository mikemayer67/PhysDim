"""Physical Dimension Array

This is a subclass of a numpy ndarray that adds a physical
dimensionality to the array.

Note that there are no traditional units attached to a given
instance of PhysDim.Array.  That is introduced through
PhysDim.Unit.

The dimensionality is added to the underlying ndarray via the
pdim attribute (which is an instance of PhysDim.Dim)
"""

import numpy as np
from numbers import Number

from . import _ufunc
from . import Dim

class Array(np.ndarray):
    __slot__ = ('pdim')

    def __new__(cls, *args, **kwargs):
        parsed_args = _parse_args(args,kwargs)
        if kwargs:
            raise TypeError(" ".join(( 
                "Unrecognized inputs to Array constructor:",
                ", ".join(kwargs.keys()))))
        if 'ref' in parsed_args:
            obj = parsed_args['ref']
            if 'scale' in parsed_args:
                obj = parsed_args['scale'] * obj
                # if ref is dimensionless, scaling will return a np.ndarray
                if type(obj) is not Array:
                    obj = obj.view(Array)
                    obj.pdim = Dim()
            else:
                obj = parsed_args['ref']

        else:
            if 'array' in parsed_args:
                obj = parsed_args['array']
            elif 'values' in parsed_args:
                obj = np.array(parsed_args['values'])
            elif 'scale' in parsed_args:
                obj = np.array(parsed_args['scale'])
            else:
                obj = np.array(1)
            obj = obj.view(Array)
            obj.pdim = parsed_args['pdim'] if 'pdim' in parsed_args else Dim()
        return obj

    def __array_finalize__(self,obj):
        if obj is None: return
        if not issubclass(self.dtype.type, np.number):
            raise TypeError(f"Cannot create Array of {self.dtype}")
        self.pdim = getattr(obj,'pdim',None)

    @property 
    def pdim_string(self):
        return str(self.pdim)

    def __repr__(self):
        value = super().__repr__()
        pdim = self.pdim.__repr__()
        return f"{value[:-1]}, pdim={pdim})"

    # @@@TODO: change pdim string to units once unit class is defined
    def __str__(self):
        value = super().__str__()
        pdim = self.pdim.__str__()
        return f"{value} ({pdim})"

    @property
    def is_angle(self):
        return self.pdim.is_angle

    @property
    def is_dimensionless(self):
        return self.pdim.is_dimensionless


    def __array_ufunc__(self,ufunc,method,*args,out=None,**kwargs):
        pdim = _ufunc.io_map(ufunc,self,args)

        # down-convert any Array inputs to numpy.ndarray 
        in_args = [ 
            arg.view(np.ndarray) if isinstance(arg,Array) else arg
            for arg in args
            ]

        # down-convert any user specified Array outputs to numpy.ndarray
        if out is not None:
            kwargs['out'] = tuple(
                o.view(np.ndarray) if isinstance(o,Array) else o
                for o in out 
                )

        # invoke the ufunc without Array arguments
        results = super().__array_ufunc__(ufunc, method, *in_args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        # If (all) pdim is dimensionless no convertion necessary
        if pdim is None:
            return results

        if ufunc.nout == 1:
            return _convert_result(results, pdim, out=out)
        else:
            if out is None:
                out = (None,)*ufunc.nout
            return tuple(
                _convert_result(r,p,out=o)
                for r,p,o in zip(results, pdim, out)
                )

# Support functions

def _convert_result(result, pdim, *, out=None): 
    if pdim:
        if type(out) is Array:
            out.pdim = pdim
            return out
        elif isinstance(result,np.ndarray):
            r = result.view(Array)
            r.pdim = pdim
            return r
        elif isinstance(result,np.number):
            return Array([result],pdim=pdim)
        else:
            return Array(np.asarray(result),pdim=pdim)
    else:
        if type(out) is Array:
            return out.view(np.ndarray)
        else:
            return result

_exclusive_constructor_args = { 
    ('ref','array'),('ref','values'),('ref','pdim'),
    ('array','values'),('array','scale'),
    ('scale','values'), }

def _add_arg(args, arg, key):
    if key in args:
        raise TypeError(f"Can only specify one {key} argument")
    for k in args.keys():
        if (key,k) in _exclusive_constructor_args or (k,key) in _exclusive_constructor_args:
            raise TypeError(f"Cannot specify both {key} and {k} arguments")
    args[key] = arg

def _parse_args(args,kwargs):
    rval = {}
    if 'pdim' in kwargs:
        rval['pdim'] = kwargs['pdim']
        del kwargs['pdim']
    for arg in args:
        if isinstance(arg,Array):
            _add_arg(rval,arg,'ref')
        elif isinstance(arg,np.ndarray):
            if arg.size == 1:
                _add_arg(rval,arg.flat[0],'scale')
            else:
                _add_arg(rval,arg,'array')
        elif isinstance(arg,Dim):
            _add_arg(rval,arg,'pdim')
        elif isinstance(arg,Number):
            _add_arg(rval,arg,'scale')
        else:
            _add_arg(rval,arg,'values')
    return rval


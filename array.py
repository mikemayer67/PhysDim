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

from . import _afunc
from . import _ufunc

class Array(np.ndarray):
    __slot__ = ('pdim')

    def __new__(cls, array=None, *, shape=None, pdim, **kwargs):
        if array is not None:
            if shape is not None:
                raise TypeError(f"Cannot initialize {cls} with both shape and array")
            try:
                obj = np.asarray(array).view(cls)
            except Exception as e:
                raise TypeError(f"The input array must be a simple array of numbers:\n {e}")
        else:
            shape = shape if shape is not None else 1
            obj = super().__new__(cls, shape, **kwargs)

        obj.pdim = pdim
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

    def fmin(*args,**kwargs):
        import pdb; pdb.set_trace()
        return super().fmin(*args,**kwargs)

    def __array_function__(self, func, types, *args, **kwargs):
        print(f"{func}: {type} {args} {kwargs}")
        import pdb; pdb.set_trace()

        io_map = _afunc.io_map(func,self,args)

        if io_map is not None:
            pdim = io_map(func,self,args)

        results = super().__array_function__(func,types,*args,**kwargs)

        return results


    def __array_ufunc__(self,ufunc,method,*args,out=None,**kwargs):
        print(f"{ufunc}:: {args}")

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
            return self._convert_result(results, pdim, out=out)
        else:
            if out is None:
                out = (None,)*ufunc.nout
            return tuple(
                self._convert_result(r,p,out=o)
                for r,p,o in zip(results, pdim, out)
                )

    def _convert_result(self, result, pdim, *, out=None): 
        if type(out) is Array:
            if pdim:
                out.pdim = pdim
                return out
            else:
                return out.view(np.ndarray)
        elif pdim:
            r = result.view(Array)
            r.pdim = pdim
            return r
        else:
            return result

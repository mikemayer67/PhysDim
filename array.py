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

from .dim import Dim
from .exceptions import IncompatibleDimensions

# List of numpy ufuncs was found here:
#   https://numpy.org/devdocs/reference/ufuncs.html

_TRIG_UFUNC = ('sin','cos','tan')
_COMPARISON_UFUNC = ('less','less_equal','equal','not_equal','greater','greater_equal')
_UNARY_UFUNC = ('negative','positive','absolute','invert')

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

    def same_pdim(self,other):
        try:
            return self.pdim == other.pdim
        except:
            return False

    def assert_same_pdim(self,other):
        if not self.same_pdim(other):
            raise IncompatibleDimensions(self,other)


    def _validate_input(self,ufunc,*args):
        # trig functions can accept dimensionless or angle arguments
        #   (units will enforce that base unit for angle is radians)
        fname = ufunc.__name__
        if fname in _TRIG_UFUNC:
            if not self.pdim.is_angle:
                raise TypeError(f"Argument to {fname} must be an angle, not {self.pdim}")

        elif fname in _COMPARISON_UFUNC:
            for arg in args:
                if not self.same_pdim(arg):
                    raise IncompatibleDimensions(self,arg)

        elif fname in _UNARY_UFUNC:
            pass

        else:
            print(f"ufunc={ufunc}")





    def _output_pdim(self,ufunc,*args):
        fname = ufunc.__name__
        
        if fname in _TRIG_UFUNC or fname in _COMPARISON_UFUNC:
            return None

        if fname in _UNARY_UFUNC:
            return self.pdim

        if fname in ('isfinite', 'less_equal'):
            return None

        return None

    # @@@TODO This is boilerplate from numpy website.
    #   tailor it to our specific needs
    def __array_ufunc__(self,ufunc,method,*args,out=None,**kwargs):
        self._validate_input(ufunc,*args)
        pdim = self._output_pdim(ufunc,args)

        in_args = [
            arg.view(np.ndarray) if isinstance(arg,Array) else arg
            for arg in args
            ]

        if out is not None:
            out_args = tuple(
                arg.view(np.ndarray) if isinstance(arg,Array) else arg
                for arg in out
                )
            kwargs['out'] = tuple(out_args)
        else:
            out = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *in_args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        # only need to convert output to Array if 
        if pdim is None or pdim.dimensionless:
            return results

        if ufunc.nout == 1:
            results = (results,)

        # only need to convert outputs to Array that weren't already
        #  input as Array instances via the 'out' parameter
        results = tuple( 
            o if isinstance(o,Array) else np.array(r).view(Array)
            for r,o in zip(results,out)
            )

        # attach physical dimension determined above
        for r in results:
            r.pdim = pdim

        # return tuple if multiple outputs
        return results if ufunc.nout > 1 else results[0]

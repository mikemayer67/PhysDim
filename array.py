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


    def __array_ufunc__(self,ufunc,method,*args,out=None,**kwargs):
        print(f"{ufunc}:: {args}")
        # validate input and find output physical dimension

        fname = ufunc.__name__
        io_map_name = _ufunc.io_mapping.get(fname,f"io_map_{fname}")
        io_map = getattr(_ufunc,io_map_name,None)
        if io_map is None:
            import pdb; pdb.set_trace()
            return NotImplemented

        pdim = io_map(self,ufunc,args)

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

        # multiple outputs require special handling...

        if ufunc.nout == 1:
            # only need to convert output to Array if 
            if pdim is None or pdim.is_dimensionless:
                return results

            if type(out[0]) is Array:
                results = out[0]
            else:
                results = np.array(results).view(Array)

            results.pdim = pdim
            return results

        else:
            rval = list()
            for r,o,p in zip(results, out, pdim):
                if p is None or p.is_dimensionless:
                    rval.append(r)
                elif type(o) is Array:
                    o.pdim = p
                    rval.append(o)
                else:
                    r = np.array(r).view(Array)
                    r.pdim = p
                    rval.append(r)

            return tuple(rval)

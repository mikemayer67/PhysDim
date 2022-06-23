"""Physical Values (and Arrays)

This is a subclass of a numpy ndarray that adds a physical
dimensionality attribute to numerical values. This includes 
both array ad scalar values, the latter being expressed as 
instances of a 0-Dimension ndarray.

The physical dimensionality is based on tracking the exponents on
7 fundamental physical properties:
    - Length (L)
    - Time (T)
    - Mass (M)
    - Angle (A)
    - Electric charge (E)
    - Absolute temperature (K)
    - Intensity of light (I)

There are no traditional units attached to a given PhysicalValue instance.  
The notion of units is introduced through physdim.Unit.

PhysicalValue ensures that the any mathematical operation properly
handles the physical dimension.  In addition to returning
the values with the appropriate dimensionality, it also raises 
exceptions when attempts are made to mix incompatible
physical dimensions when evaluating expressions.  

For example:
    - NOT allowed:
      - Adding a length and a mass
      - Taking difference between a length and a mass
    - Allowed:
      - Adding two lengths
      - Taking difference between two accelerations
      - Multiplying a length and a mass (yields moment of inertia)
      - Dividing a mass by a volume (yields density)

    - Allowed:
      - Applying a trig function to an angle
      - Applying a trig function to a dimensionless value
      - Applying arctan2 to two identical physical dimensions
      - Applyng an inverse trig function to a dimensionless value
    - NOT allowed:
      - Applying a trig function to anything else
      - Applying an inverse trig function anything else
"""

import numpy as np
from numbers import Number

from .dim import PhysicalDimension
from .ufunc import io_map as ufunc_io_map
from .exceptions import IncompatibleDimensions

class PhysicalValue(np.ndarray):
    __slot__ = ('pdim')

    def __new__(cls, *args, **kwargs):
        parsed_args = _parse_args(args,kwargs)
        if kwargs:
            raise TypeError(" ".join(( 
                "Unrecognized inputs to PhysicalValue constructor:",
                ", ".join(kwargs.keys()))))
        if 'ref' in parsed_args:
            obj = parsed_args['ref']
            if 'scale' in parsed_args:
                obj = parsed_args['scale'] * obj
                # if ref is dimensionless, scaling will return a np.ndarray
                if type(obj) is not PhysicalValue:
                    obj = obj.view(PhysicalValue)
                    obj.pdim = PhysicalDimension()
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
            obj = obj.view(PhysicalValue)
            obj.pdim = parsed_args['pdim'] if 'pdim' in parsed_args else PhysicalDimension()
        return obj

    def __array_finalize__(self,obj):
        if obj is None: return
        if not issubclass(self.dtype.type, np.number):
            raise TypeError(f"Cannot create PhysicalValue for {self.dtype}")
        self.pdim = getattr(obj,'pdim',None)

    def __repr__(self):
        value = super().__repr__()
        pdim = self.pdim.__repr__()
        return f"{value[:-1]}, pdim={pdim})"

    def __str__(self):
        from .units import Units
        u = Units()
        unit_name, scale_factor = u.best_unit(self)
        value = self.view(np.ndarray)
        if scale_factor != 1:
            value = value * scale_factor
        return f"{value} {unit_name}"

    def __getattr__(self,name):
        from .units import Units
        u = Units()
        unit = getattr(u,name)
        if unit.pdim != self.pdim:
            raise IncompatibleDimensions(self,unit)
        return self/unit


    @property
    def is_angle(self):
        return self.pdim.is_angle

    @property
    def is_dimensionless(self):
        return self.pdim.is_dimensionless

    def __array_ufunc__(self,ufunc,method,*args,out=None,**kwargs):
        pdim = ufunc_io_map(ufunc,self,args)

        # down-convert any PhysicalValue inputs to numpy.ndarray 
        in_args = [ 
            arg.view(np.ndarray) if isinstance(arg,PhysicalValue) else arg
            for arg in args
            ]

        # down-convert any user specified PhysicalValue outputs to numpy.ndarray
        if out is not None:
            kwargs['out'] = tuple(
                o.view(np.ndarray) if isinstance(o,PhysicalValue) else o
                for o in out 
                )

        # invoke the ufunc without PhysicalValue arguments
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
        if type(out) is PhysicalValue:
            result = out
        elif isinstance(result,np.ndarray):
            result = result.view(PhysicalValue)
        else:
            result = np.asarray(result).view(PhysicalValue)
        result.pdim = pdim
    elif type(out) is PhysicalValue:
        result = out.view(np.ndarray)
    return result

_invalid_arg_pairs = { 
    ('ref','array'),('ref','values'),('ref','pdim'),
    ('array','values'),('array','scale'),
    ('scale','values'), }

def _add_arg(args, arg, key):
    if key in args:
        raise TypeError(f"Can only specify one {key} argument")
    for k in args.keys():
        if (key,k) in _invalid_arg_pairs or (k,key) in _invalid_arg_pairs:
            raise TypeError(f"Cannot specify both {key} and {k} arguments")
    args[key] = arg

def _parse_args(args,kwargs):
    rval = {}
    if 'pdim' in kwargs:
        rval['pdim'] = kwargs['pdim']
        del kwargs['pdim']
    for arg in args:
        if isinstance(arg,PhysicalValue):
            _add_arg(rval,arg,'ref')
        elif isinstance(arg,np.ndarray):
            if arg.size == 1:
                _add_arg(rval,arg.flat[0],'scale')
            else:
                _add_arg(rval,arg,'array')
        elif isinstance(arg,PhysicalDimension):
            _add_arg(rval,arg,'pdim')
        elif isinstance(arg,Number):
            _add_arg(rval,arg,'scale')
        else:
            _add_arg(rval,arg,'values')
    return rval


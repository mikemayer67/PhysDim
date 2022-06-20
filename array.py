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
from .exceptions import UnsupportedUfunc

# List of numpy ufuncs was found here:
#   https://numpy.org/devdocs/reference/ufuncs.html

# ufuncs grouped by functional mapping
#
# We don't distinguish on dtype, numpy will handle exceptions on invalid
#   dtypes passed to the ufunc. We only compare about the physical
#   dimensionality

#@@@WORK_HERE (pick up with remainder, mod, etc...)

_UFUNC_BY_MAPPING = {
    # angle to number: (A)->(N)
    "A_N" : ('sin','cos','tan',),
    # number to angle: (N)->(A)
    "N_A" : ('arcsin','arccos','arctan'),
    # unary function returning same dimension as input: (X)->(X)
    "X_X" : ('negative','positive','absolute','fabs','invert','conj','conjugate',),
    # unary function returning number (or bool): (X)->(N)
    "X_N" : ('sign','heaviside','isfinite',),
    # pair of same dimension to boolean: (X,X)->(B)
    "XX_B" : ('less','less_equal','equal','not_equal','greater', 'greater_equal',),
    # pair of same dimension to returning same dimension as input: (X,X)->(X)
    "XX_X" : ('add','subtract','heaviside','hypot',
              'maximum','fmax','minimum','fmin',),
    # function multiplies units from input: (X,Y)->(XY)
    "mul" : ('multiply','matmul',),
    # function divides units from input: (X,Y)->(X/Y)
    "div" : ('divide','true_divide','floor_divide',),
    # function raise units to a power: (X,N)->(X**N)
    "pow" : ('power','float_power',),
    # modulus functions: (X,X)->(X) or (X,X)->(N,X)
    "mod" : ('mod','fmod','remainder',),
    # functions which require dimensionless input
    "N_N" : ('logaddexp','logaddexp2','rint', 'exp', 'exp2', 
             'log', 'log2', 'log10', 'expm1','log1p', 'gcd','lcm',
             'sinh','cosh','tanh','arcsinh','arccosh','arctanh',
            ),
    # unsupported functions
    "fail" : ('radians','degrees','deg2rad','rad2deg',
             'bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert',
             'left_shift','right_shift', 'logical_and', 'logical_or',
             'logical_xor', 'logical_not',),
}

def assert_one_input(ufunc,args):
    assert len(args) == 1, (
        f"{ufunc.__name__} expects 1 input, {len(args)} found" )
    assert ufunc.nin == 1, (
        f"{ufunc.__name__} expects 1 input, but ufunc.nin is {ufunc.nin}" )

def assert_two_inputs(ufunc,args):
    assert len(args) == 2, (
        f"{ufunc.__name__} expects 2 inputs, {len(args)} found" )
    assert ufunc.nin == 2, (
        f"{ufunc.__name__} expects 2 inputs, but ufunc.nin is {ufunc.nin}" )


class Array(np.ndarray):
    __slot__ = ('pdim')
    _ufunc_io_mapping = None

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

    def same_pdim(self,other):
        try:
            return self.pdim == other.pdim
        except:
            return False

    def assert_same_pdim(self,other):
        if not self.same_pdim(other):
            raise IncompatibleDimensions(self,other)

    def io_map_a_n(self,ufunc,args):
        if not self.pdim.is_angle:
            raise TypeError(
                f"Argument to {ufunc.__name__} must be an angle, not {self.pdim}")
        return None

    def io_map_n_a(self,ufunc,args):
        for arg in args:
            if isinstance(arg,Array) and not arg.is_dimensionless:
                raise TypeError(" ".join((
                    f"Argument{'s' if ufunc.nin>1 else ''} to {ufunc.__name__}",
                    f"must be dimensionless, not {tuple(arg.pdim for arg in args)}")))
        return Dim(angle=1)

    def io_map_arctan2(self,ufunc,args):
        assert_two_inputs(ufunc,args)
        for arg in args:
            if not self.same_pdim(arg):
                raise IncompatibleDimensions(self,arg)
        return Dim(angle=1)

    def io_map_n_n(self,ufunc,args):
        for arg in args:
            if isinstance(arg,Array) and not arg.is_dimensionless:
                raise TypeError(" ".join((
                    f"Argument{'s' if ufunc.nin>1 else ''} to {ufunc.__name__}",
                    f"must be dimensionless, not {tuple(arg.pdim for arg in args)}")))
        return None

    def io_map_x_x(self,ufunc,args):
        return getattr(self,"pdim",None)

    def io_map_x_n(self,ufunc,args):
        return None

    def io_map_xx_b(self,ufunc,args):
        for arg in args:
            if not self.same_pdim(arg):
                raise IncompatibleDimensions(self,arg)
        return None

    def io_map_xx_x(self,ufunc,args):
        for arg in args:
            if not self.same_pdim(arg):
                raise IncompatibleDimensions(self,arg)
        return self.pdim

    def io_map_mul(self,ufunc,args):
        assert_two_inputs(ufunc,args)

        pdim = [getattr(x,'pdim',None) for x in args]
        if pdim[0] is None:
            return pdim[1]
        if pdim[1] is None:
            return pdim[0]
        return pdim[0] * pdim[1]

    def io_map_div(self,ufunc,args):
        assert_two_inputs(ufunc,args)

        pdim = [getattr(x,'pdim',None) for x in args]
        if pdim[0] is None:
            return pdim[1].inverse
        if pdim[1] is None:
            return pdim[0]
        return pdim[0] / pdim[1]

    def io_map_pow(self,ufunc,args):
        assert_two_inputs(ufunc,args)

        n = args[1]
        if n is Array:
            raise TypeError(" ".join((
                "Cannot raise a nubmer to a dimensional quantity:",
                f"{ufunc.__name__} {pdim[1]}")))
        if n is np.ndarray and n.size > 1:
            raise TypeError(" ".join((
                "Cannot raise a dimensional quantity to anything other than a scalar:"
                f"{ufunc.__name__} {pdim[1]}")))

        # having ruled out args[1] as an Array, args[0] must be an Array
        return args[0].pdim ** n

    def io_map_mod(self,ufunc,args):
        assert_two_inputs(ufunc,args)
        if type(args[0]) is not Array:
            raise TypeError(f"Dividend must be a {type(self)}, not {args[0]}")
        if type(args[1]) is not Array:
            raise TypeError(f"Divisor must be a {type(self)}, not {args[1]}")
        if args[0].pdim != args[1].pdim:
            raise TypeError(" ".join((
                "Dividend and divisor must have same dimensionality:",
                f"{args[0].pdim} is not the same as {args[1].pdim}")))
        return self.pdim

    def io_map_divmod(self,ufunc,args):
        # only difference between divmod and mod is that former returns two
        # values, the first of which much be dimensionless
        return (None, self.io_map_mod(ufunc,args))

    def io_map_square(self,ufunc,args):
        assert_one_input(ufunc,args)
        return self.pdim ** 2

    def io_map_sqrt(self,ufunc,args):
        assert_one_input(ufunc,args)
        return self.pdim ** 0.5 

    def io_map_cbrt(self,ufunc,args):
        print(f"cbrt: {args}")
        assert_one_input(ufunc,args)
        return self.pdim ** (1/3)

    def io_map_reciprocal(self,ufunc,args):
        assert_one_input(ufunc,args)
        return self.pdim ** -1 

    def io_map_fail(self,ufunc,args):
        raise UnsupportedUfunc(ufunc.__name__)

    def fmin(*args,**kwargs):
        import pdb; pdb.set_trace()
        return super().fmin(*args,**kwargs)


    def __array_ufunc__(self,ufunc,method,*args,out=None,**kwargs):
        print(f"{ufunc}:: {args}")
        if ufunc.__name__ == "minimum":
            import pdb; pdb.set_trace()
        # validate input and find output physical dimension

        fname = ufunc.__name__
        io_map = Array._ufunc_io_mapping.get(fname,None)
        if io_map is None:
            io_map = getattr(Array,f"io_map_{fname}",None)
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

Array._ufunc_io_mapping = {
    fname: getattr(Array,f"io_map_{mapping.lower()}")
    for mapping, ufunc_list in _UFUNC_BY_MAPPING.items()
    for fname in ufunc_list
}

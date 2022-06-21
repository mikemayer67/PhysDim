"""Universal Function (ufunc) support to PhysDim.Array

In order to keep the efficiency provided by numpy's ufuncs, we need
to cast any PhysDim.Array object passed as input to a ufunc into 
a numpy.ndarray.  For most (but not all) ufunc, we will also want
to cast the returned numpy.ndarray to a PhySim.Array with appropriate
physical dimensionality.
"""

import numpy as np

from ._funcs import output_pdim
from ._funcs import same_pdim
from ._funcs import assert_same_pdim

from .exceptions import IncompatibleDimensions
from .exceptions import UnsupportedUfunc
from .dim import Dim

# List of numpy ufuncs was found here:
#   https://numpy.org/devdocs/reference/ufuncs.html

# ufuncs grouped by functional mapping
#
# We don't distinguish on dtype, numpy will handle exceptions on invalid
#   dtypes passed to the ufunc. We only compare about the physical
#   dimensionality

#@@@WORK_HERE (pick up with remainder, mod, etc...)

_UFUNC_MAPPING = {
    # angle to number: (A)->(N)
    "A_N" : ('sin','cos','tan',),
    # number to angle: (N)->(A)
    "N_A" : ('arcsin','arccos','arctan'),
    # unary function returning same dimension as input: (X)->(X)
    "X_X" : ('negative','positive','absolute','fabs','invert','conj','conjugate',
             'fabs',),
    # unary function returning number (or bool): (X)->(N)
    "X_N" : ('sign','heaviside','isfinite','isinf','isnan','signbit'),
    # pair of same dimension to boolean: (X,X)->(B)
    "XX_B" : ('less','less_equal','equal','not_equal','greater', 'greater_equal',),
    # pair of same dimension to returning same dimension as input: (X,X)->(X)
    "XX_X" : ('add','subtract','heaviside','hypot',
              'maximum','fmax','minimum','fmin'),
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
             'sinh','cosh','tanh','arcsinh','arccosh','arctanh',),
    # unsupported functions
    "fail" : ('radians','degrees','deg2rad','rad2deg',
             'bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert',
             'left_shift','right_shift', 'logical_and', 'logical_or',
             'logical_xor', 'logical_not','isnat','nextafter','spacing',
              'modf','ldexp','frexp','floor','ceil','trunc'),
}

io_mapping = {
    fname: f"io_map_{mapping.lower()}"
    for mapping, ufunc_list in _UFUNC_MAPPING.items()
    for fname in ufunc_list
}

# Convenience functions for testing input counts

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

# Functions for validating ufunc inputs involviing a PhysDim.Array input
#   and returning the appropriate physical dimensionality of each output

def io_map_a_n(ufunc,obj,args):
    if not obj.pdim.is_angle:
        raise TypeError(
            f"Argument to {ufunc.__name__} must be an angle, not {obj.pdim}")
    return None

def io_map_n_a(ufunc,obj,args):
    for arg in args:
        if isinstance(arg,type(obj)) and not arg.is_dimensionless:
            raise TypeError(" ".join((
                f"Argument{'s' if ufunc.nin>1 else ''} to {ufunc.__name__}",
                f"must be dimensionless, not {tuple(arg.pdim for arg in args)}")))
    return Dim(angle=1)

def io_map_arctan2(ufunc,obj,args):
    assert_two_inputs(ufunc,args)
    for arg in args:
        if not same_pdim(obj,arg):
            raise IncompatibleDimensions(obj,arg)
    return Dim(angle=1)

def io_map_n_n(ufunc,obj,args):
    for arg in args:
        if isinstance(arg,type(obj)) and not arg.is_dimensionless:
            raise TypeError(" ".join((
                f"Argument{'s' if ufunc.nin>1 else ''} to {ufunc.__name__}",
                f"must be dimensionless, not {tuple(arg.pdim for arg in args)}")))
    return None

def io_map_x_x(ufunc,obj,args):
    return output_pdim(obj.pdim)

def io_map_x_n(ufunc,obj,args):
    return None

def io_map_xx_b(ufunc,obj,args):
    for arg in args:
        if not same_pdim(obj,arg):
            raise IncompatibleDimensions(obj,arg)
    return None

def io_map_xx_x(ufunc,obj,args):
    for arg in args:
        if not same_pdim(obj,arg):
            raise IncompatibleDimensions(obj,arg)
    return output_pdim(obj.pdim)

def io_map_mul(ufunc,obj,args):
    assert_two_inputs(ufunc,args)

    pdim = [getattr(x,'pdim',None) for x in args]
    if pdim[0] is None:
        return output_pdim(pdim[1])
    if pdim[1] is None:
        return output_pdim(pdim[0])
    return output_pdim(pdim[0] * pdim[1])

def io_map_div(ufunc,obj,args):
    assert_two_inputs(ufunc,args)

    pdim = [getattr(x,'pdim',None) for x in args]
    if pdim[0] is None:
        return output_pdim(pdim[1].inverse)
    if pdim[1] is None:
        return output_pdim(pdim[0])
    return output_pdim(pdim[0] / pdim[1])

def io_map_pow(ufunc,obj,args):
    assert_two_inputs(ufunc,args)

    n = args[1]
    if n is type(obj):
        raise TypeError(" ".join((
            "Cannot raise a nubmer to a dimensional quantity:",
            f"{ufunc.__name__} {pdim[1]}")))
    if n is np.ndarray and n.size > 1:
        raise TypeError(" ".join((
            "Cannot raise a dimensional quantity to anything other than a scalar:"
            f"{ufunc.__name__} {pdim[1]}")))

    # having ruled out args[1] as an Array, args[0] must be an Array
    return output_pdim(args[0].pdim ** n)

def io_map_mod(ufunc,obj,args):
    assert_two_inputs(ufunc,args)
    if type(args[0]) is not type(obj):
        raise TypeError(f"Dividend must be a {type(obj)}, not {args[0]}")
    if type(args[1]) is not type(obj):
        raise TypeError(f"Divisor must be a {type(obj)}, not {args[1]}")
    if args[0].pdim != args[1].pdim:
        raise TypeError(" ".join((
            "Dividend and divisor must have same dimensionality:",
            f"{args[0].pdim} is not the same as {args[1].pdim}")))
    return output_pdim(obj.pdim)

def io_map_divmod(ufunc,obj,args):
    # only difference between divmod and mod is that former returns two
    # values, the first of which much be dimensionless
    pdim = io_map_mod(ufunc,obj,args)
    return None if pdim is None else (None, pdim)

def io_map_square(ufunc,obj,args):
    assert_one_input(ufunc,args)
    return output_pdim(obj.pdim ** 2)

def io_map_sqrt(ufunc,obj,args):
    assert_one_input(ufunc,args)
    return output_pdim(obj.pdim ** 0.5)

def io_map_cbrt(ufunc,obj,args):
    print(f"cbrt: {args}")
    assert_one_input(ufunc,args)
    return output_pdim(obj.pdim ** (1/3))

def io_map_reciprocal(ufunc,obj,args):
    assert_one_input(ufunc,args)
    return output_pdim(obj.pdim ** -1)

def io_map_copysign(ufunc,obj,args):
    assert_two_inputs(ufunc,args)
    if type(args[0]) is not type(obj):
        return args[1].pdim
    else:
        return args[0].pdim

def io_map_fail(ufunc,obj,args):
    raise UnsupportedUfunc(ufunc.__name__)


def io_map(ufunc,obj,args):
    fname = ufunc.__name__
    io_map_name = io_mapping.get(fname,f"io_map_{fname}")
    io_map_func = globals().get(io_map_name,None)
    if io_map_func is None:
        import pdb; pdb.set_trace()
        return NotImplemented

    return io_map_func(ufunc,obj,args)

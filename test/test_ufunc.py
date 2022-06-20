import unittest

import numpy as np
import numbers

from PhysDim import Array as PDA
from PhysDim import Dim 

import PhysDim._ufunc as ufunc

from PhysDim.exceptions import IncompatibleDimensions
from PhysDim.exceptions import UnsupportedUfunc

_length = Dim(length=1)
_time = Dim(time=1)
_density = Dim(mass=1,length=-3)
_angle = Dim(angle=1)
_dimless = Dim()

def is_number(x):
    return issubclass(x.dtype.type, np.number)

def is_integer(x):
    return issubclass(x.dtype.type, np.integer)

def is_inexact(x):
    return issubclass(x.dtype.type, np.inexact)

def is_complex(x):
    return issubclass(x.dtype.type, complex)

def is_bool(x):
    return x.dtype.type is np.bool_


class UfuncTests(unittest.TestCase):

    def setUp(self):
        shape = (2,3)
        size = np.prod(shape)

        self.x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        self.y = PDA((np.arange(size)+0.5).reshape(shape),pdim=_length)
        self.z = PDA((np.arange(size)+(1+.5j)).reshape(shape),pdim=_length)
        self.t = PDA((np.arange(size)+1).reshape(shape),pdim=_time)
        self.a = PDA(np.arange(size).reshape(shape),pdim=_angle)
        self.n = PDA(np.arange(size).reshape(shape),pdim=_dimless)
        self.s = PDA(np.array(3),pdim=_length)

    def test_same_pdim(self):
        x = PDA(5,pdim=_length)
        y = PDA(5+2j,pdim=_length)
        t = PDA(1.2,pdim=_time)

        self.assertTrue(ufunc.same_pdim(x,y))
        self.assertFalse(ufunc.same_pdim(x,t))

        ufunc.assert_same_pdim(x,y)
        with self.assertRaises(IncompatibleDimensions):
            ufunc.assert_same_pdim(x,t)

    def test_trig(self):
        x, y, a, n, s  = (self.x, self.y, self.a, self.n, self.s)

        for f in (np.sin, np.cos, np.tan):
            for v in (a,n):
                result = f(v)
                self.assertTrue(type(result) is np.ndarray)
                self.assertEqual(result.shape,v.shape)

                f(v,out=np.zeros(v.shape))
                self.assertTrue(type(result) is np.ndarray)
                self.assertEqual(result.shape,v.shape)

            with self.assertRaisesRegex(TypeError, "^Argument to.*must be an angle"):
                result = f(x)

        n = self.n
        n = PDA(n / (1 + n.size),pdim=_dimless)

        for f in (np.arcsin, np.arccos, np.arctan):
            result = f(n)
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, n.shape)
            self.assertEqual(result.pdim, _angle)

            with self.assertRaisesRegex(TypeError, "^Argument to.*must be dimensionless"):
                result = f(x)

        for a,b in ((x,y),(x,s),(s,x)):
            result = np.arctan2(a,b)
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.pdim, _angle)

        result = np.hypot(x,y)
        self.assertTrue(type(result) is PDA)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, _length)

        with self.assertRaises(IncompatibleDimensions):
            result = np.hypot(self.x,self.t)

    def test_htrig(self):
        x, a, = (self.x, self.a)

        n = self.n / (1+self.n.size)
        for f in (np.sinh, np.cosh, np.tanh, np.arcsinh, np.arctanh):
            result = f(n)
            self.assertTrue(type(result) is np.ndarray)
            self.assertEqual(result.shape,n.shape)

            for v in (x,a):
                with self.assertRaisesRegex(TypeError, "^Argument to.*must be dimensionless"):
                    result = f(v)

        n = PDA(self.n.view(np.ndarray) + 2,pdim=_dimless)
        result = np.arccosh(n)
        self.assertTrue(type(result) is np.ndarray)
        self.assertEqual(result.shape,n.shape)

        for v in (x,a):
            with self.assertRaisesRegex(TypeError, "^Argument to.*must be dimensionless"):
                result = np.arccosh(v)

    def test_angle_conv_fail(self):
        x, a, n  = (self.x, self.a, self.n)
        for f in (np.degrees, np.radians, np.deg2rad, np.rad2deg):
            for v in (x,a,n):
                with self.assertRaises(UnsupportedUfunc):
                    result = f(v)

    def test_ufunc_compare(self):
        x, y, z, t  = (self.x, self.y, self.z, self.t)

        result = x < y
        self.assertTrue(type(result) is np.ndarray)
        self.assertEqual(result.dtype, bool)
        self.assertEqual(result.shape, x.shape)

        for f in (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
            for v in (y,z):
                result = f(x,v)
                self.assertTrue(type(result) is np.ndarray)
                self.assertEqual(result.dtype, bool)
                self.assertEqual(result.shape, x.shape)

            with self.assertRaises(IncompatibleDimensions):
                result = f(x,t)

    def test_ufunc_unary_op(self):
        x, y, z, t, a, n  = (self.x, self.y, self.z, self.t, self.a, self.n)

        result = -x
        self.assertTrue(type(result) is PDA)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        for f in (np.negative, np.positive, np.absolute, np.conj, np.conjugate):
            for v in (x,y,z,t,a,n):
                if f is np.invert:
                    if v is z: continue
                    if v is y: continue

                result = f(v)
                self.assertEqual(result.shape, v.shape)
                self.assertTrue(type(result) == (np.ndarray if v is n else PDA))
                if f is not np.absolute or v is not z:
                    self.assertEqual(result.dtype, v.dtype)
                if v is not n:
                    self.assertEqual(result.pdim, v.pdim)

    def test_ufunc_unary_test(self):
        x, y, z, t,n  = (self.x, self.y, self.z, self.t, self.n)

        for f in (np.sign,):
            for v in (x,y,z,t,n):
                result = f(v)
                self.assertEqual(result.shape, v.shape)
                self.assertTrue(is_number(result))

        for f in (np.isfinite,):
            for v in (x,y,z,t,n):
                result = f(v)
                self.assertEqual(result.shape, v.shape)
                self.assertTrue(is_bool(result))

    def test_ufunc_add_like(self):
        x, y, z, t = (self.x, self.y, self.z, self.t)

        result = x + y
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)
        
        for f in (np.add, np.subtract,):
            for v in (x,y,z):
                result = f(x,v)
                self.assertEqual(result.shape, v.shape)
                self.assertTrue(type(result) is PDA)
                self.assertEqual(result.dtype, v.dtype)
                self.assertEqual(result.pdim, v.pdim)

            with self.assertRaises(IncompatibleDimensions):
                result = f(x,t)

        result = np.heaviside(x,y)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_number(result))
        self.assertEqual(result.pdim, y.pdim)


    def test_ufunc_mul(self):
        x, y, z, t = (self.x, self.y, self.z, self.t)

        result = x * y
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim * y.pdim)

        result = x * z
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_complex(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim * z.pdim)

        result = x * t
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim * t.pdim)

        tt = PDA(5,pdim=_time)
        result = x * tt
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim * _time)

        result = tt * x
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim * _time)

        result = x * 5.2
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        result = (2+3j) * x
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_complex(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        n = np.arange(x.size).reshape(x.shape)
        result = x * n
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        result = n * x
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)


    def test_ufunc_div(self):
        x, y, z, t = (self.x, self.y, self.z, self.t)
        t.flat[3] = 100  # avoid div by zero

        result = x / y
        self.assertTrue(type(result) == np.ndarray)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)

        result = x / z
        self.assertTrue(type(result) == np.ndarray)
        self.assertTrue(is_complex(result))
        self.assertEqual(result.shape, x.shape)

        result = x / t
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim / t.pdim)

        result = x // t
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim / t.pdim)

        tt = PDA(5,pdim=_time)
        result = x / tt
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim / _time)

        result = x // tt
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_integer(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim / _time)

        result = tt / x
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, _time / x.pdim)

        result = x / 5.2
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        result = (2+3j) / x
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_complex(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim.inverse)

        n = (np.arange(x.size)+1).reshape(x.shape)
        result = x / n
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        result = n / x
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_inexact(result))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim.inverse)

    def test_ufunc_pow_funcs(self):
        x, y, z = (self.x, self.y, self.z)

        for v in (x,y,z):
            result = v**2
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, v.shape)
            self.assertEqual(result.pdim, _length*_length)

            result = np.square(v)
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, v.shape)
            self.assertEqual(result.pdim, _length*_length)

            result = np.sqrt(v)
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, v.shape)
            self.assertEqual(result.pdim, _length**0.5)

            if not v is z:
                result = np.cbrt(v)
                self.assertTrue(type(result) is PDA)
                self.assertEqual(result.shape, v.shape)
                self.assertEqual(result.pdim, _length**(1/3))

            result = np.reciprocal(v)
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, v.shape)
            self.assertEqual(result.pdim, _length**-1)

    def test_ufunc_pow(self):
        x, y, z, t = (self.x, self.y, self.z, self.t)
        t.flat[3] = 100  # avoid div by zero

        for n in (3, 2.5):
            result = x ** n
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.pdim, _length**n)

        with self.assertRaisesRegex(TypeError,
            "^Exponents on dimensions must all be real numbers"):
            result = x ** (1+1j)

    def test_ufunc_mod(self):
        x, y, t = (self.x, self.y, self.t)
        t.flat[3] = 100  # avoid div by zero

        n = PDA(5,pdim=_length)

        for v in (y,n):
            result = x % v
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.pdim, _length)

        for v in (y,n):
            result = v % x
            self.assertTrue(type(result) is PDA)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.pdim, _length)

        with self.assertRaisesRegex(TypeError,
            "^Dividend and divisor must have same dimensionality"):
            result = x % t

    def test_ufunc_divmod(self):
        x, y, t = (self.x, self.y, self.t)
        t.flat[3] = 100  # avoid div by zero

        n = PDA(5,pdim=_length)

        qo = PDA(np.zeros(x.shape),pdim=_dimless)
        ro = PDA(np.zeros(x.shape),pdim=_dimless)

        for v in (y,n):
            q,r = divmod(x,v)

            self.assertTrue(type(q) is np.ndarray)
            self.assertTrue(type(r) is PDA)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(r.shape, x.shape)
            self.assertEqual(r.pdim, _length)

            q,r = np.divmod(x,v,out=(qo,ro))
            self.assertTrue(type(q) is np.ndarray)
            self.assertTrue(type(r) is PDA)
            self.assertTrue(type(qo) is PDA)
            self.assertTrue(type(ro) is PDA)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(r.shape, x.shape)
            self.assertEqual(r.pdim, _length)
            self.assertEqual(qo.shape, x.shape)
            self.assertEqual(ro.shape, x.shape)
            self.assertEqual(ro.pdim, _length)

            q,r = np.divmod(x,v,out=(qo,None))
            self.assertTrue(type(q) is np.ndarray)
            self.assertTrue(type(r) is PDA)
            self.assertTrue(type(qo) is PDA)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(r.shape, x.shape)
            self.assertEqual(r.pdim, _length)
            self.assertEqual(qo.shape, x.shape)

            q,r = np.divmod(x,v,out=(None,ro))
            self.assertTrue(type(q) is np.ndarray)
            self.assertTrue(type(r) is PDA)
            self.assertTrue(type(ro) is PDA)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(r.shape, x.shape)
            self.assertEqual(r.pdim, _length)
            self.assertEqual(ro.shape, x.shape)
            self.assertEqual(ro.pdim, _length)



        with self.assertRaisesRegex(TypeError,
            "^Dividend and divisor must have same dimensionality"):
            q,r = divmod(x,t)

    def test_ufunc_dimensionless_unary(self):
        x, n = (self.x, self.n)

        n = PDA(n.view(np.ndarray) + 2,pdim=_dimless)

        for f in (np.rint, np.exp, np.exp2, np.log, np.log2, 
                  np.log10, np.expm1, np.log1p):

            result = f(n)
            self.assertTrue(type(result) is np.ndarray)
            self.assertEqual(result.shape, n.shape)

            with self.assertRaisesRegex(TypeError,"^Argument to.*must be dimensionless"):
                result = f(x)

    def test_ufunc_dimensionless_op(self):
        (x,y,n) = (self.x, self.y, self.n)
        m = np.transpose(n).reshape(n.shape)

        for f in (np.logaddexp,np.logaddexp2,np.gcd,np.lcm):
            result = f(n,m)
            self.assertTrue(type(result) is np.ndarray)
            self.assertEqual(result.shape, n.shape)

            with self.assertRaisesRegex(TypeError,"^Arguments to.*must be dimensionless"):
                result = f(x,y)

            with self.assertRaisesRegex(TypeError,"^Arguments to.*must be dimensionless"):
                result = f(x,n)

    def test_ufunc_bitwise(self):
        (x,y,n) = (self.x, self.y, self.n)
        s = 5

        for v in (y,n,s):
            with self.assertRaises(UnsupportedUfunc):
                result = x & v
            with self.assertRaises(UnsupportedUfunc):
                result = x | v
            with self.assertRaises(UnsupportedUfunc):
                result = x ^ v
            with self.assertRaises(UnsupportedUfunc):
                result = x << v
            with self.assertRaises(UnsupportedUfunc):
                result = x >> v

            with self.assertRaises(UnsupportedUfunc):
                result = v & x
            with self.assertRaises(UnsupportedUfunc):
                result = v | x
            with self.assertRaises(UnsupportedUfunc):
                result = v ^ x
            with self.assertRaises(UnsupportedUfunc):
                result = v << x
            with self.assertRaises(UnsupportedUfunc):
                result = v >> x

        with self.assertRaises(UnsupportedUfunc):
            result = ~x


    def test_ufunc_logical(self):
        (x,y,n) = (self.x, self.y, self.n)
        s = 5

        for f in (np.logical_and, np.logical_or, np.logical_xor):
            for v in (y,n,s):
                with self.assertRaises(UnsupportedUfunc):
                    result = f(x,v)
                with self.assertRaises(UnsupportedUfunc):
                    result = f(v,x)

        with self.assertRaises(UnsupportedUfunc):
            result = np.logical_not(x)


#    def test_ufunc_minmax(self):
#        (x,y,z,n) = (self.x, self.y, self.z, self.n)
#        s = 5
#        
#        for f in (np.min, np.max, np.minimum, np.maximum):
#            for v in (x,y):
#                result = f(x,v)
#                self.assertTrue(type(result) is PDA)
#                self.assertEqual(result.shape, x.shape)
#                self.assertEqual(result.pdim, _length)

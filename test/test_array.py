import unittest

import numpy as np
import numbers

from PhysDim import Array as PDA
from PhysDim import Dim 

from PhysDim.exceptions import IncompatibleDimensions
from PhysDim.exceptions import UnsupportedUfunc

_length = Dim(length=1)
_time = Dim(time=1)
_angle = Dim(angle=1)
_dimless = Dim()

def is_close(a,b):
    return np.allclose(a.view(np.ndarray), b.view(np.ndarray))

def is_number(x):
    return issubclass(x.dtype.type, np.number)

def is_integer(x):
    return issubclass(x.dtype.type, np.integer)

def is_inexact(x):
    return issubclass(x.dtype.type, np.inexact)

def is_complex(x):
    return issubclass(x.dtype.type, np.complex)

def is_bool(x):
    return x.dtype.type is np.bool_

class ArrayTests(unittest.TestCase):

    def test_init_array(self):
        array = [[1,2],[3,4],[5,6]]
        x = PDA(array, pdim=_length)

        expected = np.array(array)
        self.assertEqual(x.shape, expected.shape)
        self.assertTrue(is_close(x,expected))
        self.assertEqual(x.pdim, _length)

        array = np.arange(24).reshape((6,-1))
        x = PDA(array, pdim=_time)
        self.assertEqual(x.shape, array.shape)
        self.assertTrue(is_close(x, array))
        self.assertEqual(x.pdim, _time)

        x = PDA(82,pdim=_length/_time)
        self.assertEqual(x.shape, ())
        self.assertEqual(int(x),82)
        self.assertEqual(x.pdim, Dim(length=1,time=-1))

    def test_init_shape(self):
        x = PDA(shape=(2,3,4),pdim=Dim())
        self.assertEqual(x.shape,(2,3,4))
        self.assertEqual(x.pdim, _length/_length)

        x = PDA(shape=(),pdim=_length)
        self.assertEqual(x.shape,())
        self.assertEqual(x.pdim, _length)

    def test_init_failure(self):
        with self.assertRaisesRegex(TypeError,"^Cannot initialize.*with both shape and array") as cm:
            x = PDA([1,2,3],shape=(5,2),pdim=_length)
        with self.assertRaisesRegex(TypeError,"^The input array must be a simple array") as cm:
            x = PDA(['cat','dog','pencil'],pdim=_time)

    def test_same_pdim(self):
        x = PDA(5,pdim=_length)
        y = PDA(5+2j,pdim=_length)
        t = PDA(1.2,pdim=_time)

        self.assertTrue(x.same_pdim(y))
        self.assertFalse(x.same_pdim(t))

        x.assert_same_pdim(y)
        with self.assertRaises(IncompatibleDimensions):
            x.assert_same_pdim(t)

    def test_ufunc_trig(self):
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        a = PDA(np.arange(size).reshape(shape),pdim=_angle)
        n = PDA(np.arange(size).reshape(shape),pdim=_dimless)

        for f in (np.sin, np.cos, np.tan):
            for v in (a,n):
                result = f(n)
                self.assertTrue(type(result) is np.ndarray)
                self.assertEqual(result.shape,shape)

                f(n,out=np.zeros(shape))
                self.assertTrue(type(result) is np.ndarray)
                self.assertEqual(result.shape,shape)

            with self.assertRaisesRegex(TypeError, "^Argument to.*must be an angle") as cm:
                result = f(x)

    def test_ufunc_compare(self):
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)

        result = x < y
        self.assertTrue(type(result) is np.ndarray)
        self.assertEqual(result.dtype, bool)
        self.assertEqual(result.shape, shape)

        for f in (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
            for v in (y,z):
                result = f(x,v)
                self.assertTrue(type(result) is np.ndarray)
                self.assertEqual(result.dtype, bool)
                self.assertEqual(result.shape, shape)

            with self.assertRaises(IncompatibleDimensions) as cm:
                result = f(x,t)

    def test_ufunc_unary_op(self):
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)
        a = PDA(np.arange(size).reshape(shape),pdim=_angle)
        n = PDA(np.arange(size).reshape(shape),pdim=_dimless)

        result = -x
        self.assertTrue(type(result) is PDA)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.pdim, x.pdim)

        for f in (np.negative, np.positive, np.absolute, np.invert,
                 np.conj, np.conjugate):
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
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)
        n = PDA(np.arange(size).reshape(shape),pdim=_dimless)

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
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)

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

            with self.assertRaises(IncompatibleDimensions) as cm:
                result = f(x,t)

        result = np.heaviside(x,y)
        self.assertEqual(result.shape, v.shape)
        self.assertTrue(type(result) is PDA)
        self.assertTrue(is_number(result))
        self.assertEqual(result.pdim, v.pdim)


    def test_ufunc_mul(self):
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)

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
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)
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

    def test_ufunc_pow(self):
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        z = PDA((np.arange(size,dtype=complex)+.5j).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)
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
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)
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
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)
        t = PDA((np.arange(size)-3).reshape(shape),pdim=_time)
        t.flat[3] = 100  # avoid div by zero

        n = PDA(5,pdim=_length)

        for v in (y,n):
            q,r = divmod(x,y)

            self.assertTrue(type(q) is np.ndarray)
            self.assertTrue(type(r) is PDA)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(r.shape, x.shape)
            self.assertEqual(r.pdim, _length)

        with self.assertRaisesRegex(TypeError,
            "^Dividend and divisor must have same dimensionality"):
            q,r = divmod(x,t)

    def test_ufunc_unallowed(self):
        shape = (2,3)
        size = np.prod(shape)

        x = PDA((np.arange(size)+1).reshape(shape),pdim=_length)
        y = PDA(np.transpose(np.arange(size)+0.5).reshape(shape),pdim=_length)

        for f in (np.logaddexp,np.logaddexp2,):
            with self.assertRaises(UnsupportedUfunc):
                result = f(x,y)

        for f in (np.rint, np.exp, np.exp2, np.log, np.log2, 
                  np.log10, np.expm1, np.log1p):
            with self.assertRaises(UnsupportedUfunc):
                result = f(x)


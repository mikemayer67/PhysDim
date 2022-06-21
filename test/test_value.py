import unittest
import warnings

import numpy as np

from physdim import PhysicalValue as PV
from physdim import PhysicalDimension as Dim 

_length = Dim(length=1)
_time = Dim(time=1)
_density = Dim(mass=1,length=-3)
_angle = Dim(angle=1)
_dimless = Dim()

def is_close(a,b):
    return np.allclose(a.view(np.ndarray), b.view(np.ndarray))

class PhysicalValueTests(unittest.TestCase):

    def test_init_unit(self):
        x = PV()
        self.assertEqual(x.size, 1)
        self.assertTrue(x.flat[0], 1)
        self.assertEqual(x.pdim, _dimless)

        x = PV(_length)
        self.assertEqual(x.size, 1)
        self.assertTrue(x.flat[0], 1)
        self.assertEqual(x.pdim, _length)

        for n in (1, 2.5, 3+4j):
            x = PV(n)
            self.assertEqual(x.size, 1)
            self.assertTrue(x.flat[0], n)
            self.assertEqual(x.pdim, _dimless)

        for n in (1, 2.5, 3+4j):
            for pdim in (_dimless, _length):
                x = PV(n,pdim)
                self.assertEqual(x.size, 1)
                self.assertTrue(x.flat[0], n)
                self.assertEqual(x.pdim, pdim)

        with self.assertRaisesRegex(TypeError,"^Cannot create PhysicalValue for"):
            x = PV('cow')

    def test_init_pdim(self):
        x = PV(_length)
        y = PV(pdim=_length)
        self.assertEqual(x.pdim,y.pdim)

        for n in (1, 2.5, 3+4j):
            for pdim in (_dimless, _length):
                x = PV(n,pdim)
                y = PV(n,pdim=pdim)
                self.assertEqual(x.pdim,y.pdim)

        good_cases = ([1,2,8,9], [8.2, -5.2], [1+2j],
            [[10*r+c for c in range(5)] for r in range(3)],
            )

        for v in ([1,2,8,9],[1+2j],[[1,2],[4,3]]):
            x = PV(v,_density)
            y = PV(v, pdim=_density)
            self.assertEqual(x.pdim,y.pdim)

            nv = np.array(v)
            x = PV(nv,_density)
            y = PV(nv, pdim=_density)
            self.assertEqual(x.pdim,y.pdim)

    def test_init_values(self):
        good_cases = ([1,2,8,9], [8.2, -5.2], [1+2j],
            [[10*r+c for c in range(5)] for r in range(3)],
            )

        for v in good_cases:
            nv = np.array(v)
            x = PV(v)
            self.assertEqual(x.shape, nv.shape)
            self.assertEqual(x.dtype, nv.dtype)
            self.assertEqual(x.pdim, _dimless)

            for pdim in (_dimless, _density):
                nv = np.array(v)
                x = PV(v,pdim)
                self.assertEqual(x.shape, nv.shape)
                self.assertEqual(x.dtype, nv.dtype)
                self.assertEqual(x.pdim, pdim)
                x = PV(pdim,v)
                self.assertEqual(x.shape, nv.shape)
                self.assertEqual(x.dtype, nv.dtype)
                self.assertEqual(x.pdim, pdim)

            with self.assertRaisesRegex(TypeError,"^Cannot specify both"):
                x = PV(5,v,_time)

            with self.assertRaisesRegex(TypeError,"^Can only specify one"):
                x = PV(v,v,_time)

        # temporarily turn of this deprecation warning as we want are intentionally
        #   attempting a bad list with intent of PV constructor failing it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            with self.assertRaisesRegex(TypeError,"^Cannot create PhysicalValue for"):
                x = PV([[1,2,3],[4,5],6])

        with self.assertRaisesRegex(TypeError,"^Cannot create PhysicalValue for"):
            x = PV(['cat',5,3.2])

    def test_init_ndarray(self):
        base_arrays = (
            np.zeros((5,2)),
            np.ones((2,3,4,5),dtype=float),
            np.arange(22).reshape((2,-1)),
            np.array([1+2j, 1-3j, 5.2, 3]),
            )

        for v in base_arrays:
            x = PV(v)
            self.assertEqual(x.shape, v.shape)
            self.assertEqual(x.dtype, v.dtype)
            self.assertTrue(x.base is v)
            self.assertEqual(x.pdim, _dimless)

            for pdim in (_dimless, _density):
                x = PV(v,pdim)
                self.assertEqual(x.shape, v.shape)
                self.assertEqual(x.dtype, v.dtype)
                self.assertTrue(x.base is v)
                self.assertEqual(x.pdim, pdim)
                x = PV(pdim,v)
                self.assertEqual(x.shape, v.shape)
                self.assertEqual(x.dtype, v.dtype)
                self.assertTrue(x.base is v)
                self.assertEqual(x.pdim, pdim)

            with self.assertRaisesRegex(TypeError,"^Cannot specify both"):
                x = PV(5,v,_time)

            with self.assertRaisesRegex(TypeError,"^Cannot specify both"):
                x = PV(v,[1,2,3],_time)

            with self.assertRaisesRegex(TypeError,"^Can only specify one"):
                x = PV(v,base_arrays[0],_time)

    def test_init_array(self):
        base_arrays = (
            np.zeros((5,2)),
            np.ones((2,3,4,5),dtype=float),
            np.arange(22).reshape((2,-1)),
            np.array([1+2j, 1-3j, 5.2, 3]),
            )

        base_physdims = tuple( PV(a,pdim) for a in base_arrays for pdim in (_dimless,_time) )

        for v in base_physdims:
            x = PV(v)
            self.assertEqual(x.shape, v.shape)
            self.assertEqual(x.dtype, v.dtype)
            self.assertTrue(x.base is v.base)
            self.assertEqual(x.pdim, v.pdim)

            for s in (1, 3.0, 1-1j, np.array(-2.1)):
                x = PV(s,v)
                self.assertEqual(x.shape, v.shape)
                self.assertFalse(x.base is v.base)
                self.assertEqual(x.pdim, v.pdim)

            for pdim in (_dimless, _density):
                with self.assertRaisesRegex(TypeError,"^Cannot specify both"):
                    x = PV(v,pdim)

            with self.assertRaisesRegex(TypeError,"^Cannot specify both"):
                x = PV(v,[1,2,3])

            with self.assertRaisesRegex(TypeError,"^Cannot specify both"):
                x = PV(v,base_arrays[0])

            with self.assertRaisesRegex(TypeError,"^Can only specify one"):
                x = PV(v,base_physdims[0])

    def test_init_failure(self):
        with self.assertRaisesRegex(TypeError,"^Unrecognized inputs.*shape"):
            x = PV([1,2,3],shape=(5,2),pdim=_length)
        with self.assertRaisesRegex(TypeError,"^Cannot create PhysicalValue for"):
            x = PV(['cat','dog','pencil'],pdim=_time)

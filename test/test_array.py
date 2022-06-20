import unittest

import numpy as np

from PhysDim import Array as PDA
from PhysDim import Dim 

_length = Dim(length=1)
_time = Dim(time=1)
_density = Dim(mass=1,length=-3)
_angle = Dim(angle=1)
_dimless = Dim()

def is_close(a,b):
    return np.allclose(a.view(np.ndarray), b.view(np.ndarray))

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
        with self.assertRaisesRegex(TypeError,"^Cannot initialize.*with both shape and array"):
            x = PDA([1,2,3],shape=(5,2),pdim=_length)
        with self.assertRaisesRegex(TypeError,"^The input array must be a simple array"):
            x = PDA(['cat','dog','pencil'],pdim=_time)

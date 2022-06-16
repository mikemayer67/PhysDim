import unittest

from PhysDimArray.dim import Dim 

_length = (1,0,0,0,0,0,0)
_time = (0,1,0,0,0,0,0)
_mass = (0,0,1,0,0,0,0)
_angle = (0,0,0,1,0,0,0)

from PhysDimArray.exceptions import IncompatibleDimensionality
from PhysDimArray.exceptions import NotADimObject

class DimTests(unittest.TestCase):
    def test_init_kwargs(self):
        l = Dim(length=1)
        self.assertEqual(l.dim,_length)

        t = Dim(time=1)
        self.assertEqual(t.dim,_time)

        v = Dim(length=1,time=-1)
        v_dim = tuple(l-t for l,t in zip(_length,_time))
        self.assertEqual(v.dim, v_dim)

    def test_init_tuple(self):
        l = Dim(_length)
        self.assertEqual(l.dim,_length)

        t = Dim(_time)
        self.assertEqual(t.dim,_time)

        v_dim = tuple(l-t for l,t in zip(_length,_time))
        v = Dim(v_dim)
        self.assertEqual(v.dim, v_dim)

    def test_init_copy(self):
        l = Dim(_length)
        x = Dim(l)
        self.assertEqual(l.dim,x.dim)

    def test_bad_init(self):
        with self.assertRaises(NotADimObject):
            v = Dim('cat')
        with self.assertRaises(NotADimObject):
            v = Dim((1, 2, 0))
        with self.assertRaises(NotADimObject):
            v = Dim((1, 2, 0, 'cat'))
        with self.assertRaises(TypeError):
            v = Dim(_length, length=1)

    def test_dimensionless(self):
        l = Dim(_length)
        self.assertFalse(l.dimensionless)
        x = Dim()
        self.assertTrue(x.dimensionless)
        x = l/l
        self.assertTrue(x.dimensionless)

    def test_assert_compatible(self):
        l = Dim(_length)
        t = Dim(_time)
        v = l/t
        with self.assertRaises(IncompatibleDimensionality) as cm:
            v.assert_compatible(l)

    def test_mul(self):
        l = Dim(_length)
        t = Dim(_time)
        lt = l * t
        self.assertEqual(lt.dim, tuple(a+b for a,b in zip(_length,_time)))

        with self.assertRaises(NotADimObject) as cm:
            lx = l * 5
        with self.assertRaises(NotADimObject) as cm:
            lx = 5 * l

    def test_div(self):
        l = Dim(_length)
        t = Dim(_time)
        lt = l / t
        self.assertEqual(lt.dim, tuple(a-b for a,b in zip(_length,_time)))

        with self.assertRaises(NotADimObject) as cm:
            lx = l / 5
        with self.assertRaises(NotADimObject) as cm:
            lx = 5 / l

    def test_pow(self):
        l = Dim(_length) / Dim(_time)
        ll = l**3
        self.assertEqual(ll.dim,tuple(3*(a-b) for a,b in zip(_length,_time)))
        ll = l**3.2
        self.assertEqual(ll.dim,tuple(3.2*(a-b) for a,b in zip(_length,_time)))

    def test_derived_dims(self):
        l = Dim(_length)
        t = Dim(_time)
        m = Dim(_mass)
        q = Dim(_angle)
        c = Dim((0,0,0,0,1,0,0))
        k = Dim((0,0,0,0,0,1,0))
        i = Dim((0,0,0,0,0,0,1))
        self.assertEqual(str(l), "length")
        self.assertEqual(str(t), "time")
        self.assertEqual(str(m), "mass")
        self.assertEqual(str(q), "angle")
        self.assertEqual(str(c), "charge")
        self.assertEqual(str(k), "temperature")
        self.assertEqual(str(i), "illumination")
        self.assertEqual(str(l*l), "area")
        self.assertEqual(str(l**3), "volume")
        self.assertEqual(str(q**2), "solid angle")
        self.assertEqual(str(t**-1), "frequency")
        self.assertEqual(str(m/l**3), "density")
        self.assertEqual(str(l/t), "velocity")
        self.assertEqual(str(q/t), "angular velocity")
        self.assertEqual(str(l/t**2), "acceleration")
        self.assertEqual(str(m*l/t), "momentum")
        self.assertEqual(str(m*l/t**2), "force")
        self.assertEqual(str(m*l**2/t**2), "energy")
        self.assertEqual(str(m*l**2/t**3), "power")
        self.assertEqual(str(m/(l*t**2)), "pressure")


       





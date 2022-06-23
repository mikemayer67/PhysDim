import unittest

from physdim import PhysicalDimension as Dim 

_mass = (1,0,0,0,0,0,0)
_length = (0,1,0,0,0,0,0)
_time = (0,0,1,0,0,0,0)
_angle = (0,0,0,1,0,0,0)

from physdim.exceptions import NotDimLike

class PhysicalDimTests(unittest.TestCase):
    def test_init_kwargs(self):
        l = Dim(length=1)
        self.assertEqual(l._exp,_length)

        t = Dim(time=1)
        self.assertEqual(t._exp,_time)

        v = Dim(length=1,time=-1)
        v_dim = tuple(l-t for l,t in zip(_length,_time))
        self.assertEqual(v._exp, v_dim)

    def test_init_tuple(self):
        l = Dim(_length)
        self.assertEqual(l._exp,_length)

        t = Dim(_time)
        self.assertEqual(t._exp,_time)

        v_dim = tuple(l-t for l,t in zip(_length,_time))
        v = Dim(v_dim)
        self.assertEqual(v._exp, v_dim)

        v = Dim([0,1,-1,0,0,0,0])
        self.assertEqual(v._exp, v_dim)

    def test_init_copy(self):
        l = Dim(_length)
        x = Dim(l)
        self.assertEqual(l._exp,x._exp)

    def test_bad_init(self):
        with self.assertRaises(NotDimLike):
            v = Dim('cat')
        with self.assertRaises(NotDimLike):
            v = Dim((1, 2, 0))
        with self.assertRaises(NotDimLike):
            v = Dim((1, 2, 0, 'cat'))
        with self.assertRaisesRegex(TypeError,
            "^Exponents on dimensions must all be real numbers"):
            v = Dim((1,2,3,4.0,5+1j,6,7))
        with self.assertRaisesRegex(TypeError,
            "^Cannot specify both"):
            v = Dim(_length, length=1)

    def test_dimensionless(self):
        l = Dim(_length)
        self.assertFalse(l.is_dimensionless)
        x = Dim()
        self.assertTrue(x.is_dimensionless)
        x = l/l
        self.assertTrue(x.is_dimensionless)

    def test_inverse(self):
        l = Dim(length=1, time=-1)
        li = l.inverse
        self.assertEqual(li, Dim(length=-1,time=1))

    def test_is_angle(self):
        l = Dim(_length)
        t = Dim(_time)
        th = Dim(_angle)
        x = Dim()
        v = l/t

        self.assertFalse(l.is_angle)
        self.assertTrue(th.is_angle)
        self.assertTrue(x.is_angle)
        self.assertFalse((th*th).is_angle)
        self.assertTrue((l/l).is_angle)

    def test_mul(self):
        l = Dim(_length)
        t = Dim(_time)
        lt = l * t
        self.assertEqual(lt._exp, tuple(a+b for a,b in zip(_length,_time)))

        with self.assertRaises(NotDimLike):
            lx = l * 5
        with self.assertRaises(NotDimLike):
            lx = 5 * l

    def test_div(self):
        l = Dim(_length)
        t = Dim(_time)
        lt = l / t
        self.assertEqual(lt._exp, tuple(a-b for a,b in zip(_length,_time)))

        with self.assertRaises(NotDimLike):
            lx = l / 5
        with self.assertRaises(NotDimLike):
            lx = 5 / l

    def test_pow(self):
        l = Dim(_length) / Dim(_time)
        ll = l**3
        self.assertEqual(ll._exp,tuple(3*(a-b) for a,b in zip(_length,_time)))
        ll = l**3.2
        self.assertEqual(ll._exp,tuple(3.2*(a-b) for a,b in zip(_length,_time)))

    def test_dim_types(self):
        l = Dim(_length)
        t = Dim(_time)
        m = Dim(_mass)
        q = Dim(_angle)
        c = Dim((0,0,0,0,1,0,0))
        k = Dim((0,0,0,0,0,1,0))
        i = Dim((0,0,0,0,0,0,1))

        self.assertEqual((m).type, "mass")
        self.assertEqual((l).type, "length")
        self.assertEqual((t).type, "time")
        self.assertEqual((q).type, "angle")
        self.assertEqual((c).type, "charge")
        self.assertEqual((k).type, "temperature")
        self.assertEqual((i).type, "illumination")
        self.assertEqual((l*l).type, "area")
        self.assertEqual((l**3).type, "volume")
        self.assertEqual((q**2).type, "solid angle")
        self.assertEqual((t**-1).type, "frequency")
        self.assertEqual((m/l**3).type, "density")
        self.assertEqual((l/t).type, "velocity")
        self.assertEqual((q/t).type, "angular velocity")
        self.assertEqual((l/t**2).type, "acceleration")
        self.assertEqual((m*l/t).type, "momentum")
        self.assertEqual((m*l/t**2).type, "force")
        self.assertEqual((m*l**2/t**2).type, "energy or torque")
        self.assertEqual((m*l**2/t**3).type, "power")
        self.assertEqual((m/(l*t**2)).type, "pressure")

    def test_dim_string(self):
        m = Dim(_mass)
        l = Dim(_length)
        t = Dim(_time)
        q = Dim(_angle)
        c = Dim((0,0,0,0,1,0,0))
        k = Dim((0,0,0,0,0,1,0))
        i = Dim((0,0,0,0,0,0,1))

        self.assertEqual(str(m), "[M]")
        self.assertEqual(str(l), "[L]")
        self.assertEqual(str(t), "[T]")
        self.assertEqual(str(q), "[A]")
        self.assertEqual(str(c), "[C]")
        self.assertEqual(str(k), "[K]")
        self.assertEqual(str(i), "[I]")
        self.assertEqual(str(l*l), "[L^2]")
        self.assertEqual(str(l**3), "[L^3]")
        self.assertEqual(str(q**2), "[A^2]")
        self.assertEqual(str(t**-1), "[1/T]")
        self.assertEqual(str(m/l**3), "[M/L^3]")
        self.assertEqual(str(l/t), "[L/T]")
        self.assertEqual(str(q/t), "[A/T]")
        self.assertEqual(str(l/t**2), "[L/T^2]")
        self.assertEqual(str(m*l/t), "[M L/T]")
        self.assertEqual(str(m*l/t**2), "[M L/T^2]")
        self.assertEqual(str(m*l**2/t**2), "[M L^2/T^2]")
        self.assertEqual(str(m*l**2/t**3), "[M L^2/T^3]")
        self.assertEqual(str(m/(l*t**2)), "[M/L T^2]")

    def test_dim_base_units(self):
        m = Dim(_mass)
        l = Dim(_length)
        t = Dim(_time)
        q = Dim(_angle)
        c = Dim((0,0,0,0,1,0,0))
        k = Dim((0,0,0,0,0,1,0))
        i = Dim((0,0,0,0,0,0,1))

        base_units = ('kg','m','sec','rad','coul')

        self.assertEqual((m).to_str(base_units), "[kg]")
        self.assertEqual((l).to_str(base_units), "[m]")
        self.assertEqual((t).to_str(base_units), "[sec]")
        self.assertEqual((q).to_str(base_units), "[rad]")
        self.assertEqual((c).to_str(base_units), "[coul]")
        self.assertEqual((k).to_str(base_units), "[K]")
        self.assertEqual((i).to_str(base_units), "[I]")
        self.assertEqual((l*l).to_str(base_units), "[m^2]")
        self.assertEqual((l**3).to_str(base_units), "[m^3]")
        self.assertEqual((q**2).to_str(base_units), "[rad^2]")
        self.assertEqual((t**-1).to_str(base_units), "[1/sec]")
        self.assertEqual((m/l**3).to_str(base_units), "[kg/m^3]")
        self.assertEqual((l/t).to_str(base_units), "[m/sec]")
        self.assertEqual((q/t).to_str(base_units), "[rad/sec]")
        self.assertEqual((l/t**2).to_str(base_units), "[m/sec^2]")
        self.assertEqual((m*l/t).to_str(base_units), "[kg m/sec]")
        self.assertEqual((m*l/t**2).to_str(base_units), "[kg m/sec^2]")
        self.assertEqual((m*l**2/t**2).to_str(base_units), "[kg m^2/sec^2]")
        self.assertEqual((m*l**2/t**3).to_str(base_units), "[kg m^2/sec^3]")
        self.assertEqual((m/(l*t**2)).to_str(base_units), "[kg/m sec^2]")



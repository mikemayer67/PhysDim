import unittest

import numpy as np

from physdim import Units
from physdim import PhysicalValue as Value
from physdim import PhysicalDimension as Dim
from physdim.exceptions import AttemptToRedefineUnit
from physdim.exceptions import AttemptToAssignToUnit
from physdim.exceptions import UndefinedUnit

class UnitsTests(unittest.TestCase):
    def setUp(self):
        # need to reset Units singleton between tests, but as this isn't
        #  supported by the Units class, we need to do a little hackery here
        Units._Instance = None

    def test_add_unit(self):
        u = Units()
        # junk = m/rad^2
        junk = u.add('junk',length=1,angle=-2)
        self.assertEqual(junk, u.m/u.rad**2)
        self.assertEqual(u.junk, junk)
        # stuff = 5.2 junk
        stuff = u.add('stuff',5.2, length=1,angle=-2)
        self.assertEqual(stuff, 5.2*u.m/u.rad**2)
        self.assertEqual(u.stuff, stuff)

    def test_add_prefixable_unit(self):
        u = Units()
        # junk = m/rad^2
        junk = u.add('junk',length=1,angle=-2,can_scale=True)
        self.assertEqual(u.junk, junk)
        self.assertEqual(junk, u.m/u.rad**2)
        self.assertEqual(u.junk, 1000*u.mjunk)
        self.assertEqual(u.kjunk, 1000*u.junk)

    def test_add_scaled_unit(self):
        u = Units()
        # fig = 5.2 newtons
        fig = u.add('fig', 5.2, u.N)
        self.assertEqual(u.fig, fig)
        self.assertEqual(u.fig, 5.2*u.N)
        # prune = 10 figs
        prune = u.add('prune', 10*u.fig)
        self.assertEqual(u.prune, prune)
        self.assertEqual(u.prune, 10*u.fig)
        self.assertEqual(u.prune, 52*u.N)

    def test_predefined_units(self):
        u = Units()
        self.assertEqual(u.m, Value(1, Dim(length=1)))
        self.assertEqual(u.kg, Value(1, Dim(mass=1)))
        self.assertEqual(u.s, Value(1, Dim(time=1)))
        self.assertEqual(u.C, Value(1, Dim(charge=1)))
        self.assertEqual(u.K, Value(1, Dim(temp=1)))
        self.assertEqual(u.cd, Value(1, Dim(illum=1)))
        self.assertEqual(u.rad, Value(1, Dim(angle=1)))

        self.assertEqual(u.g, u.kg/1000)
        self.assertEqual(u.sec, u.s)
        self.assertEqual(u.coul, u.C)

        self.assertEqual(u.deg, u.rad*np.pi/180)
        self.assertEqual(u.sr, u.rad**2)

        self.assertEqual(u.min, 60*u.sec)
        self.assertEqual(u.hr, 3600*u.sec)
        self.assertEqual(u.day, 86400*u.sec)
        self.assertEqual(u.week, 604800*u.sec)

        self.assertEqual(u.tonne, 1000*u.kg)
        self.assertEqual(u.l, 1000*u.ml)
        self.assertEqual(u.litre, u.l)
        self.assertEqual(u.litre, u.liter)

        self.assertEqual(u.Hz*u.sec, 1)
        self.assertEqual(u.N, u.m*u.kg/u.sec**2)
        self.assertAlmostEqual(u.dyne/u.N, 1.e-5)
        self.assertEqual(u.J, u.kg*u.m**2/u.sec**2)
        self.assertAlmostEqual(u.erg/u.J, 1.e-7)
        self.assertEqual(u.W, u.kg*u.m**2/u.sec**3)
        self.assertEqual(u.Pa, u.kg/(u.m*u.sec**2))
        self.assertEqual(u.coul, u.A*u.sec)
        self.assertEqual(u.J, u.V*u.coul)
        self.assertEqual(u.coul, u.F*u.V)
        self.assertEqual(u.V, u.A*u.ohm)
        self.assertEqual(u.lm, u.cd*u.sr)
        self.assertEqual(u.lm, u.lx*u.m**2)

        self.assertEqual(u.hertz, u.Hz)
        self.assertEqual(u.newton, u.N)
        self.assertEqual(u.joule, u.J)
        self.assertEqual(u.watt, u.W)
        self.assertEqual(u.pascal, u.Pa)
        self.assertEqual(u.amp, u.A)
        self.assertEqual(u.volt, u.V)
        self.assertEqual(u.farad, u.F)
        self.assertEqual(u.ohm, u.Ohm)
        self.assertEqual(u.lumen, u.lm)
        self.assertEqual(u.lux, u.lx)

        self.assertEqual(u.inch, 2.54*u.cm)
        self.assertEqual(u.ft, 12*u.inch)
        self.assertEqual(u.yd, 3*u.ft)
        self.assertEqual(u.mi, 1760*u.yd)
        self.assertEqual(u.league, 3*u.mi)
        self.assertEqual(u.nmi, 1852*u.m)
        self.assertEqual(u.acre, 4840*u.yd**2)
        self.assertEqual(u.pt, 28.875*u.inch**3)
        self.assertEqual(u.qt, 2*u.pt)
        self.assertEqual(u.gal, 4*u.qt)
        self.assertEqual(u.pt, 16*u.fl_oz)
        self.assertEqual(u.oz, 28.375*u.g)
        self.assertEqual(u.lb, 16*u.oz)
        self.assertEqual(u.ton, 2000*u.lb)
        self.assertEqual(u.lbf, 4.44822162*u.N)
        self.assertEqual(u.BTU, 1055.05585*u.J)        
        self.assertEqual(u.hp, 745.699872*u.W)

        self.assertEqual(u.foot, u.ft)
        self.assertEqual(u.feet, u.ft)
        self.assertEqual(u.yard, u.yd)
        self.assertEqual(u.yards, u.yd)
        self.assertEqual(u.mile, u.mi)
        self.assertEqual(u.miles, u.mi)
        self.assertEqual(u.pint, u.pt)
        self.assertEqual(u.quart, u.qt)
        self.assertEqual(u.gallon, u.gal)

    def test_unit_lock(self):
        u = Units()
        with self.assertRaises(AttemptToAssignToUnit):
            u.m = u.kg
        with self.assertRaises(AttemptToAssignToUnit):
            u.junk = u.kg

    def test_unit_stability(self):
        u = Units()
        u.add('junk',u.m/u.sec)
        # cannot change base units
        with self.assertRaises(AttemptToRedefineUnit):
            u.add('junk',u.m/u.sec**2)
        # cannot change prefixability
        with self.assertRaises(AttemptToRedefineUnit):
            u.add('junk',u.m/u.sec,can_prefix=True)

    def test_undefined_units(self):
        u = Units()
        # completely bogus unit
        with self.assertRaises(UndefinedUnit):
            x = u.junk
        # attempting to prefix unprefixable unit
        with self.assertRaises(UndefinedUnit):
            x = u.kmile


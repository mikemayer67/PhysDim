"""Physical Constants

Singleton slass for accessing a handful of useful physical constants.

The physdim.Constants singleton can be accessed either through the constructor
or by simply using the instance provided by the physdim module.
The following two snippets have the identical result of assigning
the physdim.Constants singleton to k.

    ``
    from physdim.units import Constants
    k = Constants()
    ``

    or

    ``
    from physdim import constants as k
    ``
"""

from .units import Units
from .exceptions import AttemptToRedefineConstant

import numpy as np

class Constants (object):
    # This is a singleton class.  Constants._Instance is that singleton instance
    _Instance = None
    __slots__ = ('initialized','__dict__')

    def __new__(cls):
        if cls._Instance is None:
            k = super(Constants,cls).__new__(cls)
            k.initialized = False
            cls._Instance = k
        return cls._Instance

    def __init__(self):
        if not self.initialized:
            self.add_math_constants()
            self.add_phys_constants()
            self.add_earth_constants()
            self.initialized = True

    def add_math_constants(self):
        self.pi = np.pi
        # base of natural log
        self.e = np.e
        self.euler = np.e
        # golden ratio
        self.phi = (1 + np.sqrt(5))/2
        self.dozen = 12
        self.gross = 144

    def add_phys_constants(self):
        u = Units()
        # Source: https://en.wikipedia.org/wiki/List_of_physical_constants
        # speed of light in vacuum
        self.c = 299792458 * (u.m/u.sec)
        self.speed_of_light = self.c
        # gravitational constant
        self.G = 6.67430e-11 * (u.m*3/(u.kg*u.sec**2))
        self.gravitational_constant = self.G
        # Planck constant
        self.h = 6.6260715e-34 * (u.J*u.sec)
        self.hbar = self.h/(2*np.pi)
        self.planck_constant = self.h
        self.reduced_planck_constant = self.hbar
        # Botlzmann constant
        self.k = 1.380649e-12 * (u.J/u.K)
        self.boltzmann_constant = self.k
        # Avogradro number
        self.NA = 6.02214076e23
        self.avogadro = self.NA
        # Pressure
        self.atm = 101325 * u.Pa
        self.standard_atmosphere = self.atm
        # Gravitational acceleration
        self.g = 9.80665 * (u.m/u.s**2)
        self.standard_acceleration_gravity = self.g

    def add_earth_constants(self):
        u = Units()
        # Source: https://en.wikipedia.org/wiki/Earth_physical_characteristics_tables
        self.earth_equatorial_radius = (12756.3/2) * u.km
        self.earth_polar_radius = 6356.89 * u.km
        self.earth_mass = 5.98e24 * u.kg

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise AttemptToRedefineConstant(name)
        super().__setattr__(name, value)




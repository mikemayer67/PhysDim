"""Phyaical Units

Singleton class for accessing and defining base units. 

The physdim.Units singleton can be accessed either through the constructor
or by simply using the instance provided by the physdim module.
The following two snippets have the identical result of assigning
the physdim.Units singleton to u.

    ``
    from physdim.units import Units
    u = Units()
    ``

    or

    ``
    from physdim import units as u
    ``

The attributes of a Units instance are the currently defined
physical unit quantities.  

Example:
    u = Units()
    x = 3 * u.miles
    y = 4.2 * u.km
    KE = 15 * u.J
    torque = 52 * u.ft * u.lbf

Each unit is an instance of physdim.PhysicalValue where 
numeric values of 1 are designed to be modified MKS units:
    - Length: meters
    - Time: seconds
    - Mass: kilograms(*)
    - Angle: radians
    - Electric charge: coulomb
    - Absolute temperature: Kelvin
    - Intensity of Light: candela
* Note that this will cause issues with parsing of SI prefixes
of mass units.

A subset of the units can be prefixed with SI style modifiers

    x = 3 * u.km
    y = 3000 * u.m

Additional units may be defined using the add method
"""

from .value import PhysicalValue
from .dim import PhysicalDimension

from .exceptions import AttemptToRedefineUnit
from .exceptions import AttemptToAssignToUnit
from .exceptions import UndefinedUnit

import numpy as np
from numbers import Number

# Because this is a singleton class, we have the option of putting all
#   of the properties either at file scope or as class attributes.
# We are going with the former.
_FundamentalUnits = ('kg','m','sec','rad','coul','K','cd')

# define constants for each of the fundamental dimensions
_MassDim   = PhysicalDimension(mass=1)
_LengthDim = PhysicalDimension(length=1)
_TimeDim   = PhysicalDimension(time=1)
_AngleDim  = PhysicalDimension(angle=1)
_ChargeDim = PhysicalDimension(charge=1)
_TempDim   = PhysicalDimension(temp=1)
_IllumDim  = PhysicalDimension(illum=1)

# Most SI units allow for scaling via prefixes.
#   The recognized ones are as follow.
#
# Note that centi(c) is left off the list as length is the only dimension 
#   that commonly uses it. 
# Note too that we will need to uniquely handle the prefixes of mass 
#   as the fundamental unit is kg rather than g.
_Prefixes   = ('f','p','n','u','m',None,'k','M','G','T','P','E')
_PrefixMags = tuple(-15 + 3*i for i in range(len(_Prefixes)))
_PrefixScales = { n: 10**m for n,m in zip(_Prefixes,_PrefixMags) if m}
_PrefixLookup = { m: n for n,m in zip(_Prefixes,_PrefixMags) if m}
_PrefixLimits = ( min(_PrefixLookup.keys()), max(_PrefixLookup.keys()) )

class Units (object):
    # This is a singleton class.  Units._Instance is that singleton instance
    _Instance = None

    # We want the content of __dict__ to be exclusively the defined units.
    #   Everything else will be stored explicitly in attribute slots:
    #   
    #   scalable is a dictionary of all scalable units
    #       key: the name of the unit
    #       value: tuple containing scalability limits
    #
    #   base_units is a dictionary of the defined units to be used
    #     when formatting a value into a string
    #       key: PhysicalDimension
    #       value: name of the base unit
    __slots__ = ('initialized','scalable','base_units','__dict__')

    def __new__(cls):
        if cls._Instance is None:
            units = super(Units,cls).__new__(cls)
            units.initialized = False
            units.scalable = dict()
            units.base_units = dict()
            cls._Instance = units
        return cls._Instance

    def __init__(self):
        if not self.initialized:
            self.add_base_units()
            self.add_angle_units()
            self.add_time_units()
            self.add_mks_units()
            self.add_imperial_units()
            self.initialized = True

    @property
    def defined(self):
        """list of all currently defined units"""
        return tuple(self.__dict__.keys())

    def add(self, name, *args, can_scale=False, scale_limits=None, **kwargs):
        # Parse the name, args, and kwargs to construct the PhysicalValue
        #   associated with this unit

        assert len(args) < 3
        if len(args) == 0:
            # only a name was provided via the args, the dimension (if any)
            #   must be defined via kwargs
            value = PhysicalValue(1,PhysicalDimension(**kwargs))
        elif len(args) == 2:
            # two arguments follow name---use them to construct a PhysicalValue
            #   Physicalvalue constructor will handle their validity
            value = PhysicalValue(*args)
        elif isinstance(args[0],PhysicalValue):
            # a physical value was provided, the work has been done for us
            value = args[0]
        elif isinstance(args[0],PhysicalDimension):
            # a physical dimension was provided, create the corresponding unit value
            value = PhysicalValue(1,args[0])
        else:
            # a magnitude was provided with dimensions, the dimension (if any)
            #  must be defined via kwargs
            value = PhysicalValue(args[0],PhysicalDimension(**kwargs))

        if name in self.__dict__:
            raise AttemptToRedefineUnit(name)
        else:
            self.__dict__[name] = value

        # Parse the scaling args
        if scale_limits:
            self.scalable[name] = scale_limits
        elif can_scale:
            self.scalable[name] = True

        # If this is the first unit with this physical dimensionality
        #   make it the base unit for that dimension
        if value.pdim not in self.base_units:
            self.base_units[value.pdim] = name

        return value

    def add_base_units(self):
        self.add('kg',mass=1)
        self.add('m',length=1, scale_limits=(None,3))
        self.add('sec',time=1, scale_limits=(None,0))
        self.add('rad',angle=1, scale_limits=(-3,0))
        self.add('C',charge=1)
        self.add('K',temp=1)
        self.add('cd',illum=1)

        # synonyms
        self.add('g',0.001,self.kg, scale_limits=(None,3))
        self.add('cm',0.01,self.m)
        self.add('s',self.sec)
        self.add('coul',self.C)

    def add_angle_units(self):
        self.add('deg',np.pi/180,self.rad)
        self.add('cycle',360,self.deg)
        self.add('sr',self.rad**2)

    def add_time_units(self):
        self.add('min',60,self.sec)
        self.add('hr',60,self.min)
        self.add('day',24,self.hr)
        self.add('week',7,self.day)

    def add_mks_units(self):
        # mass
        self.add('tonne',1000,self.kg)
        self.add('Tonne',self.tonne, can_scale=True)

        # volumne
        self.add('ml', self.cm**3)
        self.add('l', 1000, self.ml,  can_scale=True)

        self.add('litre',self.l)
        self.add('liter',self.l)

        # derivative units
        self.add('Hz',1/self.sec, can_scale=True)
        self.add('N',self.kg*self.m/self.sec**2, can_scale=True)
        self.add('dyne',self.g*self.cm/self.sec**2)
        self.add('J',self.N*self.m, can_scale=True)
        self.add('erg',self.dyne*self.cm)
        self.add('cal',4.184, self.J, can_scale=True)
        self.add('W',self.J/self.s, can_scale=True)
        self.add('Pa',self.N/self.m**2, can_scale=True)
        self.add('A',self.coul/self.sec, can_scale=True)
        self.add('V',self.J/self.coul, can_scale=True)
        self.add('F',self.coul/self.V, can_scale=True)
        self.add('Ohm',self.V/self.A, can_scale=True)
        self.add('lm',self.cd*self.sr)
        self.add('lx',self.lm/self.m**2)

        self.add('hertz',self.Hz)
        self.add('newton',self.N)
        self.add('joule',self.J)
        self.add('watt',self.W)
        self.add('pascal',self.Pa)
        self.add('amp',self.A)
        self.add('volt',self.V)
        self.add('farad',self.F)
        self.add('ohm',self.Ohm)
        self.add('lumen',self.lm)
        self.add('lux',self.lx)

    def add_imperial_units(self):
        # length
        self.add('inch', 2.54, self.cm)
        self.add('ft', 12, self.inch)
        self.add('foot', self.ft)
        self.add('feet', self.ft)
        self.add('yd', 3, self.ft)
        self.add('yard', self.yd)
        self.add('yards', self.yd)
        self.add('mi', 1760, self.yd)
        self.add('mile', self.mi)
        self.add('miles', self.mi)
        self.add('league', 3, self.mi)
        self.add('nmi', 1852, self.m)

        # area
        self.add('acre',4840, self.yd**2)

        # volume
        self.add('pt',28.875, self.inch**3)
        self.add('qt', 2, self.pt)
        self.add('gal', 4, self.qt)
        self.add('fl_oz', self.pt/16)

        self.add('pint',self.pt)
        self.add('quart',self.qt)
        self.add('gallon',self.gal)

        # weight
        self.add('oz', 28.375, self.g)
        self.add('lb', 16, self.oz)
        self.add('ton', 2000, self.lb)
        self.add('Ton', self.ton, can_scale=True)

        # force, energy, power
        self.add('lbf', 4.44822162, self.N)
        self.add('BTU', 1055.05585, self.J)        
        self.add('hp', 745.699872, self.W)

    def best_unit(self,value):
        try:
            # mass is a special case due to kg as fundamental unit
            if value.pdim == _MassDim:
                unit_name = 'g'
                unit = self.g
                scale_factor = 1000
            else:
                unit_name = self.base_units[value.pdim]
                unit = self.__dict__[unit_name]
                scale_factor = 1

            # add a prefix if appropriate
            if unit_name in self.scalable:
                value_adj = np.abs(value/unit)
                # exclude any value of 0
                value_mag = np.log10(value_adj[value_adj>0])
                value_mag = np.average(value_mag)
                value_mag = 3 * int(value_mag//3)

                value_mag = max(value_mag, _PrefixLimits[0])
                value_mag = min(value_mag, _PrefixLimits[1])

                mag_limits = self.scalable[unit_name] 
                if type(mag_limits) is tuple:
                    (mag_min,mag_max) = mag_limits
                    if mag_min is not None:
                        value_mag = max(value_mag,mag_min)
                    if mag_max is not None:
                        value_mag = min(value_mag,mag_max)

                if value_mag:
                    prefix = _PrefixLookup[value_mag]
                    unit_name = prefix + unit_name
                    scale_factor *= 10**(-value_mag)

        except:
            unit_name = value.pdim.to_str(_FundamentalUnits)
            scale_factor = 1

        return ( unit_name, scale_factor )

    def __setattr__(self, name, value):
        if name not in Units.__slots__:
            raise AttemptToAssignToUnit(name)
        super().__setattr__(name, value)


    def __getattr__(self, name):
        # unit not in dictionary, see if we can create it from
        #   a prefixable unit
        try:
            assert len(name) > 1
            prefix, base_unit = (name[:1], name[1:])
            assert base_unit in self.scalable
            base_unit = self.__dict__[base_unit]
            scale_factor = _PrefixScales[prefix]
            return self.add(name, scale_factor, base_unit)
        except:
            raise UndefinedUnit(name)

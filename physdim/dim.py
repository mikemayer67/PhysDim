"""Physical Dimensionality

This class captures the physical dimensionality of a measurable quantity
without scale.  It does so by tracking the relevant exponents on 7 
fundamental physical properties:

    - Mass (M)
    - Length (L)
    - Time (T)
    - Angle (A)
    - Electric charge (E)
    - Absolute temperature (K)
    - Intensity of light (I)

Internally, it stores these exponents as a tuple in the order (L,T,M,A,E,K,I).
"""

import numpy as np
import numbers

from .exceptions import IncompatibleDimensions
from .exceptions import NotDimLike

_DERIVED_DIMS = {
    (1,0,0,0,0,0,0):'mass',
    (0,1,0,0,0,0,0):'length',
    (0,0,1,0,0,0,0):'time',
    (0,0,0,1,0,0,0):'angle',
    (0,0,0,0,1,0,0):'charge',
    (0,0,0,0,0,1,0):'temperature',
    (0,0,0,0,0,0,1):'illumination',
    (0,2,0,0,0,0,0):'area',
    (0,3,0,0,0,0,0):'volume',
    (0,0,0,2,0,0,0):'solid angle',
    (0,0,-1,0,0,0,0):'frequency',
    (1,-3,0,0,0,0,0):'density',
    (0,1,-1,0,0,0,0):'velocity',
    (0,0,-1,1,0,0,0):'angular velocity',
    (0,1,-2,0,0,0,0):'acceleration',
    (1,1,-1,0,0,0,0):'momentum',
    (1,1,-2,0,0,0,0):'force',
    (1,2,-2,0,0,0,0):'energy or torque',
    (1,2,-3,0,0,0,0):'power',
    (1,-1,-2,0,0,0,0):'pressure',
}


class PhysicalDimension(object):
    __slot__ = ('_exp')

    def __init__(self, dim=None, *, 
                 length=0, time=0, mass=0, angle=0, charge=0, temp=0, illum=0 ):
        """PhysicalDimension constructor

        A PhysicalDimension object can be contructed by one of three methods:

            - specify the exponents for each of the fundamental properties:
              - mass (number): optional 
              - length (number): optional 
              - time (number): optional 
              - angle (number): optional 
              - charge (number): optional 
              - temp (number): optional 
              - illum (number): optional 

            - specify a list of the exponents on the 7 fundamental properties:
              - dim (tuple): (mass, length, time, angle, charge, temp, illum)

            - copy constructor
              - dim (PhysicalDimension): existing PhysicalDimension object
        """
        arg_exp = (mass,length,time,angle,charge,temp,illum)
        if dim is not None:
            if np.any(arg_exp):
                raise TypeError("Cannot specify both the dim tuple and individual exponents")
            if type(dim) is PhysicalDimension:
                # assume that if dim is already a PhysicalDimension, we don't need to check the exponent types
                exp = dim._exp
            else:
                # Try to convert it to a tuple and verify that the tuple has 7 entries
                try:
                    exp = tuple(dim)
                    assert len(exp) == 7
                except:
                    raise NotDimLike(dim)
                # Verify that all 7 entries are floating point numbers (and not complex)
                if not np.all([isinstance(x,numbers.Real) for x in dim]):
                    raise TypeError(f"Exponents on dimensions must all be real numbers, not {dim}")
        else:
            # dim not specified, so we're using the kwargs arguments ( after verifying them )
            keys = ('mass','length','time','angle','charge','temp','illum')
            for k,v in zip(keys,arg_exp):
                if not isinstance(v,numbers.Real):
                    raise TypeError(f"Exponent on {k} must be a real number, not {v}")
            exp = arg_exp 

        self._exp = exp

    def __hash__(self):
        return hash(self._exp)

    @property
    def is_dimensionless(self):
        return not np.any(self._exp)

    @property
    def is_angle(self):
        return self._exp in ((0,0,0,1,0,0,0),(0,0,0,0,0,0,0))

    @property
    def inverse(self):
        return PhysicalDimension(tuple(-t for t in self._exp))

    def __eq__(self,other):
        try:
            return self._exp == other._exp
        except:
            return False

    def __mul__(self,other):
        try:
            return PhysicalDimension(tuple(a+b for a,b in zip(self._exp,other._exp)))
        except:
            raise NotDimLike(other)

    __rmul__ = __mul__

    def __truediv__(self,other):
        try:
            return PhysicalDimension(tuple(a-b for a,b in zip(self._exp,other._exp)))
        except:
            raise NotDimLike(other)

    def __rtruediv__(self,other):
        try:
            return PhysicalDimension(tuple(b-a for a,b in zip(self._exp,other._exp)))
        except:
            raise NotDimLike(other)

    def __pow__(self,n):
        return PhysicalDimension(tuple(a*n for a in self._exp))

    def __repr__(self):
        keys = ('mass','length','time','angle','charge','temp','illum')
        return (f"PhysicalDimension("
                + ",".join(f"{n}={e}" for n,e in zip(keys,self._exp) if e)
                + ",)")

    def __str__(self):
        return self.to_str()

    def to_str(self,names=()):
        default_names = ('M','L','T','A','C','K','I')
        names = names + default_names[len(names):]

        num = " ".join(f"{n}" if e==1 else f"{n}^{e}" for n,e in
                       zip(names,self._exp) if e>0)
        den = " ".join(f"{n}" if e==-1 else f"{n}^{-e}" for n,e in
                       zip(names,self._exp) if e<0)

        if num and den:
            return f"[{num}/{den}]"
        elif num:
            return f"[{num}]"
        elif den:
            return f"[1/{den}]"
        else:
            return f"[]"

    @property
    def type(self):
        return _DERIVED_DIMS.get(self._exp, str(self))

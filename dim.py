"""Physical Dimensionality

This class captures the physical dimensionality of a measurable quantity
without scale.  It does so by tracking the relevant exponents on 7 
fundamental physical properties:

    - Length (L)
    - Time (T)
    - Mass (M)
    - Angle (A)
    - Electric charge (E)
    - Absolute temperature (K)
    - Intensity of light (I)

Internally, it stores these exponents as a tuple in the order (L,T,M,A,E,K,I).
"""

import numpy as np

from .exceptions import IncompatibleDimensionality
from .exceptions import NotADimObject

_DERIVED_DIMS = {
    (1,0,0,0,0,0,0):'length',
    (0,1,0,0,0,0,0):'time',
    (0,0,1,0,0,0,0):'mass',
    (0,0,0,1,0,0,0):'angle',
    (0,0,0,0,1,0,0):'charge',
    (0,0,0,0,0,1,0):'temperature',
    (0,0,0,0,0,0,1):'illumination',
    (2,0,0,0,0,0,0):'area',
    (3,0,0,0,0,0,0):'volume',
    (0,0,0,2,0,0,0):'solid angle',
    (0,-1,0,0,0,0,0):'frequency',
    (-3,0,1,0,0,0,0):'density',
    (1,-1,0,0,0,0,0):'velocity',
    (0,-1,0,1,0,0,0):'angular velocity',
    (1,-2,0,0,0,0,0):'acceleration',
    (1,-1,1,0,0,0,0):'momentum',
    (1,-2,1,0,0,0,0):'force',
    (2,-2,1,0,0,0,0):'energy',
    (2,-3,1,0,0,0,0):'power',
    (-1,-2,1,0,0,0,0):'pressure',
}

class Dim(object):
    __slot__ = ('dim')

    def __init__(self, dim=None, *, 
                 length=0, time=0, mass=0, angle=0, charge=0, temp=0, illum=0 ):
        self.dim = (length,time,mass,angle,charge,temp,illum)
        if dim is not None:
            if hasattr(dim,'dim'):
                dim = dim.dim
            try:
                t = [int(dim[i]) for i in range(7)]
            except:
                raise NotADimObject(dim)
            if np.any(self.dim):
                raise TypeError("Cannot specify both the dim tuple and individual exponents")
            self.dim = dim[:7]

    @property
    def dimensionless(self):
        return not np.any(self.dim)

    def assert_compatible(self,other):
        if self != other:
            raise IncompatibleDimensionality(self,other)

    def __eq__(self,other):
        print("__eq__")
        try:
            return self.dim == other.dim
        except:
            return False

    def __mul__(self,other):
        try:
            return Dim(tuple(a+b for a,b in zip(self.dim,other.dim)))
        except:
            raise NotADimObject(other)

    __rmul__ = __mul__

    def __truediv__(self,other):
        try:
            return Dim(tuple(a-b for a,b in zip(self.dim,other.dim)))
        except:
            raise NotADimObject(other)

    def __rtruediv__(self,other):
        try:
            return Dim(tuple(b-a for a,b in zip(self.dim,other.dim)))
        except:
            raise NotADimObject(other)

    def __pow__(self,n):
        return Dim(tuple(a*n for a in self.dim))

    def __repr__(self):
        keys = ('length','time','mass','angle','charge','temp','illum')
        return (f"Dim("
                + ",".join(f"{n}={e}" for n,e in zip(keys,self.dim) if e)
                + ",)")

    def __str__(self):
        if self.dim in _DERIVED_DIMS:
            return _DERIVED_DIMS[self.dim]

        names = ('L','T','M','A','C','K','I')
        num = " ".join(f"{n}" if e==1 else f"{n}^{e}" for n,e in
                       zip(names,self.dim) if e>0)
        den = " ".join(f"{n}" if e==-1 else f"{n}^{-e}" for n,e in
                       zip(names,self.dim) if e<0)

        if num:
            if den:
                return f"[{num}/{den}]"
            else:
                return f"[{num}]"
        elif den:
            return f"[1/{den}]"
        else:
            return f"[]"
        




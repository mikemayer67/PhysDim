class IncompatibleDimensions(TypeError):
  def __init__(self,a,b):
      a_pdim = "dimensionless" if a.is_dimensionless else a.pdim
      b_pdim = "dimensionless" if b.is_dimensionless else b.pdim
      TypeError.__init__(self,f"Operands have a different fundamental dimensions: {a_pdim} vs {b_pdim}")

class NotDimLike(TypeError):
    def __init__(self,x):
        TypeError.__init__(self,f"Expected a 7-tuple or a PhysicalDimension object, not {x}")

class UnsupportedUfunc(RuntimeError):
    def __init__(self,func_name):
        RuntimeError.__init__(self,f"PhysicalValue does not support {func_name}")

class AttemptToRedefineUnit(RuntimeError):
    def __init__(self,name):
        RuntimeError.__init__(self, f"The {name} unit has already been defined")

class AttemptToRedefineConstant(RuntimeError):
    def __init__(self,name):
        RuntimeError.__init__(self, f"The {name} constant has already been defined")

class AttemptToAssignToUnit(RuntimeError):
    def __init__(self,name):
        RuntimeError.__init__(self,
            f"Cannot directly assign value to units.{name}" )

class UndefinedUnit(KeyError):
    def __init__(self,name):
        KeyError.__init__(self,
            f"Unrecognized unit {name}. Consider adding it with the Units.add() method")

class IncompatibleDimensions(TypeError):
  def __init__(self,a,b):
      TypeError.__init__(self,f"{a} vs {b}")

class NotDimLike(TypeError):
    def __init__(self,x):
        TypeError.__init__(self,f"Expected a Dim object, not {x}")

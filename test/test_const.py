import unittest

import numpy as np

from physdim import constants as k

class ConstantsTest(unittest.TestCase):
    def test_math_const(self):
        self.assertEqual(k.pi,np.pi)


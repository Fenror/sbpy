import unittest

import numpy as np
import sbpy.operators

class TestOperators(unittest.TestCase):

    def test_get_fd_op(self):
        N = 100
        x = np.linspace(0, 1, 100)
        tol = 1e-14

        D,P,Q = sbpy.operators.get_fd_op(N, 1.0/(N-1), 2)
        self.assertTrue(np.max(np.abs(D@x - 1.0)) < tol)
        self.assertTrue(np.max(np.abs(D@np.ones(N))) < tol)

if __name__ == '__main__':
    unittest.main()

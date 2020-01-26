import unittest

import numpy as np
import sbpy.operators

class TestOperators(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.x = np.linspace(0, 1, self.N)
        self.dx = 1/(self.N-1)
        self.accuracy = 2
        self.sbp_op = sbpy.operators.SBP1D(self.N, self.dx, self.accuracy)
        self.D = self.sbp_op.D
        self.P = self.sbp_op.P
        self.Q = self.sbp_op.Q


    def test_differential_accuracy(self):
        tol = 1e-14
        self.assertTrue(np.max(np.abs(self.D@self.x - 1.0)) < tol)
        self.assertTrue(np.max(np.abs(self.D@np.ones(self.N))) < tol)


    def test_integral_accuracy(self):
        tol = 1e-14
        self.assertTrue(np.max(np.abs(sum(self.P@self.x) - 0.5)) < tol)
        self.assertTrue(np.max(np.abs(sum(self.P@np.ones(self.N)) - 1.0)) < tol)


    def test_sbp_property(self):
        tol = 1e-14
        E        = np.zeros((self.N,self.N))
        E[0,0]   = -1
        E[-1,-1] = 1
        self.assertTrue(np.all(self.Q + np.transpose(self.Q) == E))


if __name__ == '__main__':
    unittest.main()

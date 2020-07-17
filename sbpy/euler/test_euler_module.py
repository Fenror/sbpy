import unittest

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator

class TestSpatialOperator(unittest.TestCase):

    (X,Y) = get_circle_sector_grid(3, 0.0, 3.14/2, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid)
    U = X
    V = Y
    P = X**2 + Y**2
    state = np.array([U, V, P]).flatten()

    def test_euler_jacobian(self):
        S, J = euler_operator(self.sbp, self.state)

        for i,grad in enumerate(J):
            f = lambda x: euler_operator(self.sbp, x)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_wall_jacobian(self):
        S, J = wall_operator(self.sbp, self.state, 0, 'w')
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: wall_operator(self.sbp, x, 0, 'w')[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_case1(self):

        S1,J1 = euler_operator(self.sbp, self.state)
        S2,J2 = wall_operator(self.sbp, self.state, 0, 'w')
        S3,J3 = wall_operator(self.sbp, self.state, 0, 'e')
        S4,J4 = wall_operator(self.sbp, self.state, 0, 's')
        S5,J5 = wall_operator(self.sbp, self.state, 0, 'n')
        J = J1+J2+J3+J4+J5

        for i,grad in enumerate(J):
            def f(x):
                S1,J1 = euler_operator(self.sbp, x)
                S2,J2 = wall_operator(self.sbp, x, 0, 'w')
                S3,J3 = wall_operator(self.sbp, x, 0, 'e')
                S4,J4 = wall_operator(self.sbp, x, 0, 's')
                S5,J5 = wall_operator(self.sbp, x, 0, 'n')
                return (S1+S2+S3+S4+S5)[i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)



if __name__ == '__main__':
    unittest.main()


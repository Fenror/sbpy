"""This module contains functions for getting SBP operators."""

import numpy as np
from scipy import sparse

class SBP1D:
    """ Class representing a 1D finite difference SBP operator. """

    def __init__(self, N, dx):
        """ Initializes an SBP1D object.

        Args:
            N: The number of grid points.
            dx: The spacing between the grid points.
        """

        if accuracy != 2:
            raise ValueError('Accuracy must be 2')

        self.N  = N
        self.dx = dx

        stencil = np.array([-0.5, 0.0, 0.5])
        MID = sparse.diags(stencil,
                           [0, 1, 2],
                           shape=(N-2, N))

        TOP = sparse.bsr_matrix(([-0.5, 0.5], ([0, 0], [0, 1])),
                                shape=(1, N))
        BOT = sparse.bsr_matrix(([-0.5, 0.5], ([0, 0], [N-2, N-1])),
                                shape=(1, N))

        self.Q = sparse.vstack([TOP, MID, BOT])

        p     = np.ones(self.N)
        p     = dx*p
        p[0]  = 0.5*dx
        p[-1] = 0.5*dx
        p_inv = 1/p

        self.P = sparse.diags([p], [0])
        self.P_inv = sparse.diags([p_inv], [0])
        self.D = self.P_inv*self.Q


class SBP2D:
    """ Class representing 2D finite difference SBP operators.

    This class defines 2D curvilinear SBP operators on a supplied grid X, Y.
    Here X and Y are 2D numpy arrays representing the x- and y-values of the
    grid. X and Y should be structured such that (X[i,j], Y[i,j]) is equal to
    the (i,j):th grid node (x_i, y_j).
    """

    def __init__(self, X, Y):
        """ Initializes an SBP2D object.

        Args:
            X: The x-values of the grid nodes.
            Y: The y-values of the grid nodes.

        """
        assert(X.shape == Y.shape)

        self.X = X
        self.Y = Y
        (self.Nx, self.Ny) = X.shape

        self.sbp_xi  = SBP1D(Nx, 1/(Nx-1))
        self.sbp_eta = SBP1D(Ny, 1/(Ny-1))
        self.dx_dxi  = self.sbp_xi.D @ X
        self.dx_deta = X @ np.transpose(self.sbp_eta.D)
        self.dy_dxi  = self.sbp_xi.D @ Y
        self.dy_deta = Y @ np.transpose(self.sbp_eta.D)
        self.J       = dx_dxi*dy_deta + dx_deta*dy_dxi

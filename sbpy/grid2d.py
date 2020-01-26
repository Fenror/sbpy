""" This module contains functions and classes for managing 2D grids. """

from sbpy.operators import get_sbp_op


class Grid:


    def __init__(X, Y):
        assert(X.shape == Y.shape)
        self.X = X
        self.Y = Y
        self.Nx, self.Ny = X.shape


    def _construct_sbp_operators(self):




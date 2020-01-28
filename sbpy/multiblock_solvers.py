""" This module contains classes for solving partial differential equations on
multiblock grids. """

import numpy as np
from sbpy import operators
from sbpy import grid2d

class AdvectionSolver:
    """ A linear scalar advection solver. """

    def __init__(self, grid):
        self.grid = grid
        self.velocity = np.array([1.0,1.0])
        self.ops = []
        for (X,Y) in zip(grid.X_blocks, grid.Y_blocks):
            self.ops.append(operators.SBP2D(X,Y))

        self.inflow = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in {'s', 'e', 'n', 'w'}:
                bd = self.ops[k].normals[side]
                inflow = np.array([ self.velocity@n < 0 for n in bd ],
                                  dtype = bool)
                self.inflow[k][side] = inflow


    def Dt(self, U):
        Ut = np.array(shape=U.shape)

        for (i, u) in enumerate(U):
            Ut[i,:,:] = -self.velocity[0]*self.ops[i].diffx(u) \
                        -self.velocity[1]*self.ops[i].diffy(u)

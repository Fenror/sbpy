""" This module contains classes for solving partial differential equations on
multiblock grids. """

import numpy as np
from scipy import integrate
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
        Ut = np.zeros(shape=U.shape)
        Ux = np.array([self.ops[i].diffx(U[i,:,:]) for
            i in range(self.grid.num_blocks)])
        Uy = np.array([self.ops[i].diffy(U[i,:,:]) for
            i in range(self.grid.num_blocks)])

        for (block_idx, (ux, uy)) in enumerate(zip(Ux, Uy)):
            Ut[block_idx,:,:] = -(self.velocity[0]*ux+self.velocity[1]*uy)

        #for (block_idx, interfaces) in enumerate(self.grid.interfaces):
        #    for interface in interfaces.items():
        #        my_side       = interface[0]
        #        neighbor_idx  = interface[1][0]
        #        neighbor_side = interface[1][1]
        #        pinv    = self.ops[block_idx].pinv[my_side]
        #        bd_quad = self.ops[block_idx].boundary_quadratures[my_side]
        #        vel     = self.ops[block_idx].normals[my_side]@self.velocity
        #        my_sol  = grid2d.get_function_boundary(U[block_idx,:,:], my_side)
        #        neighbor_sol = grid2d.get_function_boundary(
        #                U[block_idx,:,:], neighbor_side)
        #        sigma = -0.5*vel*pinv*bd_quad*self.inflow[block_idx][my_side]

        #        if my_side == 's':
        #            Ut[block_idx,:,0] += sigma*(my_sol-neighbor_sol)
        #        elif my_side == 'e':
        #            Ut[block_idx,-1,:] += sigma*(my_sol-neighbor_sol)
        #        elif my_side == 'n':
        #            Ut[block_idx,:,-1] += sigma*(my_sol-neighbor_sol)
        #        elif my_side == 'w':
        #            Ut[block_idx,0,:] += sigma*(my_sol-neighbor_sol)

        #for (block_idx, non_interfaces) in enumerate(self.grid.non_interfaces):
        #    for side in non_interfaces:
        #        pinv    = self.ops[block_idx].pinv[side]
        #        bd_quad = self.ops[block_idx].boundary_quadratures[side]
        #        vel     = self.ops[block_idx].normals[side]@self.velocity
        #        sol     = grid2d.get_function_boundary(U[block_idx,:,:], side)
        #        sigma   = -0.5*vel*pinv*bd_quad*self.inflow[block_idx][side]
        #        if side == 's':
        #            Ut[block_idx,:,0] += sigma*(my_sol)
        #        elif side == 'e':
        #            Ut[block_idx,-1,:] += sigma*(my_sol)
        #        elif side == 'n':
        #            Ut[block_idx,:,-1] += sigma*(my_sol)
        #        elif side == 'w':
        #            Ut[block_idx,0,:] += sigma*(my_sol)

        return Ut

    def solve(self):
        init = np.ones((4,10,10))
        T = 4.0
        def f(t, y):
            y = np.reshape(y,(4,10,10))
            yt = self.Dt(y)
            return yt.flatten()

        self.sol = integrate.solve_ivp(f, (0.0, T), init.flatten())


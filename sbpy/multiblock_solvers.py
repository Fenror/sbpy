""" This module contains classes for solving partial differential equations on
multiblock grids. """

import numpy as np
from scipy import integrate
from sbpy import operators
from sbpy import grid2d
from scipy.stats import norm

class TestSolver:
    """ A linear scalar advection solver. """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.ops = operators.SBP2D(X,Y)
        self.velocity = np.array([1.0,0.0])
        self.inflow = {}
        self.Nx, self.Ny = X.shape
        for side in {'s', 'e', 'n', 'w'}:
            bd = self.ops.normals[side]
            inflow = np.array([ self.velocity@n < 0 for n in bd ],
                              dtype = bool)
            self.inflow[side] = inflow


    def Dt(self, U):
        Ut = np.zeros(shape=U.shape)
        Ux = self.ops.diffx(U)
        Uy = self.ops.diffy(U)

        Ut[:,:] = -(self.velocity[0]*Ux+self.velocity[1]*Uy)

        for side in ['s', 'e', 'n', 'w']:
            pinv    = self.ops.pinv[side]
            bd_quad = self.ops.boundary_quadratures[side]
            in_vel  = -self.ops.normals[side]@self.velocity
            sol     = grid2d.get_function_boundary(U, side)
            sigma   = -0.5*in_vel*pinv*bd_quad*self.inflow[side]
            if side == 's':
                Ut[:,0] += sigma*sol
            elif side == 'e':
                Ut[-1,:] += sigma*sol
            elif side == 'n':
                Ut[:,-1] += sigma*sol
            elif side == 'w':
                Ut[0,:] += sigma*sol

        return Ut

    def solve(self):
        #init = np.ones((self.Nx,self.Ny))
        init = 0.1*norm.pdf(self.Y,loc=0.0,scale=0.05) \
                  *norm.pdf(self.X,loc=-0.6,scale=0.05)

        T = 1.0
        def f(t, y):
            y = np.reshape(y,(self.Nx,self.Ny))
            yt = self.Dt(y)
            return yt.flatten()

        self.sol = integrate.solve_ivp(f, (0.0, T), init.flatten())


class AdvectionSolver:
    """ A multiblock linear scalar advection solver. """

    def __init__(self, grid):
        self.grid = grid
        self.velocity = np.array([1.0,1.0])
        self.ops = []
        self.U  = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Ux = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Uy = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Ut = [ np.zeros(shape) for shape in grid.get_shapes() ]

        for (X,Y) in grid.get_blocks():
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

        for (block_idx, interfaces) in enumerate(self.grid.interfaces):
            for interface in interfaces.items():
                my_side       = interface[0]
                neighbor_idx  = interface[1][0]
                neighbor_side = interface[1][1]
                pinv    = self.ops[block_idx].pinv[my_side]
                bd_quad = self.ops[block_idx].boundary_quadratures[my_side]
                in_vel  = -self.ops[block_idx].normals[my_side]@self.velocity
                my_sol  = grid2d.get_function_boundary(U[block_idx,:,:], my_side)
                neighbor_sol = self.grid.get_neighbor_boundary(
                        U[neighbor_idx,:,:], block_idx, my_side)
                sigma = -0.5*in_vel*pinv*bd_quad*self.inflow[block_idx][my_side]

                bd_slice = grid2d.get_boundary_slice(Ut[block_idx], my_side)
                Ut[block_idx][bd_slice] += sigma*(my_sol-neighbor_sol)

        for (block_idx, non_interfaces) in enumerate(self.grid.non_interfaces):
            for side in non_interfaces:
                pinv    = self.ops[block_idx].pinv[side]
                bd_quad = self.ops[block_idx].boundary_quadratures[side]
                in_vel  = -self.ops[block_idx].normals[side]@self.velocity
                sol     = grid2d.get_function_boundary(U[block_idx,:,:], side)
                sigma   = -0.5*in_vel*pinv*bd_quad*self.inflow[block_idx][side]
                bd_slice = grid2d.get_boundary_slice(Ut[block_idx], side)
                Ut[block_idx][bd_slice] += sigma*sol

        return Ut

    def solve(self):
        init = np.ones((4,50,50))
        for (k, (X,Y)) in enumerate(self.grid.get_blocks()):
            init[k,:,:] = 0.1*norm.pdf(Y,loc=0.0,scale=0.05) \
                             *norm.pdf(X,loc=-0.6,scale=0.05)
        T = 1.0
        def f(t, y):
            y = np.reshape(y,(4,50,50))
            yt = self.Dt(y)
            return yt.flatten()

        self.sol = integrate.solve_ivp(f, (0.0, T), init.flatten())

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
    """ A multiblock linear scalar advection solver. Currently assumes same
    resolution in all blocks. """

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

        # Save bool arrays determining inflows. For example, if inflow[k]['w'][j]
        # is True, then the j:th node of the western boundary of the k:th block
        # is an inflow node.
        self.inflow = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in {'s', 'e', 'n', 'w'}:
                bd = self.ops[k].normals[side]
                inflow = np.array([ self.velocity@n < 0 for n in bd ],
                                  dtype = bool)
                self.inflow[k][side] = inflow


        # Save penalty coefficients for each boundary
        self.penalty_coeffs = [ {} for _ in range(self.grid.num_blocks) ]
        for block_idx in range(self.grid.num_blocks):
            for side in ['s', 'e', 'n', 'w']:
                sbp_op  = self.grid.sbp_ops[block_idx]
                pinv    = sbp_op.pinv[side]
                bd_quad = sbp_op.boundary_quadratures[side]
                in_vel  = -self.ops[block_idx].normals[side]@self.velocity
                self.penalty_coeffs[block_idx][side] = \
                        -0.5*in_vel*pinv*bd_quad*self.inflow[block_idx][side]


    def update_sol(self, U):
        self.U = U


    def compute_spatial_derivatives(self):
        self.Ux = self.grid.diffx(self.U)
        self.Uy = self.grid.diffy(self.U)


    def compute_temporal_derivative(self):
        a = self.velocity[0]
        b = self.velocity[1]
        self.Ut = [ -(a*ux + b*uy) for (ux,uy) in zip(self.Ux, self.Uy) ]

        for (local_idx, interfaces) in enumerate(self.grid.get_interfaces()):
            for interface in interfaces.items():
                local_side    = interface[0]
                neighbor_idx  = interface[1][0]
                neighbor_side = interface[1][1]
                local_u       = self.U[local_idx]
                neighbor_u    = self.U[neighbor_idx]
                local_bd      = grid2d.get_function_boundary(local_u, local_side)
                neighbor_bd   = self.grid.get_neighbor_boundary(
                                    neighbor_u, local_idx, local_side)
                sigma         = self.penalty_coeffs[local_idx][local_side]
                bd_slice      = self.grid.get_boundary_slice(local_idx, local_side)
                self.Ut[local_idx][bd_slice] += sigma*(local_bd-neighbor_bd)

        for (block_idx, ext_bds) in enumerate(self.grid.get_external_boundaries()):
            for side in ext_bds:
                sol     = grid2d.get_function_boundary(self.U[block_idx], side)
                sigma   = self.penalty_coeffs[block_idx][side]
                bd_slice = self.grid.get_boundary_slice(block_idx, side)
                self.Ut[block_idx][bd_slice] += sigma*sol


    def solve(self):
        init = np.ones((4,50,50))
        for (k, (X,Y)) in enumerate(self.grid.get_blocks()):
            init[k,:,:] = 0.1*norm.pdf(Y,loc=0.0,scale=0.05) \
                             *norm.pdf(X,loc=-0.6,scale=0.05)
        T = 1.0
        def f(t, y):
            self.update_sol(np.reshape(y,(4,50,50)))
            self.compute_spatial_derivatives()
            self.compute_temporal_derivative()
            return np.concatenate([ ut.flatten() for ut in self.Ut ])

        self.sol = integrate.solve_ivp(f, (0.0, T), init.flatten())

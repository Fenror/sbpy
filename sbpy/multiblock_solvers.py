""" This module contains classes for solving partial differential equations on
multiblock grids. """

import numpy as np
from scipy import integrate
from sbpy import operators
from sbpy import grid2d
from scipy.stats import norm
import tqdm

def solve_ivp_pbar(tspan):
    """ Returns a decorator used with solve_ivp to get a progress bar.

    Args:
        tspan: The tspan argument passed to solve_ivp

    Example:
        tspan = (0.0, 1.5)

        @solve_ivp_bar(tspan)
        def f(t,y):
            ...
            return y_derivative

        sol = solve_ivp(f, tspan, some_initial_data)
    """
    def decorator(f):
        def new_f(t,y):
            if t - new_f.prev_t > (tspan[1]-tspan[0])/100 and t < tspan[1]:
                progress = (t-tspan[0])/(tspan[1]-tspan[0])*100
                new_f.pbar.n = int(np.ceil(progress))
                new_f.prev_t = t
                new_f.pbar.refresh()
            return f(t,y)

        new_f.pbar = tqdm.tqdm(total=100)
        new_f.prev_t = tspan[0]

        return new_f
    return decorator


class AdvectionSolver:
    """ A multiblock linear scalar advection solver. """

    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.t = 0
        self.U  = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Ux = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Uy = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Ut = [ np.zeros(shape) for shape in grid.get_shapes() ]

        if 'initial_data' in kwargs:
            assert(grid.is_shape_consistent(kwargs['initial_data']))
            self.U = kwargs['initial_data']

        if 'boundary_data' in kwargs:
            self.boundary_data = kwargs['boundary_data']
        else:
            self.boundary_data = None

        if 'source_term' in kwargs:
            self.source_term = kwargs['source_term']
        else:
            self.source_term = None

        if 'velocity' in kwargs:
            self.velocity = kwargs['velocity']
        else:
            self.velocity = np.array([1.0,1.0])

        # Save bool arrays determining inflows. For example, if inflow[k]['w'][j]
        # is True, then the j:th node of the western boundary of the k:th block
        # is an inflow node.
        self.inflow = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in {'s', 'e', 'n', 'w'}:
                bd = self.grid.get_normals(k,side)
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
                in_vel  = -self.grid.get_normals(block_idx,side)@self.velocity
                self.penalty_coeffs[block_idx][side] = \
                        -0.5*in_vel*pinv*bd_quad*self.inflow[block_idx][side]


    def _update_sol(self, U):
        self.U = U


    def _compute_spatial_derivatives(self):
        self.Ux = self.grid.diffx(self.U)
        self.Uy = self.grid.diffy(self.U)


    def _compute_temporal_derivative(self):
        a = self.velocity[0]
        b = self.velocity[1]
        self.Ut = [ -(a*ux + b*uy) for (ux,uy) in zip(self.Ux, self.Uy) ]

        if self.source_term is not None:
            for (k,(X,Y)) in enumerate(self.grid.get_blocks()):
                self.Ut[k] += self.source_term(self.t, X, Y)


        # Add interface penalties
        for (local_idx, interfaces) in enumerate(self.grid.get_block_interfaces()):
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

        # Add external boundary penalties
        for (block_idx, ext_bds) in enumerate(self.grid.get_external_boundaries()):
            for side in ext_bds:
                sol     = grid2d.get_function_boundary(self.U[block_idx], side)
                sigma   = self.penalty_coeffs[block_idx][side]
                bd_slice = self.grid.get_boundary_slice(block_idx, side)
                if self.boundary_data is not None:
                    (X,Y) = self.grid.get_block(block_idx)
                    (x,y) = grid2d.get_boundary(X,Y,side)
                    self.Ut[block_idx][bd_slice] += \
                            sigma*(sol-self.boundary_data(self.t,x,y))
                else:
                    self.Ut[block_idx][bd_slice] += sigma*sol


    def solve(self, tspan):
        init = grid2d.multiblock_to_array(self.grid, self.U)

        @solve_ivp_pbar(tspan)
        def f(t, y):
            U = grid2d.array_to_multiblock(self.grid, y)
            self.t = t
            self._update_sol(U)
            self._compute_spatial_derivatives()
            self._compute_temporal_derivative()

            return np.concatenate([ ut.flatten() for ut in self.Ut ])

        eval_pts = np.linspace(tspan[0], tspan[1], int(30*(tspan[1]-tspan[0])))
        self.sol = integrate.solve_ivp(f, tspan, init,
                                       rtol=1e-12, atol=1e-12,
                                       t_eval=eval_pts)

class AdvectionDiffusionSolver:
    """ A multiblock linear scalar advection-diffusion solver.

    Based on the interface coupling in Carpenter & Nordström (1998)"""

    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.t = 0
        self.epsilon = 0.01
        self.U   = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Ux  = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Uy  = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Uxx = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Uyy = [ np.zeros(shape) for shape in grid.get_shapes() ]
        self.Ut  = [ np.zeros(shape) for shape in grid.get_shapes() ]

        if 'initial_data' in kwargs:
            assert(grid.is_shape_consistent(kwargs['initial_data']))
            self.U = kwargs['initial_data']

        if 'boundary_data' in kwargs:
            self.boundary_data = kwargs['boundary_data']
        else:
            self.boundary_data = None

        if 'source_term' in kwargs:
            self.source_term = kwargs['source_term']
        else:
            self.source_term = None

        if 'velocity' in kwargs:
            self.velocity = kwargs['velocity']
        else:
            self.velocity = np.array([1.0,1.0])

        # Save bool arrays determining inflows. For example, if inflow[k]['w'][j]
        # is True, then the j:th node of the western boundary of the k:th block
        # is an inflow node.
        self.inflow = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in ['s', 'e', 'n', 'w']:
                bd = self.grid.get_normals(k, side)
                inflow = np.array([ self.velocity@n < 0 for n in bd ],
                                  dtype = bool)
                self.inflow[k][side] = inflow

        # Compute interface alphas
        self.alphas = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in ['s', 'e', 'n', 'w']:
                self.alphas[k][side] = self._compute_alpha(k, side)

        # Save penalty coefficients for each interface
        self.inviscid_if_coeffs = [ {} for _ in range(self.grid.num_blocks) ]
        self.viscid_if_coeffs = [ {} for _ in range(self.grid.num_blocks) ]
        for interface in self.grid.get_interfaces():
            block_idx1 = interface[0][0]
            side1      = interface[0][1]
            block_idx2 = interface[1][0]
            side2      = interface[1][1]
            normals    = grid.get_normals(block_idx1, side1)
            flow_vel   = np.array([ self.velocity@n for n in normals ])

            pinv1 = self.grid.sbp_ops[block_idx1].pinv[side1]
            bdquad1 = self.grid.sbp_ops[block_dx1].boundary_quadratures[side1]
            pinv2 = self.grid.sbp_ops[block_idx2].pinv[side2]
            bdquad2 = self.grid.sbp_ops[block_dx2].boundary_quadratures[side2]

            alpha1 = self._compute_alpha(block_idx1, side1)
            alpha2 = self._compute_alpha(block_idx2, side2)

            s1_viscid = -1.0
            s2_viscid = 0.0
            s1_inviscid = 0.5 - 0.25*self.eps*\
                    (s1_viscid**2/alpha1 + s2_viscid**2/alpha2)
            s2_inviscid = s1_inviscid - 1
            self.inviscid_if_coeffs[block_idx1][side1] = \
                s1_inviscid*pinv1*bdquad1*flow_vel
            self.inviscid_if_coeffs[block_idx2][side2] = \
                -s2_inviscid*pinv2*bdquad2*flow_vel
            self.viscid_if_coeffs[block_idx1][side1] = \
                s1_viscid*self.eps*pinv1*bdquad1
            self.viscid_if_coeffs[block_idx2][side2] = \
                s2_viscid*self.eps*pinv2*bdquad2


    def _compute_alpha(block_idx, side):
        vol_quad = self.grid.sbp_ops[block_idx].volume_quadrature
        vol_quad = grid2d.get_function_boundary(vol_quad, side)
        normals = self.grid.get_normals(block_idx, side)
        nx = normals[:,0]
        ny = normals[:,1]
        bd_quad = self.grid.sbp_ops[block_idx].boundary_quadratures[side]
        alphas = np.concatenate([vol_quad/(nx**2 * bd_quad),
                                 vol_quad/(ny**2 * bd_quad)])
        return 0.5*min(alphas)


    def _update_sol(self, U):
        self.U = U


    def _compute_spatial_derivatives(self):
        self.Ux = self.grid.diffx(self.U)
        self.Uy = self.grid.diffy(self.U)


    def _compute_temporal_derivative(self):
        a = self.velocity[0]
        b = self.velocity[1]
        self.Ut = [ -(a*ux + b*uy) for (ux,uy) in zip(self.Ux, self.Uy) ]

        if self.source_term is not None:
            for (k,(X,Y)) in enumerate(self.grid.get_blocks()):
                self.Ut[k] += self.source_term(self.t, X, Y)


        # Add interface penalties
        for (local_idx, interfaces) in enumerate(self.grid.get_block_interfaces()):
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

        # Add external boundary penalties
        for (block_idx, ext_bds) in enumerate(self.grid.get_external_boundaries()):
            for side in ext_bds:
                sol     = grid2d.get_function_boundary(self.U[block_idx], side)
                sigma   = self.penalty_coeffs[block_idx][side]
                bd_slice = self.grid.get_boundary_slice(block_idx, side)
                if self.boundary_data is not None:
                    (X,Y) = self.grid.get_block(block_idx)
                    (x,y) = grid2d.get_boundary(X,Y,side)
                    self.Ut[block_idx][bd_slice] += \
                            sigma*(sol-self.boundary_data(self.t,x,y))
                else:
                    self.Ut[block_idx][bd_slice] += sigma*sol


    def solve(self, tspan):
        init = grid2d.multiblock_to_array(self.grid, self.U)

        @solve_ivp_pbar(tspan)
        def f(t, y):
            U = grid2d.array_to_multiblock(self.grid, y)
            self.t = t
            self._update_sol(U)
            self._compute_spatial_derivatives()
            self._compute_temporal_derivative()

            return np.concatenate([ ut.flatten() for ut in self.Ut ])

        eval_pts = np.linspace(tspan[0], tspan[1], int(30*(tspan[1]-tspan[0])))
        self.sol = integrate.solve_ivp(f, tspan, init,
                                       rtol=1e-12, atol=1e-12,
                                       t_eval=eval_pts)

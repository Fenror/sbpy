"""This module contains functions for getting SBP operators."""

import pdb

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sbpy import grid2d

class SBP1D:
    """ Class representing a 1D finite difference SBP operator.

    Attributes:
        P: Quadrature matrix.
        Q: An almost skew-symmetric matrix (Q+Q^T = diag(-1,0,0,...,1)), such
            that P^(-1)Q is an SBP operator.
        D: The SBP operator P^(-1)Q.
    """

    def __init__(self, N, dx, accuracy = 2):
        """ Initializes an SBP1D object.

        Args:
            N: The number of grid points.
            dx: The spacing between the grid points.
            accuracy: The accuracy of the interior stencil (2 or 4)
        """

        assert(accuracy in [2,4])

        self.N  = N
        self.dx = dx

        if accuracy == 2:
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

        if accuracy == 4:
            h1=17/48
            h2=59/48
            h3=43/48
            h4=49/48
            q1=1/12
            q2=2/3
            Q = sparse.diags([(N-2)*[q1], (N-1)*[-q2], N*[0], (N-1)*[q2], (N-2)*[-q1]],
                             [-2,-1,0,1,2])
            Q = Q.tolil()
            Q[:4,:4] = [[-1/2  , 59/96 , -1/12 , -1/32],
                        [-59/96, 0     , 59/96 , 0    ],
                        [1/12  , -59/96, 0     , 59/96],
                        [1/32  , 0     , -59/96, 0    ]]
            Q[-4:,-4:] = -np.rot90(Q[:4,:4].todense(), k=2)
            self.Q = Q
            p = dx*np.concatenate([[h1,h2,h3,h4], np.ones(N-8), [h4, h3, h2, h1]])
            p_inv = 1/p


        self.P = sparse.diags([p], [0])
        self.P_inv = sparse.diags([p_inv], [0])
        self.D = self.P_inv@self.Q


class SBP2D:
    """ Class representing 2D finite difference SBP operators.

    This class defines 2D curvilinear SBP operators on a supplied grid X, Y,
    based on Ålund & Nordström (JCP, 2019).  Here X and Y are 2D numpy arrays
    representing the x- and y-values of the grid. X and Y should be structured
    such that (X[i,j], Y[i,j]) is equal to the (i,j):th grid node (x_ij, y_ij).

    Attributes:
        normals: A dictionary containing the normals for each boundary. The keys
            are 's' for south, 'e' for east, 'n' for north, 'w' for west.
            For example, normals['w']
        boundary_quadratures: A dictionary containing boundary quadratures for
            each boundary. I.e. arrays of weights that can be used to compute
            integrals over the boundaries.
        volume_quadrature: A matrix representing a quadrature over the domain.
    """

    def __init__(self, X, Y, accuracy = 2):
        """ Initializes an SBP2D object.

        Args:
            X: The x-values of the grid nodes.
            Y: The y-values of the grid nodes.

        Optional:
            accuracy: The accuracy of the interior stencils (2 or 4).

        """
        assert(X.shape == Y.shape)
        assert(accuracy in [2,4])

        self.X = X
        self.Y = Y
        (self.Nx, self.Ny) = X.shape

        self.Ix      = sparse.eye(self.Nx)
        self.Iy      = sparse.eye(self.Ny)
        self.sbp_xi  = SBP1D(self.Nx, 1/(self.Nx-1), accuracy)
        self.sbp_eta = SBP1D(self.Ny, 1/(self.Ny-1), accuracy)
        self.dx_dxi  = self.sbp_xi.D @ X
        self.dx_deta = X @ np.transpose(self.sbp_eta.D)
        self.dy_dxi  = self.sbp_xi.D @ Y
        self.dy_deta = Y @ np.transpose(self.sbp_eta.D)
        self.jac     = self.dx_dxi*self.dy_deta - self.dx_deta*self.dy_dxi
        self.sides   = { 'w': np.array([[x,y] for x,y in zip(X[0,:], Y[0,:])]),
                         'e': np.array([[x,y] for x,y in zip(X[-1,:], Y[-1,:])]),
                         's': np.array([[x,y] for x,y in zip(X[:,0], Y[:, 0])]),
                         'n': np.array([[x,y] for x,y in zip(X[:,-1], Y[:,-1])])}

        # Construct 2D SBP operators.
        self.J    = sparse.diags(self.jac.flatten())
        self.Jinv = sparse.diags(1/self.jac.flatten())
        self.Xxi  = sparse.diags(self.dx_dxi.flatten())
        self.Xeta = sparse.diags(self.dx_deta.flatten())
        self.Yxi  = sparse.diags(self.dy_dxi.flatten())
        self.Yeta = sparse.diags(self.dy_deta.flatten())
        self.Dxi  = sparse.kron(self.sbp_xi.D, self.Iy)
        self.Deta = sparse.kron(self.Ix, self.sbp_eta.D)
        self.Dx   = 0.5*self.Jinv*(self.Yeta @ self.Dxi +
                                   self.Dxi @ self.Yeta -
                                   self.Yxi @ self.Deta -
                                   self.Deta @ self.Yxi)
        self.Dy   = 0.5*self.Jinv*(self.Xxi @ self.Deta +
                                   self.Deta @ self.Xxi -
                                   self.Xeta @ self.Dxi -
                                   self.Dxi @ self.Xeta)
        self.P = self.J@sparse.kron(self.sbp_xi.P, self.sbp_eta.P)
        self.Pinv = sparse.diags(1/self.P.data)

        # Save matrix version of volume quadrature.
        self.volume_quadrature = np.reshape(self.P.diagonal(),
                                            (self.Nx, self.Ny))


        # Construct boundary quadratures.
        self.boundary_quadratures = {}
        self.pxi = np.diag(self.sbp_xi.P.todense())
        self.peta = np.diag(self.sbp_eta.P.todense())

        dx_deta_w = grid2d.get_function_boundary(self.dx_deta, 'w')
        dy_deta_w = grid2d.get_function_boundary(self.dy_deta, 'w')
        self.boundary_quadratures['w'] = \
                self.peta*np.sqrt(dx_deta_w**2 + dy_deta_w**2)

        dx_deta_e = grid2d.get_function_boundary(self.dx_deta, 'e')
        dy_deta_e = grid2d.get_function_boundary(self.dy_deta, 'e')
        self.boundary_quadratures['e'] = \
                self.peta*np.sqrt(dx_deta_e**2 + dy_deta_e**2)

        dx_dxi_s = grid2d.get_function_boundary(self.dx_dxi, 's')
        dy_dxi_s = grid2d.get_function_boundary(self.dy_dxi, 's')
        self.boundary_quadratures['s'] = \
                self.pxi*np.sqrt(dx_dxi_s**2 + dy_dxi_s**2)

        dx_dxi_n = grid2d.get_function_boundary(self.dx_dxi, 'n')
        dy_dxi_n = grid2d.get_function_boundary(self.dy_dxi, 'n')
        self.boundary_quadratures['n'] = \
                self.pxi*np.sqrt(dx_dxi_n**2 + dy_dxi_n**2)

        # Construct P^(-1) at boundaries.
        self.pinv = {}
        for side in ['s','e','n','w']:
            self.pinv[side] = 1/grid2d.get_function_boundary(
                    self.jac*np.outer(self.pxi, self.peta), side)

        # Compute normals.
        self.normals = {}
        self.normals['w'] = \
            np.array([ np.array([-nx, ny])/np.linalg.norm([nx, ny]) for
              (nx,ny) in zip(self.dy_deta[0,:], self.dx_deta[0,:]) ])
        self.normals['e'] = \
            np.array([ np.array([nx, -ny])/np.linalg.norm([nx, ny]) for
              (nx,ny) in zip(self.dy_deta[-1,:], self.dx_deta[-1,:]) ])
        self.normals['s'] = \
            np.array([ np.array([nx, -ny])/np.linalg.norm([nx, ny]) for
             (nx,ny) in zip(self.dy_dxi[:,0], self.dx_dxi[:,0]) ])
        self.normals['n'] = \
            np.array([ np.array([-nx, ny])/np.linalg.norm([nx, ny]) for
             (nx,ny) in zip(self.dy_dxi[:,-1], self.dx_dxi[:,-1]) ])


    def plot(self):
        """ Plots the grid and normals. """

        diam  = np.array([self.X[0,0]-self.X[-1,-1],self.Y[0,0]-self.Y[-1,-1]])
        scale = np.linalg.norm(diam) / np.max([self.Nx, self.Ny])

        fig, ax = plt.subplots()
        xmin    = np.min(self.X)
        xmax    = np.max(self.X)
        ymin    = np.min(self.Y)
        ymax    = np.max(self.Y)
        ax.set_xlim([xmin-scale,xmax+scale])
        ax.set_ylim([ymin-scale,ymax+scale])
        ax.plot(self.X, self.Y, 'b')
        ax.plot(np.transpose(self.X), np.transpose(self.Y), 'b')
        for side in ['w','e','s','n']:
            for p,n in zip(self.sides[side], self.normals[side]):
                ax.arrow(p[0], p[1], scale*n[0], scale*n[1],
                         head_width=0.01,
                         fc='k', ec='k')
        ax.axis('equal')
        plt.show()


    def diffx(self, u):
        """ Differentiates a grid function with respect to x. """
        return np.reshape(self.Dx@u.flatten(), (self.Nx, self.Ny))


    def diffy(self, u):
        """ Differentiates a grid function with respect to y. """
        return np.reshape(self.Dy@u.flatten(), (self.Nx, self.Ny))


    def integrate(self, u):
        """ Integrates a grid function over the domain. """
        return np.sum(self.P@u.flatten())

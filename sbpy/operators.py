"""This module contains functions for getting SBP operators."""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

class SBP1D:
    """ Class representing a 1D finite difference SBP operator. """

    def __init__(self, N, dx):
        """ Initializes an SBP1D object.

        Args:
            N: The number of grid points.
            dx: The spacing between the grid points.
        """

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
    the (i,j):th grid node (x_ij, y_ij).
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

        self.Ix      = sparse.eye(self.Nx)
        self.Iy      = sparse.eye(self.Ny)
        self.sbp_xi  = SBP1D(self.Nx, 1/(self.Nx-1))
        self.sbp_eta = SBP1D(self.Ny, 1/(self.Ny-1))
        self.dx_dxi  = self.sbp_xi.D @ X
        self.dx_deta = X @ np.transpose(self.sbp_eta.D)
        self.dy_dxi  = self.sbp_xi.D @ Y
        self.dy_deta = Y @ np.transpose(self.sbp_eta.D)
        self.jac     = self.dx_dxi*self.dy_deta - self.dx_deta*self.dy_dxi
        self.sides   = { 'w': np.array([[x,y] for x,y in zip(X[0,:], Y[0,:])]),
                         'e': np.array([[x,y] for x,y in zip(X[-1,:], Y[-1,:])]),
                         's': np.array([[x,y] for x,y in zip(X[:,0], Y[:, 0])]),
                         'n': np.array([[x,y] for x,y in zip(X[:,-1], Y[:,-1])])}

        """ Construct 2D SBP operators. """
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

        """ Compute normals. """
        self.normals = {}
        self.normals['w'] = \
            [ np.array([-nx, ny])/np.linalg.norm([nx, ny]) for
              (nx,ny) in zip(self.dy_deta[0,:], self.dx_deta[0,:]) ]
        self.normals['e'] = \
            [ np.array([nx, -ny])/np.linalg.norm([nx, ny]) for
              (nx,ny) in zip(self.dy_deta[-1,:], self.dx_deta[-1,:]) ]
        self.normals['s'] = \
            [ np.array([nx, -ny])/np.linalg.norm([nx, ny]) for
             (nx,ny) in zip(self.dy_dxi[:,0], self.dx_dxi[:,0]) ]
        self.normals['n'] = \
            [ np.array([-nx, ny])/np.linalg.norm([nx, ny]) for
             (nx,ny) in zip(self.dy_dxi[:,-1], self.dx_dxi[:,-1]) ]


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
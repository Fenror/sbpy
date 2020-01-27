""" This module contains functions and classes for managing 2D grids. """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


class Multiblock:
    """ Represents a structured multiblock grid.

    Attributes:
        X_blocks: A list of 2D numpy arrays containing x-values for each
                  block.
        Y_blocks: A list of 2D numpy arrays containing y-values for each
                  block.
        corners: A list of unique corners in the grid.
        faces: A list of index-quadruples defining the blocks. For example, if
               faces[n] = [i,j,k,l], and (X,Y) are the matrices corresponding to
               block n, then (X[0,0],Y[0,0]) = corners[i]
                             (X[-1,0],Y[-1,0]) = corners[j]
                             (X[-1,-1],Y[-1,-1]) = corners[k]
                             (X[0,-1],Y[0,-1]) = corners[l]
    """

    def __init__(self, X_blocks, Y_blocks):
        """ Initializes a Multiblock object.

        Args:
            X_blocks: A list of 2D numpy arrays containing x-values for each
                      block.
            Y_blocks: A list of 2D numpy arrays containing y-values for each
                      block.

            Note that the structure of these blocks should be such that
            (X_blocks[k][i,j], Y_blocks[k][i,j]) is the (i,j):th node in the
            k:th block.
        """

        self.X_blocks = X_blocks
        self.Y_blocks = Y_blocks

        # Save unique corners
        self.corners = []
        for X,Y in zip(X_blocks, Y_blocks):
            self.corners.append(self.get_corners(X,Y))

        self.corners = np.unique(np.concatenate(self.corners), axis=0)

        # Save faces in terms of unique corners
        self.faces = []

        for k,(X,Y) in enumerate(zip(X_blocks, Y_blocks)):
            block_corners = self.get_corners(X,Y)
            indices = []
            for c in block_corners:
                idx = np.argwhere(np.all(c == self.corners, axis=1)).item()
                indices.append(idx)
            self.faces.append(np.array(indices))


    def plot_grid(self):
        """ Plot the entire grid. """

        fig, ax = plt.subplots()
        for X,Y in zip(self.X_blocks, self.Y_blocks):
            ax.plot(X,Y,'b')
            ax.plot(np.transpose(X),np.transpose(Y),'b')
            for side in {'w', 'e', 's', 'n'}:
                x,y = self.get_boundary(X,Y,side)
                ax.plot(x,y,'k',linewidth=3)

        plt.show()


    def plot_domain(self):
        """ Fancy domain plot without gridlines. """

        fig, ax = plt.subplots()
        for k,(X,Y) in enumerate(zip(self.X_blocks, self.Y_blocks)):
            xs,ys = self.get_boundary(X,Y,'s')
            xe,ye = self.get_boundary(X,Y,'e')
            xn,yn = self.get_boundary(X,Y,'n')
            xn = np.flip(xn)
            yn = np.flip(yn)
            xw,yw = self.get_boundary(X,Y,'w')
            xw = np.flip(xw)
            yw = np.flip(yw)
            x_poly = np.concatenate([xs,xe,xn,xw])
            y_poly = np.concatenate([ys,ye,yn,yw])

            ax.fill(x_poly,y_poly,'tab:gray')
            ax.plot(x_poly,y_poly,'k')
            c = self.get_center(X,Y)
            ax.text(c[0], c[1], "$\Omega_" + str(k) + "$")


        plt.show()


    def get_boundary(self,X,Y,side):
        """ Returns the boundary a block. """

        assert(side in {'w','e','s','n'})

        if side == 'w':
            return X[0,:], Y[0,:]
        elif side == 'e':
            return X[-1,:], Y[-1,:]
        elif side == 's':
            return X[:,0], Y[:,0]
        elif side == 'n':
            return X[:,-1], Y[:,-1]


    def get_corners(self,X,Y):
        """ Returns the corners of a block.

        Starts with (X[0,0], Y[0,0]) and continues counter-clockwise.
        """
        return np.array([[X[0,0]  , Y[0,0]  ],
                         [X[-1,0] , Y[-1,0] ],
                         [X[-1,-1], Y[-1,-1]],
                         [X[0,-1] , Y[0,-1]]])


    def get_center(self,X,Y):
        """ Returns the center point of a block. """
        corners = self.get_corners(X,Y)
        return 0.25*(corners[0] + corners[1] + corners[2] + corners[3])


def load_p3d(filename):
    with open(filename) as data:
        num_blocks = int(data.readline())

        X = []
        Y = []
        Nx = []
        Ny = []
        for _ in range(num_blocks):
            size = np.fromstring(data.readline(), sep=' ', dtype=int)
            Nx.append(size[0])
            Ny.append(size[1])

        for k in range(num_blocks):
            X_cur = []
            Y_cur = []
            for n in range(Nx[k]):
                X_cur.append(np.fromstring(data.readline(), sep=' '))
            for n in range(Nx[k]):
                Y_cur.append(np.fromstring(data.readline(), sep=' '))

            X.append(np.array(X_cur))
            Y.append(np.array(Y_cur))
            for _ in range(Nx[k]):
                next(data)


    return (X,Y)

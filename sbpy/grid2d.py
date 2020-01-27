""" This module contains functions and classes for managing 2D grids. """

import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(X, Y):
        assert(X.shape == Y.shape)
        self.X = X
        self.Y = Y
        self.Nx, self.Ny = X.shape


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

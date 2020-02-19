""" This module contains functions for displaying and interacting with various
objects. """

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import path
from sbpy import grid2d
from sbpy import utils

def select_nodes(grid):
    """ Select grid nodes from a plot. """

    def flat_to_multi_idx(idx):
        block = 0
        count = 0
        shapes = grid.get_shapes()
        for (k,shape) in enumerate(shapes):
            count += shape[0]*shape[1]
            if idx < count:
                block = k
                break

        local_flat_idx = idx - (count-shapes[k][0]*shapes[k][1])
        Nx,Ny = shapes[k]
        (i,j) = (np.floor(local_flat_idx/Ny),local_flat_idx%Ny)
        return (block,i,j)


    fig, ax = plt.subplots()
    for X,Y in grid.get_blocks():
        ax.scatter(X,Y, c='b')

    def onselect(verts):
        p = path.Path(verts)
        x_points = []
        y_points = []
        for (X,Y) in grid.get_blocks():
            x_points.append(X.flatten())
            y_points.append(Y.flatten())

        x_points = np.concatenate(x_points)
        y_points = np.concatenate(y_points)
        pts = [(x,y) for x,y in zip(x_points, y_points)]

        ind = np.nonzero(p.contains_points(pts))[0]
        print([flat_to_multi_idx(i) for i in ind])

    lasso = widgets.LassoSelector(ax, onselect)
    plt.show()

if __name__ == '__main__':
    N = 11
    blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    grid = grid2d.MultiblockSBP(blocks, accuracy=4)
    select_nodes(grid)

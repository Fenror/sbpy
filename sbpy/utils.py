""" This module contains various utility functions. """

import itertools
import numpy as np
from sbpy import grid2d

def create_convergence_table(labels, errors, h, title=None, filename=None):
    """
    Creates a latex convergence table.
    Parameters:
      labels: An array strings describing each grid (e.g. number of nodes).
      errors: An array of errors.
      h: An array of grid sizes.

    Optional Parameters:
      title: Table title.
      filename: Write table to file.

    Output:
      Prints tex code for the table.
    """

    errors = np.array(errors)
    h = np.array(h)

    rates = (np.log(errors[:-1]) - np.log(errors[1:]))/(np.log(h[:-1]) - np.log(h[1:]))

    N = len(errors)
    print("\\begin{tabular}{|l|l|l|}")
    print("\hline")

    if title:
        print("\multicolumn{{3}}{{|c|}}{{{}}} \\\\".format(title))

    print("\hline")
    print("& error & rate \\\\".format(errors[0]))
    print("\hline")
    print(labels[0], " & {:.4e} & - \\\\".format(errors[0]))
    for k in range(1,N):
        print(labels[k], " & {:.4e} & {:.2f} \\\\".format(errors[k], rates[k-1]))
    print("\hline")
    print("\\end{tabular}")

    if filename:
        with open(filename,'a') as f:
            f.write("\\begin{tabular}{|l|l|l|}\n")
            f.write("\hline\n")

            if title:
                f.write("\multicolumn{{3}}{{|c|}}{{{}}} \\\\\n".format(title))

            f.write("\hline\n")
            f.write("& error & rate \\\\\n".format(errors[0]))
            f.write("\hline\n")
            f.write(str(labels[0]) + " & {:.4e} & - \\\\\n".format(errors[0]))
            for k in range(1,N):
                f.write(str(labels[k]) + " & {:.4e} & {:.2f} \\\\\n".format(errors[k], rates[k-1]))
            f.write("\hline\n")
            f.write("\\end{tabular}\n\n")


def get_circle_sector_grid(N, th0, th1, r_inner, r_outer):
    """ Returns a circle sector grid.

    Arguments:
        N: Number of gridpoints in each direction.
        th0: Start angle.
        th1: End angle.
        r_inner: Inner radius.
        r_outer: Outer radius.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    d_r = (r_outer - r_inner)/(N-1)
    d_th = (th1-th0)/(N-1)

    radii = np.linspace(r_inner, r_outer, N)
    thetas = np.linspace(th0, th1, N)

    x = np.zeros(N*N)
    y = np.zeros(N*N)

    pos = 0
    for r in radii:
        for th in thetas:
            x[pos] = r*np.cos(th)
            y[pos] = r*np.sin(th)
            pos += 1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y


def get_annulus_grid(N):
    """ Returns a list of four blocks constituting an annulus grid. """
    blocks = [get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    return blocks


def get_bump_grid(Nx,Ny):
    """ Returns a grid with two bumps in the floor and ceiling.
    Arguments:
        N: Number of gridpoints in each direction.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    x0 = -1.5
    x1 = 1.5
    dx = (x1-x0)/(Nx-1)
    y0 = lambda x: 0.0625*np.exp(-25*x**2)
    y1 = lambda y: 0.8
    x = np.zeros(Nx*Ny)
    y = np.copy(x)
    pos = 0
    for i in range(Nx):
        for j in range(Ny):
            x_val = x0 + i*dx
            x[pos] = x_val
            y[pos] = y0(x_val) + j*(y1(x_val)-y0(x_val))/(Ny-1)
            pos = pos+1

    X = np.reshape(x,(Nx,Ny))
    Y = np.reshape(y,(Nx,Ny))

    return X,Y

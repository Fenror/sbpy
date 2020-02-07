import sys
sys.path.append('..')
import numpy as np
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import animation
from sbpy import utils

def get_bump(N):
    x0 = -1.5
    x1 = 1.5
    dx = (x1-x0)/(N-1)
    y0 = lambda x: 0.0625*np.exp(-25*x**2)
    y1 = lambda y: 0.8 - 0.0625*np.exp(-25*y**2)
    x = np.zeros(N*N)
    y = np.copy(x)
    pos = 0
    for i in range(N):
        for j in range(N):
            x_val = x0 + i*dx
            x[pos] = x_val
            y[pos] = y0(x_val) + j*(y1(x_val)-y0(x_val))/(N-1)
            pos = pos+1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y


def get_pizza(N, th0, th1, r_inner, r_outer):
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


def u(t,x,y):
    return np.sin(x+y+t)

def g(t,x,y):
    return np.sin(x+y+t)

def F(t,x,y):
    return 3*np.cos(x+y+t)


errs = []
resolutions = np.array([11, 21, 41, 81, 161])
h = 1/(resolutions-1)

for N in resolutions:
    #blocks = grid2d.load_p3d('cyl' + str(N) + '.p3d')
    blocks = [get_pizza(N, 0, 0.5*np.pi, 0.2, 1.0),
              get_pizza(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              get_pizza(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              get_pizza(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    grid = grid2d.MultiblockSBP(blocks, accuracy=4)
    init = [ np.ones(shape) for shape in grid.get_shapes() ]

    for (k, (X,Y)) in enumerate(grid.get_blocks()):
        init[k] = u(0,X,Y)

    solver = multiblock_solvers.AdvectionSolver(grid, initial_data=init,
                                                boundary_data=g,
                                                source_term=F)

    tspan = (0.0, 1.0)
    solver.solve(tspan)

    final_time = solver.sol.t[-1]
    U = []
    for frame in np.transpose(solver.sol.y):
        U.append(grid2d.array_to_multiblock(grid, frame))

    U_exact = grid.evaluate_function(lambda x,y: u(final_time, x, y))

    err = [ (u - u_exact)**2 for (u,u_exact) in zip(U[-1],U_exact) ]
    errs.append(np.sqrt(grid.integrate(err)))

utils.create_convergence_table(resolutions, errs, h)


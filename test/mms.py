import sys
sys.path.append('..')
import numpy as np
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import animation
from sbpy import utils

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
    blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
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


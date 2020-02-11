import sys
sys.path.append('..')
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import animation
from sbpy import utils


#blocks = grid2d.load_p3d('cyl11.p3d')
#grid = grid2d.MultiblockSBP(blocks, accuracy=4)
N = 30
blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
grid2d.collocate_corners(blocks)
grid = grid2d.MultiblockSBP(blocks, accuracy=4)
init = [ np.ones(shape) for shape in grid.get_shapes() ]
for (k, (X,Y)) in enumerate(grid.get_blocks()):
    init[k] = 0.02*norm.pdf(Y,loc=-0.5,scale=0.05)*norm.pdf(X,loc=-0.6,scale=0.05)

def g(t,x,y):
    return np.sin(t)

solver = multiblock_solvers.AdvectionDiffusionSolver(grid, initial_data=init)
tspan = (0.0, 1.0)
solver.solve(tspan)

U = []
for frame in np.transpose(solver.sol.y):
    U.append(grid2d.array_to_multiblock(grid, frame))

animation.animate_multiblock(grid, U)

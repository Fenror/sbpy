import sys
import time
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
from sbpy import gui


N = 161
blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
grid2d.collocate_corners(blocks)
fine_grid = grid2d.MultiblockSBP(blocks, accuracy=4)

def g(t,x,y):
    return 1

velocity = np.array([1,1])/np.sqrt(2)
diffusion = 0.01

solver = multiblock_solvers.AdvectionDiffusionSolver(fine_grid,
                                                     velocity=velocity,
                                                     diffusion=diffusion)
solver.set_boundary_condition(1,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(3,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(5,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(7,{'type': 'dirichlet', 'data': g})
tspan = (0.0, 3.5)

solver.solve(tspan)

U = []
for frame in np.transpose(solver.sol.y):
    U.append(grid2d.array_to_multiblock(fine_grid, frame))

U_highres = U[-1]

#N = 21
#blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
#          utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
#          utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
#          utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
#grid2d.collocate_corners(blocks)
#coarse_grid = grid2d.MultiblockSBP(blocks, accuracy=4)
#
#selector = gui.NodeSelector(coarse_grid)
#selector()
#nodes = selector.nodes
#
#int_data, int_idx = utils.fetch_highres_data(coarse_grid,
#        nodes, fine_grid, U_highres)
#
import pickle
#with open('int_data.pkl', 'wb') as f:
#    pickle.dump([int_data, int_idx], f)

with open('highres_sol161.pkl', 'wb') as f:
    pickle.dump([U_highres], f)


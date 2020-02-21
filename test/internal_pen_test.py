import sys
sys.path.append('..')
import time
import pickle
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

#with open('int_data.pkl', 'rb') as f:
#    int_data, int_idx = pickle.load(f)
with open('highres_sol.pkl', 'rb') as f:
    U_highres, = pickle.load(f)

def g(t,x,y):
    return 1

velocity = np.array([1,1])/np.sqrt(2)
diffusion = 0.01
tspan = (0.0, 3.5)

N = 21
blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
grid2d.collocate_corners(blocks)
coarse_grid = grid2d.MultiblockSBP(blocks, accuracy=4)

N = 81
blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
grid2d.collocate_corners(blocks)
fine_grid = grid2d.MultiblockSBP(blocks, accuracy=4)

#selector = gui.NodeSelector(coarse_grid)
#selector()
#nodes = selector.nodes

nodes = utils.boundary_layer_selection(coarse_grid, [1,3,5,7], 5)

int_data, int_idx = utils.fetch_highres_data(coarse_grid,
        nodes, fine_grid, U_highres)

solver = multiblock_solvers.AdvectionDiffusionSolver(coarse_grid,
        velocity=velocity, diffusion=diffusion,
        internal_data = int_data, internal_indices = int_idx)

solver.set_boundary_condition(1,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(3,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(5,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(7,{'type': 'dirichlet', 'data': g})
solver.solve(tspan)

U = []
for frame in np.transpose(solver.sol.y):
    U.append(grid2d.array_to_multiblock(coarse_grid, frame))

animation.animate_multiblock(coarse_grid, U)

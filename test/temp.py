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


#blocks = grid2d.load_p3d('cyl50.p3d')
N = 101
blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
          utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
#x1 = np.linspace(-1,0,N)
#y1 = np.linspace(0,1,N)
#x2 = np.linspace(0,1,N)
#y2 = np.linspace(0,1,N)
#X1,Y1 = np.meshgrid(x1,y1,indexing='ij')
#X2,Y2 = np.meshgrid(x2,y2,indexing='ij')
#blocks = [(X1,Y1),(X2,Y2)]
grid2d.collocate_corners(blocks)
grid = grid2d.MultiblockSBP(blocks, accuracy=4)
init = [ np.zeros(shape) for shape in grid.get_shapes() ]
#for (k, (X,Y)) in enumerate(grid.get_blocks()):
#    init[k] = 0.1*norm.pdf(X,loc=-0.5,scale=0.2)*norm.pdf(Y,loc=0.5,scale=0.2)

def g(t,x,y):
    return 1

velocity = np.array([0.0,0.0])

solver = multiblock_solvers.AdvectionDiffusionSolver(grid, initial_data=init,
                                                     velocity = velocity)
solver.set_boundary_condition(1,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(3,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(5,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(7,{'type': 'dirichlet', 'data': g})
tspan = (0.0, 2.5)
import time

start = time.time()
solver.solve(tspan)
end = time.time()
print("Elapsed time: " + str(end - start))

U = []
for frame in np.transpose(solver.sol.y):
    U.append(grid2d.array_to_multiblock(grid, frame))

animation.animate_multiblock(grid, U, stride=4)

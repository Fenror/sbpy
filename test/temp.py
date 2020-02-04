import sys
sys.path.append('..')
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers


blocks = grid2d.load_p3d('cyl250.p3d')
grid = grid2d.MultiblockSBP(blocks)
init = [ np.ones(shape) for shape in grid.get_shapes() ]
for (k, (X,Y)) in enumerate(grid.get_blocks()):
    init[k] = 0.1*norm.pdf(Y,loc=0.0,scale=0.05)*norm.pdf(X,loc=-0.6,scale=0.05)

solver = multiblock_solvers.AdvectionSolver(grid, initial_data=init)
tspan = (0.0, 0.7)
solver.solve(tspan)
fin = grid2d.array_to_multiblock(grid, solver.sol.y[:,-1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for (i,(x,y)) in enumerate(grid.get_blocks()):
    ax.plot_surface(x,y,fin[i])

plt.xlabel('x')
plt.show()

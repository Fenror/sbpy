import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers


X_blocks,Y_blocks = grid2d.load_p3d('cyl50.p3d')
foo = grid2d.Multiblock(X_blocks,Y_blocks)
bar = multiblock_solvers.AdvectionSolver(foo)
bar.solve()
fin = np.reshape(bar.sol.y[:,25],(4,50,50))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for (i,(x,y)) in enumerate(zip(X_blocks,Y_blocks)):
    ax.plot_surface(x,y,fin[i])

plt.xlabel('x')
plt.show()

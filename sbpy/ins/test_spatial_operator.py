import pdb

import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from ins import spatial_operator

(X,Y) = get_circle_sector_grid(5, 0.0, 3.14/2, 0.2, 1.0)
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid)

l1,l2,l3 = spatial_operator(sbp, np.array([X]), np.array([Y]), np.array([Y]), 0.1)

plt.quiver(X, Y, -l1[0], -l2[0])
plt.show()

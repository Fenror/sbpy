import pdb

import numpy as np
import matplotlib.pyplot as plt
import scipy

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from ins import spatial_operator, F, flat_to_struct

(X,Y) = get_circle_sector_grid(5, 0.0, 3.14/2, 0.2, 1.0)
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid)


def foo(x):
    L,J = F(sbp, x, X, Y, 0.1)
    return L


row = 0
x0 = np.array([X, Y, X, X, Y, X]).flatten()
bla = scipy.optimize.approx_fprime(x0, lambda x: foo(x)[row], 0.000001)

L,J = F(sbp, x0, X, Y, 0.1)
print(J[row,:].todense().flatten() - bla)
print(np.linalg.norm(J[row,:].todense().flatten() - bla, ord=np.inf))

#print(foo(np.ones(75)))

#print(J)
#plt.spy(J)
#plt.quiver(X, Y, -l1[0], -l2[0])
#plt.show()

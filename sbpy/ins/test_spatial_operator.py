import pdb

import numpy as np
import matplotlib.pyplot as plt
import scipy

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from ins import spatial_operator

(X,Y) = get_circle_sector_grid(5, 0.0, 3.14/2, 0.2, 1.0)
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid)

#(l1,l2,l3), J = spatial_operator(sbp, np.array([X]), np.array([Y]), np.array([Y]))

def foo(x):
    u = np.array([np.reshape(x[0:25], (5,5))])
    v = np.array([np.reshape(x[25:50], (5,5))])
    p = np.array([np.reshape(x[50:75], (5,5))])
    L,J = spatial_operator(sbp, u, v, p)
    return L.flatten()


bla = scipy.optimize.approx_fprime(np.ones(75), lambda x: foo(x)[0], 0.000001)
print(bla)

x = np.ones(75)
u = np.array([np.reshape(x[0:25], (5,5))])
v = np.array([np.reshape(x[25:50], (5,5))])
p = np.array([np.reshape(x[50:75], (5,5))])
L,J = spatial_operator(sbp, u, v, p)
print(J[0,:].todense().flatten() - bla)

#print(foo(np.ones(75)))

#print(J)
#plt.spy(J)
#plt.quiver(X, Y, -l1[0], -l2[0])
#plt.show()

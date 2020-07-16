import pdb

import numpy as np
import matplotlib.pyplot as plt
import scipy

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from ins import spatial_operator, F, flat_to_struct

(X,Y) = get_circle_sector_grid(55, 0.0, 3.14/2, 0.2, 1.0)
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid)

dt = 0.05

rv = scipy.stats.multivariate_normal([0.5, 0.5], np.eye(2))
bla = rv.pdf(np.dstack((X,Y)))
initu = bla
initv = np.array(np.zeros(X.shape))
initp = np.array(np.ones(X.shape))

tol = 1e-10
err = 1

Psol=[]
Usol=[]
Vsol=[]

sol = np.array([initu, initv, initp, initu, initv, initp]).flatten()
for k in range(10):
    while True:
        L, J = F(sbp, sol, initu, initv, dt)
        err = np.linalg.norm(L, ord=np.inf)
        print(err)
        if err < tol:
            break
        delta = scipy.sparse.linalg.spsolve(J, L)
        sol = sol - delta

    U,V,P = flat_to_struct(grid, sol)
    Psol.append(P[0][0])
    Usol.append(U[0][0])
    Vsol.append(V[0][0])
    initu = U[1][0]
    initv = V[1][0]

plt.quiver(X,Y,Usol[-1],Vsol[-1])
#plt.quiver(X,Y,initu,initv)
plt.show()

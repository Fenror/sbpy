import pdb
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from ins import spatial_operator, Ftest

Nx = 25
Ny = 25

(X,Y) = get_circle_sector_grid(Nx, 0.0, 3.14/2, 0.2, 1.0)
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid)

dt = 0.1

rv = scipy.stats.multivariate_normal([0.4, 0.4], 0.01*np.eye(2))
bla = rv.pdf(np.dstack((X,Y)))
initu = np.array([Y])
initv = -np.array([X])
#initu = 0.0001*np.array([bla])
#initv = -0.0001*np.array([bla])
#initv = np.array([np.zeros(X.shape)])
initp = np.array([np.ones(X.shape)])
fig, ax = plt.subplots(1,1)
ax.quiver(X,Y,initu[0],initv[0])
plt.show()

tol = 1e-10
err = 1

Psol=[]
Usol=[]
Vsol=[]

sol = np.array([initu, initv, initp]).flatten()
nt = 30
for k in range(nt):
    while True:
        L, J = Ftest(sbp, sol, initu, initv, dt)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break
        delta = scipy.sparse.linalg.spsolve(J, L)
        sol = sol - delta

    print("Iter {}, error: {}".format(k, err))

    sol = np.reshape(sol, (3,1,Nx,Ny))
    Usol.append(sol[0][0])
    Vsol.append(sol[1][0])
    Psol.append(sol[2][0])
    initu = sol[0]
    initv = sol[1]
    sol = sol.flatten()

fig, ax = plt.subplots(1,1)
velocity_plot = ax.quiver(X,Y,Usol[0],Vsol[0])
#pressure_plot = ax.pcolormesh(X,Y,Psol[-1])

def update_quiver(num, velocity_plot):
    U = Usol[num%nt]
    V = Vsol[num%nt]
    velocity_plot.set_UVC(U,V)
    ax.set_title("t = {:.2f}".format((num%nt)*dt))

    return velocity_plot,

def update_contour(num, pressure_plot):
    P = Psol[num%nt]
    ax.clear()
    pressure_plot = ax.pcolormesh(X,Y,P)
    ax.set_title("t = {:.2f}".format((num%nt)*dt))

    return pressure_plot,

anim = animation.FuncAnimation(fig, update_quiver,
                               fargs=(velocity_plot,),
                               interval=1000*dt, blit=False)
#anim = animation.FuncAnimation(fig, update_contour, fargs=(pressure_plot,), interval=1000*dt, blit=False)

#fig, axes = plt.subplots(1,2)
#axes[0].quiver(X,Y,Usol[0],Vsol[0])
#axes[1].quiver(X,Y,Usol[-1],Vsol[-1])
plt.show()

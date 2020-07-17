import pdb
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, backward_euler, vec_to_tensor

Nx = 25
Ny = 25

(X,Y) = get_circle_sector_grid(Nx, 0.0, 3.14/2, 0.2, 1.0)
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid)


initu = np.array([Y])
initv = -np.array([X])
initp = np.array([np.ones(X.shape)])
plt.quiver(X,Y,initu[0],initv[0])
plt.show()

def spatial_op(state):
    S,J = euler_operator(sbp, state) + \
          wall_operator(sbp, state, 0, 'w') + \
          wall_operator(sbp, state, 0, 'e') + \
          wall_operator(sbp, state, 0, 's') + \
          wall_operator(sbp, state, 0, 'n')

    return S, J

tol = 1e-10
err = 1

Psol=[]
Usol=[]
Vsol=[]

sol = np.array([initu, initv, initp]).flatten()
nt = 30
dt = 0.1
for k in range(nt):
    sol = backward_euler(spatial_op, sol, dt, tol)

    sol = vec_to_tensor(sbp.grid, sol)
    Usol.append(sol[0][0])
    Vsol.append(sol[1][0])
    Psol.append(sol[2][0])
    sol = sol.flatten()

fig, ax = plt.subplots(1,1)
velocity_plot = ax.quiver(X,Y,Usol[0],Vsol[0])

def update_quiver(num, velocity_plot):
    U = Usol[num%nt]
    V = Vsol[num%nt]
    velocity_plot.set_UVC(U,V)
    ax.set_title("t = {:.2f}".format((num%nt)*dt))

    return velocity_plot,

anim = animation.FuncAnimation(fig, update_quiver,
                               fargs=(velocity_plot,),
                               interval=1000*dt, blit=False)

plt.show()

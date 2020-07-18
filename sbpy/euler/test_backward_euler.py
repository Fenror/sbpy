import pdb
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy
from tqdm import tqdm

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, backward_euler, vec_to_tensor, outflow_operator, pressure_operator, sbp_in_time
from animation import animate_pressure, animate_velocity, animate_solution

Nx = 55
Ny = 55

(X,Y) = get_circle_sector_grid(Nx, 0.0, 3.14, 0.2, 1.0)
#(X,Y) = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid, accuracy=4)


initu = np.array([Y])
initv = -np.array([X])
rv = scipy.stats.multivariate_normal([0.0, 0.5], 0.01*np.eye(2))
gauss_bell = rv.pdf(np.dstack((X,Y)))
#initu = 2*np.array([gauss_bell])/np.max(gauss_bell.flatten())
initu = -np.array([0*X])
initv = 2*np.array([gauss_bell])/np.max(gauss_bell.flatten())
#initv = -np.array([0*X])
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

tol = 1e-12
err = 1

Psol=[]
Usol=[]
Vsol=[]

sol = np.array([initu, initv, initp]).flatten()
nt = 100
dt = 0.1
for k in tqdm(range(nt)):
    try:
        sol = backward_euler(spatial_op, sol, dt, tol)
    except:
        sol = backward_euler(spatial_op, sol, 0.1*dt, tol)

    sol = vec_to_tensor(sbp.grid, sol)
    Usol.append(sol[0][0])
    Vsol.append(sol[1][0])
    Psol.append(sol[2][0])
    sol = sol.flatten()

#animate_pressure(sbp.grid, Psol, dt)
#animate_velocity(sbp.grid, Usol, Vsol, dt)
animate_solution(sbp.grid, Usol, Vsol, Psol, dt)

import pdb
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy
from tqdm import tqdm

from sbpy.utils import get_circle_sector_grid, get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, backward_euler, vec_to_tensor, outflow_operator, pressure_operator, sbp_in_time, inflow_operator
from animation import animate_pressure, animate_velocity, animate_solution

##Resolution
Nx = 15
Ny = 15


##Grid

#(X,Y) = get_circle_sector_grid(Nx, 0.0, 3.14/2, 0.2, 1.0)

#(X,Y) = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
#X = np.transpose(X)
#Y = np.transpose(Y)

(X,Y) = get_bump_grid(Nx)

grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid, accuracy=2)


##Initial data

##Rotating velocity field
#initu = np.array([Y])
#initv = -np.array([X])

##Colliding whirls
#rv1 = scipy.stats.multivariate_normal([-0.5, 0.4], 0.01*np.eye(2))
#gauss_bell1 = rv1.pdf(np.dstack((X,Y)))
#rv2 = scipy.stats.multivariate_normal([0.5, 0.4], 0.01*np.eye(2))
#gauss_bell2 = -rv2.pdf(np.dstack((X,Y)))
#initu = 2*np.array([gauss_bell1 + gauss_bell2])/np.max(gauss_bell1.flatten())
#initv = np.array([0*X])

##Single whirl
#rv = scipy.stats.multivariate_normal([0.3, 0.4], 0.01*np.eye(2))
#gauss_bell = rv.pdf(np.dstack((X,Y)))
#initu = 2*np.array([gauss_bell])/np.max(gauss_bell.flatten())
#initv = np.array([0*X])


#Constant velocity
initu = np.array([np.ones(X.shape)])
initv = np.array([0*X])


initp = np.array([np.ones(X.shape)])

plt.quiver(X,Y,initu[0],initv[0])
plt.show()


##Build spatial operator
def spatial_op(state):
    S,J = euler_operator(sbp, state) + \
          inflow_operator(sbp, state, 0, 'w', -1, 0) + \
          pressure_operator(sbp, state, 0, 'e') + \
          wall_operator(sbp, state, 0, 's') + \
          wall_operator(sbp, state, 0, 'n')

    return S, J



##Solve
Psol=[]
Usol=[]
Vsol=[]
sol = np.array([initu, initv, initp]).flatten()
nt = 150
dt = 0.1
tol = 1e-12
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

##Plot solution
#animate_pressure(sbp.grid, Psol, dt)
animate_velocity(sbp.grid, Usol, Vsol, dt)
#animate_solution(sbp.grid, Usol, Vsol, Psol, dt)

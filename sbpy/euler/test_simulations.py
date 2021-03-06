import unittest,pdb

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid, get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
from sbpy.euler.animation import animate_pressure, animate_velocity, animate_solution, animate_speed, plot_speed, plot_velocity
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, outflow_operator, solve


def get_gauss_initial_data(X, Y, cx, cy):
    rv1 = multivariate_normal([cx,cy], 0.01*np.eye(2))
    gauss_bell = rv1.pdf(np.dstack((X,Y)))
    normalize = np.max(gauss_bell.flatten())
    initu = 2*np.array([gauss_bell])/normalize
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])

    return initu, initv, initp


def bump_const_inflow_pressure_outflow(
        Nx = 49,
        Ny = 17,
        num_timesteps = 10,
        dt = 0.1):

    X,Y = get_bump_grid(Nx,Ny)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              inflow_operator(sbp, state, 0, 'w', -1, 0) + \
              pressure_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def bump_const_inflow_pressure_speed_outflow(
        Nx = 49,
        Ny = 17,
        num_timesteps = 10,
        dt = 0.1):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              inflow_operator(sbp, state, 0, 'w', -1, 0) + \
              outflow_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def bump_walls_and_pressure_speed_outflow(
        Nx = 49,
        Ny = 17,
        num_timesteps = 10,
        dt = 0.1):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,-0.5,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_walls_and_pressure_speed_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_pressure_speed_outflow_everywhere(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              outflow_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              outflow_operator(sbp, state, 0, 's') + \
              outflow_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def circle_sector_pressure_speed_outflow_everywhere(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        angle = 0.5*np.pi):

    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              outflow_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              outflow_operator(sbp, state, 0, 's') + \
              outflow_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_cavity_flow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              wall_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def circle_sector_cavity_flow(
        N = 40,
        num_timesteps = 30,
        dt = 0.1,
        angle = 0.5*np.pi):

    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              wall_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


#SELECT ONE OF THE FUNCTIONS BELOW AND RUN SCRIPT
#-----------------------------------
#bump_const_inflow_pressure_outflow()
#bump_const_inflow_pressure_speed_outflow()
#bump_walls_and_pressure_speed_outflow()
#square_walls_and_pressure_speed_outflow()
#square_pressure_speed_outflow_everywhere()
#circle_sector_pressure_speed_outflow_everywhere()
#square_cavity_flow()
#circle_sector_cavity_flow()

if __name__ == '__main__':
    #grid,U,V,P,dt = circle_sector_pressure_speed_outflow_everywhere(num_timesteps=50)
    grid,U,V,P,dt = circle_sector_cavity_flow(num_timesteps=400, dt=0.04)
    animate_solution(grid,U,V,dt,save_gif=True)



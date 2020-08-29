import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

def animate_pressure(grid, P, dt):
    nt = len(P)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    plot = ax.pcolormesh(X,Y,P[0])

    def update(num, pressure_plot):
        p = P[num%nt]
        pressure_plot = ax.pcolormesh(X,Y,p)
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return pressure_plot,

    anim = animation.FuncAnimation(fig, update, fargs=(plot,), interval=1000*dt, blit = True)
    plt.show()


def animate_velocity(grid, U, V, dt):
    nt = len(U)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    plot = ax.quiver(X,Y,U[0],V[0])

    def update(num, plot):
        u = U[num%nt]
        v = V[num%nt]
        plot.set_UVC(u,v)
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return plot,

    anim = animation.FuncAnimation(fig, update, fargs=(plot,), interval=1000*dt, blit = True)
    plt.show()


def animate_solution(grid, U, V, dt):
    nt = len(U)
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    X,Y = grid.get_block(0)
    S = np.sqrt(np.array(U)**2 + np.array(V)**2)
    s_plot = ax.pcolormesh(X,Y,S[0])
    w_plot = ax.quiver(X,Y,U[0],V[0])
    s_min = np.min(np.array(S).flatten())
    s_max = np.max(np.array(S).flatten())
    s_plot.set_clim([s_min, s_max])
    fig.colorbar(s_plot, ax=ax)

    def update(num, s_plot, w_plot):
        u = U[num%nt]
        v = V[num%nt]
        s = S[num%nt]

        s_plot.set_array(s[:-1,:-1].ravel())
        w_plot.set_UVC(u,v)
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return s_plot, w_plot

    anim = animation.FuncAnimation(fig, update, fargs=(s_plot, w_plot), interval=1000*dt, blit = True)
    plt.show()


def animate_speed(grid, U, V, dt):
    nt = len(U)
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    X,Y = grid.get_block(0)
    S = np.sqrt(np.array(U)**2 + np.array(V)**2)
    s_plot = ax.pcolormesh(X,Y,S[0])
    s_min = np.min(np.array(S).flatten())
    s_max = np.max(np.array(S).flatten())
    s_plot.set_clim([s_min, s_max])
    fig.colorbar(s_plot, ax=ax)

    def update(num, s_plot):
        u = U[num%nt]
        v = V[num%nt]
        s = S[num%nt]

        s_plot.set_array(s[:-1,:-1].ravel())
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return s_plot,

    anim = animation.FuncAnimation(fig, update, fargs=(s_plot,), interval=1000*dt, blit = True)
    plt.show()


def plot_speed(grid, U, V):
    matplotlib.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    X,Y = grid.get_block(0)
    S = np.sqrt(np.array(U)**2 + np.array(V)**2)
    s_plot = ax.pcolormesh(X, Y, S,
            edgecolors = 'face', antialiased = True,
            shading='gouraud')
    s_min = np.min(np.array(S).flatten())
    s_max = np.max(np.array(S).flatten())
    s_plot.set_clim([s_min, s_max])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s_plot, cax=cax)
    plt.show()


def plot_velocity(grid, U, V):
    matplotlib.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    X,Y = grid.get_block(0)
    S = np.sqrt(np.array(U)**2 + np.array(V)**2)
    s_plot = ax.pcolormesh(X, Y, S,
            edgecolors = 'face', antialiased = True,
            shading='gouraud')
    s_min = np.min(np.array(S).flatten())
    s_max = np.max(np.array(S).flatten())
    s_plot.set_clim([s_min, s_max])
    w_plot = ax.quiver(X,Y,U,V)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s_plot, cax=cax)
    plt.show()

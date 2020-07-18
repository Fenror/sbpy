import matplotlib.pyplot as plt
from matplotlib import animation

def animate_pressure(grid, P, dt):
    nt = len(P)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    plot = ax.pcolormesh(X,Y,P[0])

    def update(num, pressure_plot):
        p = P[num%nt]
        ax.clear()
        pressure_plot = ax.pcolormesh(X,Y,p)
        ax.set_title("t = {:.2f}".format((num%nt)*dt))

        return pressure_plot,

    anim = animation.FuncAnimation(fig, update, fargs=(plot,), interval=1000*dt)
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
        ax.set_title("t = {:.2f}".format((num%nt)*dt))

        return plot,

    anim = animation.FuncAnimation(fig, update, fargs=(plot,), interval=1000*dt)
    plt.show()


def animate_solution(grid, U, V, P, dt):
    nt = len(U)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    p_plot = ax.pcolormesh(X,Y,P[0])
    w_plot = ax.quiver(X,Y,U[0],V[0])
    fig.colorbar(p_plot, ax=ax)

    def update(num, p_plot, w_plot):
        u = U[num%nt]
        v = V[num%nt]
        p = P[num%nt]
        ax.clear()
        p_plot = ax.pcolormesh(X,Y,p)
        w_plot = ax.quiver(X,Y,u,v)
        ax.set_title("t = {:.2f}".format((num%nt)*dt))

        return p_plot, w_plot

    anim = animation.FuncAnimation(fig, update, fargs=(p_plot, w_plot), interval=1000*dt)
    plt.show()


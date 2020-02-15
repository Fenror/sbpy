""" This module contains visualization tools for PDE solutions. """

import itertools
import numpy as np
from mayavi import mlab


def animate_multiblock(grid, F, **kwargs):
    """ Animates a list of multiblock functions.

    Arguments:
        F: A list of multiblock functions.

    Optional:
        fps: A positive integer representing the number of frames per second
        stored in F.

    """

    if 'fps' in kwargs:
        fps = kwargs['fps']
    else:
        fps = 30

    Fmin = np.min(np.array(F))
    Fmax = np.max(np.array(F))
    xmin = np.min(np.array([X for X,Y in grid.get_blocks()]))
    xmax = np.max(np.array([X for X,Y in grid.get_blocks()]))
    ymin = np.min(np.array([Y for X,Y in grid.get_blocks()]))
    ymax = np.max(np.array([Y for X,Y in grid.get_blocks()]))
    surfaces = [ mlab.mesh(X, Y, Z, vmax = Fmax, vmin = Fmin) for ((X,Y),Z) in
                 zip(grid.get_blocks(), F[0]) ]


    @mlab.animate(delay=int(1000/fps))
    def anim():
        for f in itertools.cycle(F):
            for (s,f_block) in zip(surfaces, f):
                s.mlab_source.trait_set(scalars=f_block)
                s.mlab_source.trait_set(z=f_block)
            yield

    mlab.axes(x_axis_visibility = True,
              y_axis_visibility = True,
              z_axis_visibility = True,
              extent=[xmin,xmax,ymin,ymax,Fmin,Fmax])

    frame_gen = anim()
    mlab.show()

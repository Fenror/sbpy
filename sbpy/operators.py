"""This module contains functions for getting SBP operators."""

import numpy as np
from scipy import sparse

def get_fd_op(num_grid_points, h, accuracy=2):
    """Get finite difference operator.

    For now supports only 2nd order operator.

    Args:
        num_grid_points: The number of grid points.
        h: Distance between consecutive grid points.
        accuracy: The interior accuracy of the operator. Currently only supports
            the default value 2.

    Returns:
        D, P, Q: Here D is the SBP operator as a numpy 2d array.
                 P is the integration matrix.
                 Q is such that D = P^(-1) Q
    """

    if accuracy != 2:
        raise ValueError('Accuracy must be 2')

    N = num_grid_points
    stencil = np.array([-0.5, 0.0, 0.5])
    MID = \
        sparse.diags(stencil, [0, 1, 2], shape=(N-2, N))

    TOP = sparse.bsr_matrix(([-0.5, 0.5], ([0, 0], [0, 1])),
                            shape=(1, N))
    BOT = sparse.bsr_matrix(([-0.5, 0.5], ([0, 0], [N-2, N-1])),
                            shape=(1, N))

    Q = sparse.vstack([TOP, MID, BOT])
    p = np.ones(N)
    p = h*p
    p[0] = 0.5*h
    p[-1] = 0.5*h
    p_inv = 1/p
    P = sparse.diags([p], [0])
    P_inv = sparse.diags([p_inv], [0])
    D = P_inv*Q

    return D,P,Q

D,P,Q = get_fd_op(5,1/4,2)

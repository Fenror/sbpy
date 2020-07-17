"""
Differential and SAT operators for Euler. A state should be thought of as a numpy array [U, V, P], where U, V, P are multiblock functions (i.e. lists of grid functions). The operators here operate on flattened versions of such a state. I.e. state = np.array([U, V, P]).flatten().

A complete spatial operator can be built by using the euler_operator function
together with boundary operators, for example, let sbp be a MultiblockSBP. Then:

def op(state):
    S,J = euler_operator(sbp, state) +
          wall_operator(sbp, state, 0, 'w') +
          wall_operator(sbp, state, 0, 'e') +
          wall_operator(sbp, state, 0, 's') +
          wall_operator(sbp, state, 0, 'n')
    return S,J

defines a spatial operator with wall conditions, which returns the operator and
its jacobian evaluated at the given state.

The system can then be integrated using the backward_euler function, or the
sbp_in_time function.
"""

import pdb

import numpy as np
from scipy import sparse

from sbpy import operators

def vec_to_tensor(grid, vec):
    shapes = grid.get_shapes()
    component_length = np.prod([Nx*Ny for (Nx,Ny) in shapes])
    vec = np.reshape(vec, (3,component_length))

    start = 0
    U = []
    V = []
    P = []
    for Nx,Ny in shapes:
        U.append(np.reshape(vec[0][start:(start+Nx*Ny)], (Nx,Ny)))
        V.append(np.reshape(vec[1][start:(start+Nx*Ny)], (Nx,Ny)))
        P.append(np.reshape(vec[2][start:(start+Nx*Ny)], (Nx,Ny)))

    return np.array([U, V, P])

def euler_operator(sbp, state):
    """ The Euler spatial operator.
    Arguments:
        sbp - A MultilbockSBP object.
        state - A state vector

    Returns:
        S - The euler operator evaluated at the given state.
        J - The Jacobian of S at the given state.
    """

    u,v,p = vec_to_tensor(sbp.grid, state)

    dudx = sbp.diffx(u)
    dudy = sbp.diffy(u)
    duudx = sbp.diffx(u*u)
    dvdx = sbp.diffx(v)
    dvdy = sbp.diffy(v)
    dvvdy = sbp.diffy(v*v)
    duvdx = sbp.diffx(u*v)
    duvdy = sbp.diffy(u*v)
    dpdx = sbp.diffx(p)
    dpdy = sbp.diffy(p)

    l1 = 0.5*(u*dudx + v*dudy + duudx + duvdy) + dpdx
    l2 = 0.5*(u*dvdx + v*dvdy + dvvdy + duvdx) + dpdy
    l3 = dudx + dvdy

    L = np.array([l1,l2,l3]).flatten()

    #Jacobian
    Dx = sbp.get_Dx(0)
    Dy = sbp.get_Dy(0)
    U = sparse.diags(u[0].flatten())
    V = sparse.diags(v[0].flatten())
    Ux = sparse.diags(dudx.flatten())
    Uy = sparse.diags(dudy.flatten())
    Vx = sparse.diags(dvdx.flatten())
    Vy = sparse.diags(dvdy.flatten())

    dl1du = 0.5*(U@Dx + Ux + V@Dy + 2*Dx@U + Dy@V)
    dl1dv = 0.5*(Uy + Dy@U)
    dl1dp = Dx
    dl2du = 0.5*(Vx + Dx@V)
    dl2dv = 0.5*(U@Dx + V@Dy + Vy + Dx@U + 2*Dy@V)
    dl2dp = Dy
    dl3du = Dx
    dl3dv = Dy
    dl3dp = None

    J = sparse.bmat([[dl1du, dl1dv, dl1dp],
                     [dl2du, dl2dv, dl2dp],
                     [dl3du, dl3dv, dl3dp]])

    J = sparse.csr_matrix(J)

    return np.array([L, J], dtype=object)


def wall_operator(sbp, state, block_idx, side):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    u_bd = u[block_idx][bd_slice]
    v_bd = v[block_idx][bd_slice]
    wn = u_bd*nx + v_bd*ny
    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = -0.5*lift*u_bd*wn
    s2[block_idx][bd_slice] = -0.5*lift*v_bd*wn
    s3[block_idx][bd_slice] = -lift*wn

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = 0.5*lift*(u_bd*nx + wn)
    ds1du = sparse.diags(ds1du.flatten())
    ds1dv = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = 0.5*lift*u_bd*ny
    ds1dv = sparse.diags(ds1dv.flatten())
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = 0.5*lift*v_bd*nx
    ds2du = sparse.diags(ds2du.flatten())
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = 0.5*lift*(v_bd*ny + wn)
    ds2dv = sparse.diags(ds2dv.flatten())
    ds2dp = np.zeros((Nx,Ny))
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = np.zeros((Nx,Ny))
    ds3du[bd_slice] = lift*nx
    ds3du = sparse.diags(ds3du.flatten())
    ds3dv = np.zeros((Nx,Ny))
    ds3dv[bd_slice] = lift*ny
    ds3dv = sparse.diags(ds3dv.flatten())
    ds3dp = np.zeros((Nx,Ny))
    ds3dp = sparse.diags(ds3dp.flatten())


    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def pressure_operator(sbp, state, block_idx, side):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    p_bd = p[block_idx][bd_slice]

    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s1[block_idx][bd_slice] = -lift*nx*(p_bd-1)
    s2 = np.zeros((num_blocks,Nx,Ny))
    s2[block_idx][bd_slice] = -lift*ny*(p_bd-1)
    s3 = np.zeros((num_blocks,Nx,Ny))

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dp = np.zeros((Nx,Ny))
    ds1dp[bd_slice] = lift*nx
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dp = np.zeros((Nx,Ny))
    ds2dp[bd_slice] = lift*ny
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dp = sparse.csr_matrix((Nx*Ny, Nx*Ny))


    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def outflow_operator(sbp, state, block_idx, side):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    u_bd = u[block_idx][bd_slice]
    v_bd = v[block_idx][bd_slice]
    p_bd = p[block_idx][bd_slice]
    wn = u_bd*nx + v_bd*ny

    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))
    s1[block_idx][bd_slice] = -lift*nx*(0.5*u_bd**2 + 0.5*v_bd**2 + p_bd)
    s2[block_idx][bd_slice] = -lift*ny*(0.5*u_bd**2 + 0.5*v_bd**2 + p_bd)

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = nx*lift*u_bd
    ds1du = sparse.diags(ds1du.flatten())

    ds1dv = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = nx*lift*v_bd
    ds1dv = sparse.diags(ds1dv.flatten())

    ds1dp = np.zeros((Nx,Ny))
    ds1dp[bd_slice] = nx*lift
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = ny*lift*u_bd
    ds2du = sparse.diags(ds2du.flatten())

    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = ny*lift*v_bd
    ds2dv = sparse.diags(ds2dv.flatten())

    ds2dp = np.zeros((Nx,Ny))
    ds2dp[bd_slice] = ny*lift
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dp = sparse.csr_matrix((Nx*Ny, Nx*Ny))


    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])


    return np.array([S,J], dtype=object)


def backward_euler(op, prev_state, dt, tol):
    N = int(len(prev_state)/3)

    def F(new_state, prev_state):
        T = np.concatenate(
                [(new_state[0:int(2*N)] - prev_state[0:int(2*N)])/dt,
                 np.zeros(N)])
        S,Js = op(new_state)

        #Jacobian
        I = sparse.identity(N)
        O = sparse.csr_matrix((N,N))
        Jt = (1/dt)*sparse.bmat([[I, O, O],
                                 [O, I, O],
                                 [O, O, O]])

        return T+S, Jt+Js

    new_state = prev_state.copy()
    while True:
        L, J = F(new_state, prev_state)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        delta = sparse.linalg.spsolve(J,L)
        new_state -= delta

    #print("Error: {:.2e}".format(err))
    return new_state


def sbp_in_time(op, cur_state, dt, tol):
    N = int(len(cur_state)/3)

    def F(prev_state, next_state):
        T = np.concatenate([(next_state[0:2*N] - 0.75*prev_state[0:2*N] - 0.25*cur_state[0:2*N])/dt,
                     np.zeros(N),
                     (next_state[0:2*N] - prev_state[0:2*N])/dt,
                     np.zeros(N)])


        S0,Js0 = op(prev_state)
        S1,Js1 = op(next_state)
        S = np.concatenate([S0, S1])

        #Jacobian
        I = sparse.identity(N)
        O = sparse.csr_matrix((N,N))
        Jt = (1/dt)*sparse.bmat([[-0.75*I,       O, O, I, O, O],
                                 [      O, -0.75*I, O, O, I, O],
                                 [      O,       O, O, O, O, O],
                                 [     -I,       O, O, I, O, O],
                                 [      O,      -I, O, O, I, O],
                                 [      O,       O, O, O, O, O]])

        J = Jt + sparse.bmat([[Js0, None],
                              [None, Js1]])

        return T+S,J

    prev_state = cur_state.copy()
    new_state = cur_state.copy()
    while True:
        L, J = F(prev_state, new_state)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        delta = sparse.linalg.spsolve(J,L)
        prev_state -= delta[0:3*N]
        new_state -= delta[3*N:]

    print("Error: {:.2e}".format(err))
    return new_state


def F(sbp, vec, u0, v0, dt):
    (Nx,Ny) = sbp.grid.get_shapes()[0]
    vec = np.reshape(vec, (2,3,1,Nx,Ny))
    ucur = vec[0][0]
    vcur = vec[0][1]
    pcur = vec[0][2]
    unext = vec[1][0]
    vnext = vec[1][1]
    pnext = vec[1][2]
    L0,J0 = spatial_operator(sbp, ucur, vcur, pcur)
    L1,J1 = spatial_operator(sbp, unext, vnext, pnext)

    T = np.array([(unext - ucur)/dt + 0.25*(ucur - u0)/dt,
                 (vnext - vcur)/dt + 0.25*(vcur - v0)/dt,
                 np.zeros((1,Nx,Ny)),
                 (unext - ucur)/dt,
                 (vnext - vcur)/dt,
                 np.zeros((1,Nx,Ny))]).flatten()

    S = np.array([L0, L1]).flatten()

    #Jacobian
    M = Nx*Ny
    I = sparse.identity(M)
    O = sparse.csr_matrix((M,M))
    Jt = (1/dt)*sparse.bmat([[-0.75*I,       O, O, I, O, O],
                             [      O, -0.75*I, O, O, I, O],
                             [      O,       O, O, O, O, O],
                             [     -I,       O, O, I, O, O],
                             [      O,      -I, O, O, I, O],
                             [      O,       O, O, O, O, O]])

    J = Jt + sparse.bmat([[J0, None],
                          [None, J1]])

    return T+S, J

import pdb

import numpy as np
from scipy import sparse

from sbpy import operators

def spatial_operator(sbp, u, v, p):
    dudx = sbp.diffx(u)
    dudy = sbp.diffy(u)
    duudx = sbp.diffx(u*u)
    #laplace_u = sbp.diffx(dudx) + sbp.diffy(dudy)
    dvdx = sbp.diffx(v)
    dvdy = sbp.diffy(v)
    dvvdy = sbp.diffy(v*v)
    #laplace_v = sbp.diffx(dvdx) + sbp.diffy(dvdy)
    duvdx = sbp.diffx(u*v)
    duvdy = sbp.diffy(u*v)
    dpdx = sbp.diffx(p)
    dpdy = sbp.diffy(p)

    l1 = 0.5*(u*dudx + v*dudy + duudx + duvdy) + dpdx# - epsilon*laplace_u
    l2 = 0.5*(u*dvdx + v*dvdy + dvvdy + duvdx) + dpdy# - epsilon*laplace_v
    l3 = dudx + dvdy

    #Jacobian
    Dx = sbp.get_sbp_ops()[0].Dx
    Dy = sbp.get_sbp_ops()[0].Dy
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

    #Wall penalty
    for side in ['w','e','s','n']:
        bd_slice = sbp.grid.get_boundary_slice(0, side)
        normals = sbp.get_normals(0, side)
        nx = normals[:,0]
        ny = normals[:,1]
        u_bd = u[0][bd_slice]
        v_bd = v[0][bd_slice]
        wn = u_bd*nx + v_bd*ny
        pinv = sbp.get_pinv(0, side)
        bd_quad = sbp.get_boundary_quadrature(0, side)
        lift = pinv*bd_quad
        l1[0][bd_slice] -= 0.5*lift*u_bd*wn
        l2[0][bd_slice] -= 0.5*lift*v_bd*wn
        l3[0][bd_slice] -= lift*wn

        #Jacobian
        ds1du = np.zeros(u[0].shape)
        ds1du[bd_slice] = 0.5*lift*(u_bd*nx + wn)
        ds1du = sparse.diags(ds1du.flatten())
        ds1dv = np.zeros(u[0].shape)
        ds1dv[bd_slice] = 0.5*lift*u_bd*ny
        ds1dv = sparse.diags(ds1dv.flatten())
        ds1dp = np.zeros(u[0].shape)
        ds1dp = sparse.diags(ds1dp.flatten())

        ds2du = np.zeros(u[0].shape)
        ds2du[bd_slice] = 0.5*lift*v_bd*nx
        ds2du = sparse.diags(ds2du.flatten())
        ds2dv = np.zeros(u[0].shape)
        ds2dv[bd_slice] = 0.5*lift*(v_bd*ny + wn)
        ds2dv = sparse.diags(ds2dv.flatten())
        ds2dp = np.zeros(u[0].shape)
        ds2dp = sparse.diags(ds2dp.flatten())

        ds3du = np.zeros(u[0].shape)
        ds3du[bd_slice] = lift*nx
        ds3du = sparse.diags(ds3du.flatten())
        ds3dv = np.zeros(u[0].shape)
        ds3dv[bd_slice] = lift*ny
        ds3dv = sparse.diags(ds3dv.flatten())
        ds3dp = np.zeros(u[0].shape)
        ds3dp = sparse.diags(ds3dp.flatten())


        Jwall = sparse.bmat([[ds1du, ds1dv, ds1dp],
                             [ds2du, ds2dv, ds2dp],
                             [ds3du, ds3dv, ds3dp]])

        J -= Jwall

    return np.array([l1,l2,l3]), J


def F(sbp, vec, u0, v0, dt):
    U,V,P = flat_to_struct(sbp.grid, vec)
    L0,J0 = spatial_operator(sbp, U[0], V[0], P[0])
    L1,J1 = spatial_operator(sbp, U[1], V[1], P[1])

    T = np.array([(U[1][0] - U[0][0])/dt + 0.25*(U[0][0] - u0)/dt,
                 (V[1][0] - V[0][0])/dt + 0.25*(V[0][0] - v0)/dt,
                 np.zeros(P[0][0].shape),
                 (U[1][0] - U[0][0])/dt,
                 (V[1][0] - V[0][0])/dt,
                 np.zeros(P[0][0].shape)]).flatten()

    S = np.array([L0, L1]).flatten()

    #Jacobian
    (Nx,Ny) = U[0][0].shape
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


def flat_to_struct(grid, vec):
    shapes = grid.get_shapes()
    (Nx,Ny) = shapes[0]
    vec = np.reshape(vec, (2,3,1,Nx,Ny))
    U = np.array([vec[0][0], vec[1][0]])
    V = np.array([vec[0][1], vec[1][1]])
    P = np.array([vec[0][2], vec[1][2]])
    return U,V,P

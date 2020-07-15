import pdb

import numpy as np
from scipy import sparse

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

    l1 = 0.5*(u*dudx + v*dudx + duudx + duvdy) + dpdx# - epsilon*laplace_u
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

    #Wall penalty
    for side in ['w','e','s','n']:
        bd_slice = sbp.grid.get_boundary_slice(0, side)
        normals = sbp.get_normals(0, side)
        wn = u[0][bd_slice]*normals[:,0] + v[0][bd_slice]*normals[:,1]
        pinv = sbp.get_pinv(0, side)
        bd_quad = sbp.get_boundary_quadrature(0, side)
        l1[0][bd_slice] -= 0.5*pinv*bd_quad*u[0][bd_slice]*wn
        l2[0][bd_slice] -= 0.5*pinv*bd_quad*v[0][bd_slice]*wn
        l3[0][bd_slice] -= pinv*bd_quad*wn


    return np.array([l1,l2,l3])

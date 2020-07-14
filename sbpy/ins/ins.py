import pdb

import numpy as np

def spatial_operator(sbp, u, v, p, epsilon):
    dudx = sbp.diffx(u)
    duudx = sbp.diffx(u*u)
    dudy = sbp.diffy(u)
    laplace_u = sbp.diffx(dudx) + sbp.diffy(dudy)
    dvdx = sbp.diffx(v)
    dvdy = sbp.diffy(v)
    dvvdy = sbp.diffy(v*v)
    laplace_v = sbp.diffx(dvdx) + sbp.diffy(dvdy)
    duvdx = sbp.diffx(u*v)
    duvdy = sbp.diffy(u*v)
    dpdx = sbp.diffx(p)
    dpdy = sbp.diffy(p)

    l1 = 0.5*(u*dudx + v*dudx + duudx + duvdy) + dpdx - epsilon*laplace_u
    l2 = 0.5*(u*dvdx + v*dvdy + dvvdy + duvdx) + dpdy - epsilon*laplace_v
    l3 = dudx + dvdy

    return np.array([l1,l2,l3])

import sys
sys.path.append('..')
import numpy as np
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers

#xi = np.linspace(0,1,10)
#eta = np.linspace(0,1,7)
#XI,ETA = np.meshgrid(xi,eta,indexing='ij')
#X = XI-ETA
#Y = XI+ETA
#foo = operators.SBP2D(X,Y)
#import matplotlib.pyplot as plt
#print(np.diag(foo.P.todense()))
#print(foo.jac)
##print(foo.normals['w'])
##print(foo.normals['e'])
##print(foo.normals['s'])
##print(foo.normals['n'])
#tangent = np.array([X[0,0]-X[-1,0],Y[0,0]-Y[-1,0]])
#n = foo.normals['s'][0]
#foo.plot()

X,Y = grid2d.load_p3d('cyl.p3d')
#foo = operators.SBP2D(X[0],Y[0])
#foo.plot()
foo = grid2d.Multiblock(X,Y)
foo.plot_domain()

bar = multiblock_solvers.AdvectionSolver(foo)


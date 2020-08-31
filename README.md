# Sbpy
A 2D finite difference library for python based on curvilinear summation-by-parts operators.

## How does it work?
Sbpy deals with structured curvilinear multiblock grids. 2D numpy arrays X and Y are used to define a structured grid on a block. The SBP2D class is then instantiated using X,Y, and handles discrete differentiation. The MultiblockSBP class provides a number of useful functions for constructing and operating on multiblock grids (see docstrings for details).

# EXAMPLE
![](sbpy/demo/animation.gif)

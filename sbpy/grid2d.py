""" This module contains functions and classes for managing 2D grids. The
conceptual framework used throughout the module is that 2D numpy arrays represent
function evaluations associated to some grid. For example, if f is an Nx-by-Ny
numpy array, then f[i,j] is interpreted as the evaluation of some function f in
an associated grid node (x_ij, y_ij). 2D numpy arrays representing function
evaluations on a grid are called 'grid functions'. We refer to the boundaries of
a grid function as 's' for south, 'e' for east, 'n' for north, and 'w' for west.
More precisely the boundaries of a grid function f are

    South: f[:,0]
    East:  f[-1,:]
    North: f[:,-1]
    West:  f[0,:]

Grids (also referred to as blocks) are stored as pairs of matrices (X,Y), such
that (X[i,j], Y[i,j]) is the (i,j):th node in the grid. Multiblock grids can be
thought of as lists of such pairs. A list F of grid functions is called a
'multiblock function' and should be interpreted as function evaluations on a
sequence of grids constituting a multiblock grid.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from sbpy import operators


def collocate_corners(blocks, tol=1e-15):
    """ Collocate corners of blocks if they are equal up to some tolerance. """
    for ((X1,Y1),(X2,Y2)) in itertools.combinations(blocks,2):
        for (c1,c2) in itertools.product([(0,0),(-1,0),(-1,-1),(0,-1)], repeat=2):
            if np.abs(X1[c1]-X2[c2]) < tol and np.abs(Y1[c1]-Y2[c2]) < tol:
                X1[c1] = X2[c2]
                Y1[c1] = Y2[c2]




def get_boundary(X,Y,side):
    """ Returns the boundary of a block. """

    assert(side in {'w','e','s','n'})

    if side == 'w':
        return X[0,:], Y[0,:]
    elif side == 'e':
        return X[-1,:], Y[-1,:]
    elif side == 's':
        return X[:,0], Y[:,0]
    elif side == 'n':
        return X[:,-1], Y[:,-1]

def get_function_boundary(F,side):
    """ Returns the boundary of a grid function. """

    assert(side in {'w','e','s','n'})

    if side == 'w':
        return F[0,:]
    elif side == 'e':
        return F[-1,:]
    elif side == 's':
        return F[:,0]
    elif side == 'n':
        return F[:,-1]



def get_corners(X,Y):
    """ Returns the corners of a block.

    Starts with (X[0,0], Y[0,0]) and continues counter-clockwise.
    """
    return np.array([[X[0,0]  , Y[0,0]  ],
                     [X[-1,0] , Y[-1,0] ],
                     [X[-1,-1], Y[-1,-1]],
                     [X[0,-1] , Y[0,-1]]])


def get_center(X,Y):
    """ Returns the center point of a block. """
    corners = get_corners(X,Y)
    return 0.25*(corners[0] + corners[1] + corners[2] + corners[3])


def array_to_multiblock(grid, array):
    """ Converts a flat array to a multiblock function. """
    shapes = grid.get_shapes()
    F = [ np.zeros(shape) for shape in shapes ]

    counter = 0
    for (k,(Nx,Ny)) in enumerate(shapes):
        F[k] = np.reshape(array[counter:(counter+Nx*Ny)], (Nx, Ny))
        counter += Nx*Ny

    return F


def multiblock_to_array(grid, F):
    """ Converts a multiblock function to a flat array. """
    return np.array(F).flatten()


class Multiblock:
    """ Represents a structured multiblock grid.

    Attributes:
        blocks: A list of pairs of 2D numpy arrays containing x- and y-values for
            each block.

        block_interfaces: A list of dictionaries containing the interfaces for each
            block. For example, if interfaces[i] = {'n': (j, 'w')}, then
            the northern boundary of the block i coincides with the
            western boundary of the western boundary of block j.

        corners: A list of unique corners in the grid.

        edges: A list pairs of indices to the corners list, defining all the
            unique edges in grid connectivity graph.

        faces: A list of index-quadruples defining the blocks. For example, if
            faces[n] = [i,j,k,l], and (X,Y) are the matrices corresponding to
            block n, then (X[0,0],Y[0,0]) = corners[i]
                          (X[-1,0],Y[-1,0]) = corners[j]
                          (X[-1,-1],Y[-1,-1]) = corners[k]
                          (X[0,-1],Y[0,-1]) = corners[l]

        face_edges: A list of dicts specifying the edges of each face in the
            grid connectivity graph. For example, if
            face_edges[n] = {'s': 1, 'e': 5, 'n': 3, 'w': 0}, then the
            southern boundary of the n:th face is edge 1, and so on.

        interfaces: A list of pairs of the form ( (k1, s1), (k2, s2) ), where
            k1, k2 are the indices of the blocks connected to the interface, and
            s1, s2 are the sides of the respective blocks that make up the
            interface.

        non_interfaces: A list of lists specifying the non-interfaces of each
            block. For example, if non_interfaces[i] = ['w', 'n'],
            then the western and northern sides of block i are not
            interfaces.

        num_blocks: The total number of blocks in the grid.

        Nx: A list of the number of grid points in the x-direction of each block.

        Ny: A list of the number of grid points in the y-direction of each block.
    """

    def __init__(self, blocks):
        """ Initializes a Multiblock object.

        Args:
            blocks: A list of pairs of 2D numpy arrays containing x- and y-values
                   for each block.

            Note that the structure of these blocks should be such that for the
            k:th element (X,Y) in the blocks list, we have that (X[i,j],Y[i,j])
            is the (i,j):th node in the k:th block.
        """

        for (X,Y) in blocks:
            assert(X.shape == Y.shape)

        self.blocks = blocks
        self.num_blocks = len(blocks)

        self.shapes = []
        for (X,Y) in blocks:
            self.shapes.append((X.shape[0], X.shape[1]))

        # Save unique corners
        self.corners = []
        for X,Y in self.blocks:
            self.corners.append(get_corners(X,Y))

        self.corners = np.unique(np.concatenate(self.corners), axis=0)

        # Save faces in terms of unique corners
        self.faces = []

        for k,(X,Y) in enumerate(self.blocks):
            block_corners = get_corners(X,Y)
            indices = []
            for c in block_corners:
                idx = np.argwhere(np.all(c == self.corners, axis=1)).item()
                indices.append(idx)
            self.faces.append(np.array(indices))
        self.faces = np.array(self.faces)

        # Save unique edges
        self.edges = []
        for face in self.faces:
            for k in range(4):
                self.edges.append(np.array(sorted([face[k], face[(k+1)%4]])))

        self.edges = np.unique(self.edges, axis=0)

        # Save face edges
        self.face_edges = []
        for face in self.faces:
            self.face_edges.append({})
            for k,side in enumerate(['s','e','n','w']):
                edge = np.array(sorted([face[k], face[(k+1)%4]]))
                self.face_edges[-1][side] = \
                    np.argwhere(np.all(edge == self.edges, axis=1)).item()

        # Find interfaces
        self.block_interfaces = [{} for _ in range(self.num_blocks)]
        for ((i,edges1), (j,edges2)) in \
        itertools.combinations(enumerate(self.face_edges),2):
            for (side1,side2) in \
            itertools.product(['s', 'e', 'n', 'w'], repeat=2):
                if edges1[side1] == edges2[side2]:
                    self.block_interfaces[i][side1] = (j, side2)
                    self.block_interfaces[j][side2] = (i, side1)

        self.interfaces = []
        for k in range(len(self.edges)):
            blocks = []
            sides = []
            for n in range(self.num_blocks):
                for side in ['w','s','n','e']:
                    if self.face_edges[n][side] == k:
                        blocks.append(n)
                        sides.append(side)
            if len(blocks) == 2:
                self.interfaces.append(((blocks[0],sides[0]),(blocks[1],sides[1])))

        # Find non-interfaces
        self.non_interfaces = [[] for _ in range(self.num_blocks)]
        for (i,edges) in enumerate(self.face_edges):
            is_interface = False
            other_edges = \
                np.array([ np.fromiter(other_edges.values(), dtype=float) for
                    (j, other_edges) in enumerate(self.face_edges) if j != i])
            for side in ['s', 'e', 'n', 'w']:
                if edges[side] not in other_edges.flatten():
                    self.non_interfaces[i].append(side)


    def evaluate_function(self, f):
        """ Evaluates a (vectorized) function on the grid. """
        return [ f(X,Y) for (X,Y) in self.blocks ]


    def get_blocks(self):
        """ Returns a list of matrix pairs (X,Y) representing grid blocks. """
        return self.blocks


    def get_block(self, k):
        """ Returns a matrix pair (X,Y) representing the k:th block. """
        return self.blocks[k]


    def is_shape_consistent(self, F):
        """ Check if a multiblock function F is shape consistent with grid. """
        is_consistent = True
        for (k,f) in enumerate(F):
            if F[k].shape != self.shapes[k]:
                is_consistent = False
        return is_consistent


    def get_boundary_slice(self,k,side):
        """ Get a slice representing the boundary of block. The slice can
        be used to index the given boundary of a grid function on the given block.
        For example, if slice = get_boundary_slice(k,'w') and F is a grid function,
        then F[slice] will refer to the western boundary of F.

        Args:
            F: A grid function.
            side: The side at which the boundary is located ('s', 'e', 'n', or 'w')

        Returns:
            slice: A slice that can be used to index the given boundary in F.
        """
        assert(side in ['s', 'e', 'n', 'w'])
        (Nx, Ny) = self.shapes[k]
        slice_dict = {'s': (slice(Nx), 0),
                      'e': (-1, slice(Ny)),
                      'n': (slice(Nx), -1),
                      'w': (0, slice(Ny))}

        return slice_dict[side]


    def get_interfaces(self):
        """ Returns a list of pairs of the form ( (k1, s1), (k2, s2) ), where
            k1, k2 are the indices of the blocks connected to the interface, and
            s1, s2 are the sides of the respective blocks that make up the
            interface. """
        return self.interfaces()


    def get_block_interfaces(self):
        """ Returns a list of dictionaries containing the interfaces for each
        block. For example, if interfaces = get_block_interfaces(), and
        interfaces[i] = {'n': (j, 'w')}, then the northern boundary of the block
        i coincides with the western boundary of the western boundary of block j.
        """
        return self.block_interfaces


    def get_external_boundaries(self):
        """ Returns a list of lists containing the external boundaries
        for each block. For example, if bds = get_external_boundaries(), and
        bds[k] = ['n', 'e'], then the northern and eastern boundaries of the
        block k are external boundaries.
        """
        return self.non_interfaces


    def get_shapes(self):
        """ Returns a list of the shapes of the blocks in the grid. """
        return self.shapes


    def is_interface(self, block_idx, side):
        """ Check if a given side is an interface.

        Returns True if the given side of the given block is an interface. """

        if side in self.block_interfaces[block_idx]:
            return True
        else:
            return False


    def plot_grid(self):
        """ Plot the entire grid. """

        fig, ax = plt.subplots()
        for X,Y in self.blocks:
            ax.plot(X,Y,'b')
            ax.plot(np.transpose(X),np.transpose(Y),'b')
            for side in {'w', 'e', 's', 'n'}:
                x,y = get_boundary(X,Y,side)
                ax.plot(x,y,'k',linewidth=3)

        plt.show()


    def plot_domain(self):
        """ Fancy domain plot without gridlines. """

        fig, ax = plt.subplots()
        for k,(X,Y) in enumerate(self.blocks):
            xs,ys = get_boundary(X,Y,'s')
            xe,ye = get_boundary(X,Y,'e')
            xn,yn = get_boundary(X,Y,'n')
            xn = np.flip(xn)
            yn = np.flip(yn)
            xw,yw = get_boundary(X,Y,'w')
            xw = np.flip(xw)
            yw = np.flip(yw)
            x_poly = np.concatenate([xs,xe,xn,xw])
            y_poly = np.concatenate([ys,ye,yn,yw])

            ax.fill(x_poly,y_poly,'tab:gray')
            ax.plot(x_poly,y_poly,'k')
            c = get_center(X,Y)
            ax.text(c[0], c[1], "$\Omega_" + str(k) + "$")


        plt.show()


    def get_neighbor_boundary(self, F, block_idx, side):
        """ Returns an array of boundary data from a neighboring block.

        Arguments:
            F: A 2d array of function evaluations on the neighbor block.
            block_idx: The index of the block to send data to.
            side: The side of the block to send data to ('s', 'e', 'n', or 'w').
        """
        assert(self.is_interface(block_idx, side))

        neighbor_idx, neighbor_side = self.block_interfaces[block_idx][side]

        flip = False
        if (neighbor_side, side) in [('s','e'), ('s','s'),
                                     ('e','s'), ('e','e'),
                                     ('n','w'), ('n','n'),
                                     ('w','n'), ('w','w')]:
            flip = True

        if flip:
            return np.flip(get_function_boundary(F, neighbor_side))
        else:
            return get_function_boundary(F, neighbor_side)


class MultiblockSBP(Multiblock):
    """ A class combining Multiblock functionality and SBP2D functionality.

    Attributes:
        sbp_ops: A list of SBP2D object corresponding to each block.
    """
    def __init__(self, blocks, accuracy = 2):
        """ Initializes a MultiblockSBP object.
        Args:
            blocks: A list of matrix pairs representing the blocks.
        Optional:
            accuracy: The interior accuracy of the difference operators (2 or 4).
        """
        super().__init__(blocks)

        # Create SBP2D objects for each block.
        self.sbp_ops = []
        for (X,Y) in self.get_blocks():
            self.sbp_ops.append(operators.SBP2D(X,Y,accuracy))


    def diffx(self, U):
        """ Differentiates a Multiblock function with respect to x. """
        return [ self.sbp_ops[i].diffx(U[i]) for i in range(self.num_blocks) ]


    def diffy(self, U):
        """ Differentiates a Multiblock function with respect to y. """
        return [ self.sbp_ops[i].diffy(U[i]) for i in range(self.num_blocks) ]

    def integrate(self, U):
        """ Integrates a Multiblock function over the domain. """
        return sum([ self.sbp_ops[i].integrate(U[i]) for
                     i in range(self.num_blocks) ])

    def get_normals(self, block_idx, side):
        """ Get the normals of a specified side of a particular block. """
        return self.sbp_ops[block_idx].normals[side]


def load_p3d(filename):
    with open(filename) as data:
        num_blocks = int(data.readline())

        X = []
        Y = []
        Nx = []
        Ny = []
        for _ in range(num_blocks):
            size = np.fromstring(data.readline(), sep=' ', dtype=int)
            Nx.append(size[0])
            Ny.append(size[1])

        blocks = []
        for k in range(num_blocks):
            X_cur = []
            Y_cur = []
            for n in range(Nx[k]):
                X_cur.append(np.fromstring(data.readline(), sep=' '))
            for n in range(Nx[k]):
                Y_cur.append(np.fromstring(data.readline(), sep=' '))

            blocks.append((np.array(X_cur),np.array(Y_cur)))
            #X.append(np.array(X_cur))
            #Y.append(np.array(Y_cur))
            for _ in range(Nx[k]):
                next(data)


    return blocks



## mesh.py ##


from . h2lib import *
import numpy as np
from matplotlib import pyplot as plt


class Mesh:
    
    _surface = None

    def __init__(self, surf):
        self._surface = surf
        self._update_properties()
        
    @classmethod
    def from_macrosurface3d(cls, ms, refn=2):
         return cls(build_from_macrosurface3d_surface3d(ms, refn))

    @classmethod
    def from_geometry(cls, vertices, edges, triangles, refn=2):
        return cls.from_macrosurface3d(Macrosurface3d(vertices, edges, triangles), refn)

    def __add__(self, other):

        surf1 = self._surface
        surf2 = other._surface
        return Mesh(merge_surface3d(surf1, surf2))
    
    def __iadd__(self, other):

        surf1 = self._surface
        surf2 = other._surface
        self._surface = merge_surface3d(surf1, surf2)
        self._update_properties()
        return self
        
    @property
    def vertices(self):
        return np.asarray(self._surface.x)
    
    @property
    def edges(self):
        return np.asarray(self._surface.e)

    @property
    def triangles(self):
        return np.asarray(self._surface.t)

    @property
    def normals(self):
        return np.asarray(self._surface.n)

    @property
    def g(self):
        return self._surface.g

    @property
    def hmin(self):
        return self._surface.hmin
    
    @property
    def hmax(self):
        return self._surface.hmax

    def _update_properties(self):
        prepare_surface3d(self._surface)

    def _refine(self):
        self._surface = refine_red_surface3d(self._surface)
        self._update_properties()

    def refine(self, n=1):
        for i in range(n):
            self._refine()

    def draw(self):

        vertices = self.vertices
        edges = self.edges

        plt.figure()
        plt.plot(vertices[:,0], vertices[:,1], '.')

        for e in edges:
            x1, y1, z1 = vertices[e[0], :]
            x2, y2, z2 = vertices[e[1], :]
            plt.plot([x1, x2], [y1, y2], 'b-')

        plt.axis('equal')
        plt.show()


def square(xl, yl, refn=2):

    assert refn > 1

    ms = Macrosurface3d(5, 8, 4)
    ms.set_parametrization('square')

    #  vertices 
    ms.x[0, 0] = -xl / 2 #  bottom left 
    ms.x[0, 1] = -yl / 2
    ms.x[0, 2] = 0.0
    ms.x[1, 0] = xl / 2 #  bottom right 
    ms.x[1, 1] = -yl / 2
    ms.x[1, 2] = 0.0
    ms.x[2, 0] = xl / 2 #  top right 
    ms.x[2, 1] = yl / 2
    ms.x[2, 2] = 0.0
    ms.x[3, 0] = -xl / 2 #  top left 
    ms.x[3, 1] = yl / 2
    ms.x[3, 2] = 0.0
    ms.x[4, 0] = 0.0 # center
    ms.x[4, 1] = 0.0 
    ms.x[4, 2] = 0.0

    #  edges 
    ms.e[0, 0] = 0 # bottom left
    ms.e[0, 1] = 1
    ms.e[1, 0] = 1 # bototm right
    ms.e[1, 1] = 2
    ms.e[2, 0] = 2 # top right
    ms.e[2, 1] = 3
    ms.e[3, 0] = 3 # top left
    ms.e[3, 1] = 0
    ms.e[4, 0] = 0 # left horizontal
    ms.e[4, 1] = 4
    ms.e[5, 0] = 1 # bottom vertical
    ms.e[5, 1] = 4
    ms.e[6, 0] = 2 # right horizontal
    ms.e[6, 1] = 4
    ms.e[7, 0] = 3 # right vertical
    ms.e[7, 1] = 4

    #  triangles and triangle edges 
    ms.t[0, 0] = 0 # bottom left
    ms.t[0, 1] = 1
    ms.t[0, 2] = 4
    ms.s[0, 0] = 5
    ms.s[0, 1] = 4
    ms.s[0, 2] = 0
    ms.t[1, 0] = 1 # bottom right
    ms.t[1, 1] = 2
    ms.t[1, 2] = 4
    ms.s[1, 0] = 6
    ms.s[1, 1] = 5
    ms.s[1, 2] = 1
    ms.t[2, 0] = 2 # top right
    ms.t[2, 1] = 3
    ms.t[2, 2] = 4
    ms.s[2, 0] = 7
    ms.s[2, 1] = 6
    ms.s[2, 2] = 2
    ms.t[3, 0] = 3 # top left
    ms.t[3, 1] = 0
    ms.t[3, 2] = 4
    ms.s[3, 0] = 4
    ms.s[3, 1] = 7
    ms.s[3, 2] = 3

    surf = build_from_macrosurface3d_surface3d(ms, refn)
    return Mesh(surf)


def square2(xl, yl, refn=2):

    assert refn > 1

    ms = Macrosurface3d(4, 5, 2)
    ms.set_parametrization('square')

    #  vertices 
    ms.x[0, 0] = -xl / 2 #  bottom left 
    ms.x[0, 1] = -yl / 2
    ms.x[0, 2] = 0.0
    ms.x[1, 0] = xl / 2 #  bottom right 
    ms.x[1, 1] = -yl / 2
    ms.x[1, 2] = 0.0
    ms.x[2, 0] = xl / 2 #  top right 
    ms.x[2, 1] = yl / 2
    ms.x[2, 2] = 0.0
    ms.x[3, 0] = -xl / 2 #  top left 
    ms.x[3, 1] = yl / 2
    ms.x[3, 2] = 0.0

    #  edges 
    ms.e[0, 0] = 0 #  bottom 
    ms.e[0, 1] = 1
    ms.e[1, 0] = 1 #  right
    ms.e[1, 1] = 2
    ms.e[2, 0] = 2 #  top 
    ms.e[2, 1] = 3
    ms.e[3, 0] = 3 #  left
    ms.e[3, 1] = 0
    ms.e[4, 0] = 1 #  diagonal
    ms.e[4, 1] = 3

    #  triangles and triangle edges 
    ms.t[0, 0] = 0 # bottom left
    ms.t[0, 1] = 1
    ms.t[0, 2] = 3
    ms.s[0, 0] = 4
    ms.s[0, 1] = 3
    ms.s[0, 2] = 0
    ms.t[1, 0] = 1 # top right
    ms.t[1, 1] = 2
    ms.t[1, 2] = 3
    ms.s[1, 0] = 2
    ms.s[1, 1] = 4
    ms.s[1, 2] = 1

    surf = build_from_macrosurface3d_surface3d(ms, refn)
    return Mesh(surf)


def circle(rl, refn=2):

    assert refn > 1

    ms = Macrosurface3d(5, 8, 4)
    ms.set_parametrization('circle')

    #  vertices 
    ms.x[0, 0] = -rl # left 
    ms.x[0, 1] = 0.0 
    ms.x[0, 2] = 0.0
    ms.x[1, 0] = 0.0 # bottom 
    ms.x[1, 1] = -rl 
    ms.x[1, 2] = 0.0
    ms.x[2, 0] = rl  # right 
    ms.x[2, 1] = 0.0 
    ms.x[2, 2] = 0.0
    ms.x[3, 0] = 0.0 # top 
    ms.x[3, 1] = rl
    ms.x[3, 2] = 0.0
    ms.x[4, 1] = 0.0 # center
    ms.x[4, 2] = 0.0

    #  edges 
    ms.e[0, 0] = 0 # bottom left
    ms.e[0, 1] = 1
    ms.e[1, 0] = 1 # bototm right
    ms.e[1, 1] = 2
    ms.e[2, 0] = 2 # top right
    ms.e[2, 1] = 3
    ms.e[3, 0] = 3 # top left
    ms.e[3, 1] = 0
    ms.e[4, 0] = 0 # left horizontal
    ms.e[4, 1] = 4
    ms.e[5, 0] = 1 # bottom vertical
    ms.e[5, 1] = 4
    ms.e[6, 0] = 2 # right horizontal
    ms.e[6, 1] = 4
    ms.e[7, 0] = 3 # right vertical
    ms.e[7, 1] = 4

    #  triangles and triangle edges 
    ms.t[0, 0] = 0 # bottom left
    ms.t[0, 1] = 1
    ms.t[0, 2] = 4
    ms.s[0, 0] = 5
    ms.s[0, 1] = 4
    ms.s[0, 2] = 0
    ms.t[1, 0] = 1 # bottom right
    ms.t[1, 1] = 2
    ms.t[1, 2] = 4
    ms.s[1, 0] = 6
    ms.s[1, 1] = 5
    ms.s[1, 2] = 1
    ms.t[2, 0] = 2 # top right
    ms.t[2, 1] = 3
    ms.t[2, 2] = 4
    ms.s[2, 0] = 7
    ms.s[2, 1] = 6
    ms.s[2, 2] = 2
    ms.t[3, 0] = 3 # top left
    ms.t[3, 1] = 0
    ms.t[3, 2] = 4
    ms.s[3, 0] = 4
    ms.s[3, 1] = 7
    ms.s[3, 2] = 3

    surf = build_from_macrosurface3d_surface3d(ms, refn)
    return Mesh(surf)


def square_geometry(xl, yl, center=(0,0,0)):

    ms = Macrosurface3d(5, 8, 4)
    ms.set_parametrization('square')

    # vertices 
    v = np.array((5, 3), dtype=np.float64)
    v[0, :] = -xl / 2, -yl / 2, 0.0  # bottom left 
    v[1, :] = xl / 2, -yl / 2, 0.0  # bottom right
    v[2, :] = xl / 2, yl / 2, 0.0  # top right 
    v[3, :] = -xl / 2, yl / 2, 0.0  # top left
    v[4, :] = 0.0, 0.0, 0.0  # center
    v += center  # shift

    #  edges 
    e = np.array((8,2), dtype=np.uint32)
    e[0, :] = 0, 1  # bottom left
    e[1, :] = 1, 2  # bototm right
    e[2, :] = 2, 3  # top right
    e[3, :] = 3, 0  # top left
    e[4, :] = 0, 4  # left horizontal
    e[5, :] = 1, 4  # bottom vertical
    e[6, :] = 2, 4  # right horizontal
    e[7, :] = 3, 4  # right vertical

    #  triangles and triangle edges 
    t = np.array((4, 3), dtype=np.uint32)
    s = np.array((4, 3), dtype=np.uint32)
    t[0, :] = 0, 1, 4  # bottom left
    s[0, :] = 5, 4, 0
    t[1, :] = 1, 2, 4  # bottom right
    s[1, :] = 6, 5, 1
    t[2, :] = 2, 3, 4  # top right
    s[2, :] = 7, 6, 2
    t[3, :] = 3, 0, 4  # top left
    s[3, :] = 4, 7, 3


def circle_geometry(rl, center=(0,0,0)):
    pass
    

def matrix_array(nx, ny, pitchx, pitchy, **kwargs):


    pass


def linear_array():
    pass


def from_spec():
    pass



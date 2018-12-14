''' Triangular surface mesh generation.
'''
import numpy as np
from matplotlib import pyplot as plt

from . h2lib import *


eps = np.finfo(np.float64).eps

class Mesh:
    
    _surface = None

    def __init__(self):
        self._surface = None
    
    @classmethod
    def from_surface3d(cls, surf):
        
        obj = cls()
        obj._surface = surf
        obj._update_properties()
        return obj

    @classmethod
    def from_macrosurface3d(cls, ms, center=(0,0,0), refn=2):

        # mesh must be refined at least once, otherwise h2lib throws exception
        assert refn > 1

        obj = cls.from_surface3d(build_from_macrosurface3d_surface3d(ms, refn))
        obj.translate(center)
        return obj

    @classmethod
    def from_geometry(cls, vertices, edges, triangles, triangle_edges, center=(0,0,0),
        refn=2, parametrization='square'):
        
        ms = Macrosurface3d(len(vertices), len(edges), len(triangles))
        ms.x[:] = vertices
        ms.e[:] = edges
        ms.t[:] = triangles
        ms.s[:] = triangle_edges
        ms.set_parametrization(parametrization)

        return cls.from_macrosurface3d(ms, center=center, refn=refn)

    @classmethod
    def from_abstract(cls, array, refn=2, **kwargs):
        return _from_abstract(cls, array, refn, **kwargs)

    def __add__(self, other):

        surf1 = self._surface
        surf2 = other._surface

        if surf1 is None and surf2 is None:
            return Mesh()
        elif surf1 is None:
            return Mesh.from_surface3d(surf2)
        elif surf2 is None:
            return Mesh.from_surface3d(surf1)
        else:
            return Mesh.from_surface3d(merge_surface3d(surf1, surf2))
    
    def __iadd__(self, other):

        surf1 = self._surface
        surf2 = other._surface

        if surf1 is None and surf2 is None:
            pass
        elif surf1 is None:
            self._surface = surf2
        elif surf2 is None:
            pass
        else:
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
    def triangle_edges(self):
        return np.asarray(self._surface.s)

    @property
    def normals(self):
        return np.asarray(self._surface.n)

    @property
    def g(self):
        return np.asarray(self._surface.g)

    @property
    def hmin(self):
        return self._surface.hmin
    
    @property
    def hmax(self):
        return self._surface.hmax

    @property
    def surface3d(self):
        return self._surface

    def _update_properties(self):
        prepare_surface3d(self._surface)

    def _refine(self):
        self._surface = refine_red_surface3d(self._surface)
        self._update_properties()

    def refine(self, n=1):
        for i in range(n):
            self._refine()

    def translate(self, r):
        translate_surface3d(self._surface, np.array(r, dtype=np.float64))

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


def square(xl, yl, center=(0,0,0), refn=2):
    mesh = Mesh.from_geometry(*square_geometry(xl, yl), center=center,
        refn=refn, parametrization='square')
    # check and flag boundary vertices
    mask1 = np.abs(mesh.vertices[:,0] + xl / 2) <= eps
    mask2 = np.abs(mesh.vertices[:,0] - xl / 2) <= eps
    mask3 = np.abs(mesh.vertices[:,1] + yl / 2) <= eps
    mask4 = np.abs(mesh.vertices[:,1] - yl / 2) <= eps
    mesh.on_boundary = np.any(np.c_[mask1, mask2, mask3, mask4], axis=1)
    return mesh


def square2(xl, yl, center=(0,0,0), refn=2):
    mesh = Mesh.from_geometry(*square_geometry2(xl, yl), center=center,
        refn=refn, parametrization='square')
    # check and flag boundary vertices
    mask1 = np.abs(mesh.vertices[:,0] + xl / 2) <= eps
    mask2 = np.abs(mesh.vertices[:,0] - xl / 2) <= eps
    mask3 = np.abs(mesh.vertices[:,1] + yl / 2) <= eps
    mask4 = np.abs(mesh.vertices[:,1] - yl / 2) <= eps
    mesh.on_boundary = np.any(np.c_[mask1, mask2, mask3, mask4], axis=1)
    return mesh


def circle(rl, center=(0,0,0), refn=2):
    return Mesh.from_geometry(*circle_geometry(rl), center=center,
        refn=refn, parametrization='circle')
    

def square_geometry(xl, yl):

    # vertices 
    v = np.zeros((5, 3), dtype=np.float64)
    v[0, :] = -xl / 2, -yl / 2, 0.0  # bottom left 
    v[1, :] = xl / 2, -yl / 2, 0.0  # bottom right
    v[2, :] = xl / 2, yl / 2, 0.0  # top right 
    v[3, :] = -xl / 2, yl / 2, 0.0  # top left
    v[4, :] = 0.0, 0.0, 0.0  # center

    #  edges 
    e = np.zeros((8,2), dtype=np.uint32)
    e[0, :] = 0, 1  # bottom
    e[1, :] = 1, 2  # right
    e[2, :] = 2, 3  # top
    e[3, :] = 3, 0  # left
    e[4, :] = 0, 4  # bottom left
    e[5, :] = 1, 4  # bottom right
    e[6, :] = 2, 4  # top right
    e[7, :] = 3, 4  # top left

    #  triangles and triangle edges 
    t = np.zeros((4, 3), dtype=np.uint32)
    s = np.zeros((4, 3), dtype=np.uint32)
    t[0, :] = 0, 1, 4  # bottom
    s[0, :] = 5, 4, 0
    t[1, :] = 1, 2, 4  # right
    s[1, :] = 6, 5, 1
    t[2, :] = 2, 3, 4  # top
    s[2, :] = 7, 6, 2
    t[3, :] = 3, 0, 4  # left
    s[3, :] = 4, 7, 3

    return v, e, t, s


def square_geometry2(xl, yl):

    # vertices 
    v = np.zeros((4, 3), dtype=np.float64)
    v[0, :] = -xl / 2, -yl / 2, 0.0  # bottom left 
    v[1, :] = xl / 2, -yl / 2, 0.0  # bottom right
    v[2, :] = xl / 2, yl / 2, 0.0  # top right 
    v[3, :] = -xl / 2, yl / 2, 0.0  # top left

    #  edges 
    e = np.zeros((5, 2), dtype=np.uint32)
    e[0, :] = 0, 1  # bottom
    e[1, :] = 1, 2  # right
    e[2, :] = 2, 3  # top
    e[3, :] = 3, 0  # left
    e[4, :] = 1, 3  # diagonal

    #  triangles and triangle edges 
    t = np.zeros((2, 3), dtype=np.uint32)
    s = np.zeros((2, 3), dtype=np.uint32)
    t[0, :] = 0, 1, 3  # bottom left
    s[0, :] = 4, 3, 0
    t[1, :] = 1, 2, 3  # top right
    s[1, :] = 2, 4, 1

    return v, e, t, s


def circle_geometry(rl):

    #  vertices 
    v = np.zeros((5, 3), dtype=np.float64)
    v[0, :] = -rl, 0.0, 0.0 # left 
    v[1, :] = 0.0, -rl, 0.0 # bottom 
    v[2, :] = rl, 0.0, 0.0  # right 
    v[3, :] = 0.0, rl, 0.0 # top 
    v[4, :] = 0.0, 0.0, 0.0 # center

    #  edges 
    e = np.zeros((8,2), dtype=np.uint32)
    e[0, :] = 0, 1  # bottom left
    e[1, :] = 1, 2  # bototm right
    e[2, :] = 2, 3  # top right
    e[3, :] = 3, 0  # top left
    e[4, :] = 0, 4  # left horizontal
    e[5, :] = 1, 4  # bottom vertical
    e[6, :] = 2, 4  # right horizontal
    e[7, :] = 3, 4  # right vertical

    #  triangles and triangle edges 
    t = np.zeros((4, 3), dtype=np.uint32)
    s = np.zeros((4, 3), dtype=np.uint32)
    t[0, :] = 0, 1, 4  # bottom left
    s[0, :] = 5, 4, 0
    t[1, :] = 1, 2, 4  # bottom right
    s[1, :] = 6, 5, 1
    t[2, :] = 2, 3, 4  # top right
    s[2, :] = 7, 6, 2
    t[3, :] = 3, 0, 4  # top left
    s[3, :] = 4, 7, 3

    return v, e, t, s


def matrix_array(nx, ny, pitchx, pitchy, shape='square', refn=2, **kwargs):

    lengthx, lengthy = pitchx * (nx - 1), pitchy * (ny - 1)
    xv = np.linspace(-lengthx / 2, lengthx / 2, nx)
    yv = np.linspace(-lengthy / 2, lengthy / 2, ny)
    zv = 0
    centers = np.stack(np.meshgrid(xv, yv, zv), axis=-1).reshape((-1, 3))
    
    if shape.lower() in ['circle']:

        f = circle
        rl = kwargs['radius']
        args = rl,

    elif shape.lower() in ['square']:

        f = square
        xl, yl = kwargs['lengthx'], kwargs['lengthy']
        args = xl, yl
    
    else:
        raise TypeError

    mesh = Mesh()
    for c in centers:
        mesh += f(*args, center=c, refn=refn)
    
    return mesh


def fast_matrix_array(nx, ny, pitchx, pitchy, xl, yl, refn=2, **kwargs):

    lengthx, lengthy = pitchx * (nx - 1), pitchy * (ny - 1)
    xv = np.linspace(-lengthx / 2, lengthx / 2, nx)
    yv = np.linspace(-lengthy / 2, lengthy / 2, ny)
    zv = 0
    centers = np.stack(np.meshgrid(xv, yv, zv), axis=-1).reshape((-1, 3))
    
    verts, edges, tris, tri_edges = [], [], [], []
    vidx = 0
    eidx = 0

    for c in centers:

        v, e, t, s = square_geometry(xl, yl)

        v += c
        e += vidx
        t += vidx
        s += eidx

        vidx += len(v)
        eidx += len(e)

        verts.append(v)
        edges.append(e)
        tris.append(t)
        tri_edges.append(s)

    verts = np.concatenate(verts, axis=0)
    edges = np.concatenate(edges, axis=0)
    tris = np.concatenate(tris, axis=0)
    tri_edges = np.concatenate(tri_edges, axis=0)

    mesh = Mesh.from_geometry(verts, edges, tris, tri_edges, refn=refn)
    mesh.on_boundary = np.zeros(len(mesh.vertices), dtype=np.bool)

    # check and flag boundary vertices
    x, y, z = mesh.vertices.T
    for cx, cy, cz in centers:
        xmin = cx - xl / 2  - 2 * eps
        xmax = cx + xl / 2 + 2 * eps
        ymin = cy - yl / 2 - 2 * eps
        ymax = cy + yl / 2 + 2 * eps
        mask_x = np.logical_and(x >= xmin, x <= xmax)
        mask_y = np.logical_and(y >= ymin, y <= ymax)
        mem_mask = np.logical_and(mask_x, mask_y)

        mask1 = np.abs(x[mem_mask] - xmin) <= 2 * eps
        mask2 = np.abs(x[mem_mask] - xmax) <= 2 * eps
        mask3 = np.abs(y[mem_mask] - ymin) <= 2 * eps
        mask4 = np.abs(y[mem_mask] - ymax) <= 2 * eps
        mesh.on_boundary[mem_mask] = np.any(np.c_[mask1, mask2, mask3, mask4], axis=1)

    return mesh


def linear_array():
    pass


def _from_abstract(cls, array, refn=2, **kwargs):
    '''
    Generate mesh from abstract representation of an array.
    '''
    # generate geometry in terms of vertices, edges, and triangles
    verts, edges, tris, tri_edges = [], [], [], []
    vidx = 0
    eidx = 0
    
    for elem in array.elements:
        for mem in elem.membranes:

            v, e, t, s = square_geometry(mem.length_x, mem.length_y)

            v += np.array(mem.position)
            e += vidx
            t += vidx
            s += eidx

            vidx += len(v)
            eidx += len(e)

            verts.append(v)
            edges.append(e)
            tris.append(t)
            tri_edges.append(s)

    verts = np.concatenate(verts, axis=0)
    edges = np.concatenate(edges, axis=0)
    tris = np.concatenate(tris, axis=0)
    tri_edges = np.concatenate(tri_edges, axis=0)

    # construct mesh from geometry
    mesh = cls.from_geometry(verts, edges, tris, tri_edges, refn=refn)
    
    # assign mesh vertices to patches, membranes, and elements
    nverts = len(mesh.vertices)
    patch_counter = np.zeros(nverts, dtype=np.int32) # keeps track of current patch idx for each vertex
    patch_ids = np.ones((nverts, 4), dtype=np.int32) * np.nan
    membrane_ids = np.ones(nverts, dtype=np.int32) * np.nan
    element_ids = np.ones(nverts, dtype=np.int32) * np.nan
    mesh.on_boundary = np.zeros(nverts, dtype=np.bool)
    x, y, z = mesh.vertices.T

    for elem in array.elements:
        for mem in elem.membranes:
            for pat in mem.patches:
                # determine vertices which belong to each patch, using
                # eps for buffer to account for round-off error
                pat_x, pat_y, pat_z = pat.position
                length_x, length_y = pat.length_x, pat.length_y
                xmin = pat_x - length_x / 2  - 2 * eps
                xmax = pat_x + length_x / 2 + 2 * eps
                ymin = pat_y - length_y / 2 - 2 * eps
                ymax = pat_y + length_y / 2 + 2 * eps
                mask_x = np.logical_and(x >= xmin, x <= xmax)
                mask_y = np.logical_and(y >= ymin, y <= ymax)
                mask = np.logical_and(mask_x, mask_y)
            
                patch_ids[mask, patch_counter[mask]] = pat.id
                patch_counter[mask] += 1 # increment patch idx
                membrane_ids[mask] = mem.id
                element_ids[mask] = elem.id

            # determine vertices which belong to each membrane
            mem_x, mem_y, mem_z = mem.position
            length_x, length_y = mem.length_x, mem.length_y
            xmin = mem_x - length_x / 2  - 2 * eps
            xmax = mem_x + length_x / 2 + 2 * eps
            ymin = mem_y - length_y / 2 - 2 * eps
            ymax = mem_y + length_y / 2 + 2 * eps
            mask_x = np.logical_and(x >= xmin, x <= xmax)
            mask_y = np.logical_and(y >= ymin, y <= ymax)
            mem_mask = np.logical_and(mask_x, mask_y)

            # check and flag boundary vertices
            mask1 = np.abs(x[mem_mask] - xmin) <= 2 * eps
            mask2 = np.abs(x[mem_mask] - xmax) <= 2 * eps
            mask3 = np.abs(y[mem_mask] - ymin) <= 2 * eps
            mask4 = np.abs(y[mem_mask] - ymax) <= 2 * eps
            mesh.on_boundary[mem_mask] = np.any(np.c_[mask1, mask2, mask3, mask4], axis=1)

    # check that no vertices were missed
    assert ~np.any(np.isnan(patch_ids[:,0])) # check that each vertex is assigned to at least one patch
    assert ~np.any(np.isnan(membrane_ids))
    assert ~np.any(np.isnan(element_ids))

    mesh.patch_ids = patch_ids
    mesh.membrane_ids = membrane_ids
    mesh.element_ids = element_ids

    return mesh


def calc_refn_square(lx, ly, wavelen, step_per_wavelen=10, maxrefn=20):

    refn = 2

    while True:
        if refn > maxrefn:
            raise Exception('Mesh refinement limit reached')

        hmax = square(lx, ly, refn=refn).hmax
        if wavelen / hmax > step_per_wavelen:
            break
        else:
            refn += 1

    return refn
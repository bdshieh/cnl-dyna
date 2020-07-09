'''Routines for generation of triangular surface meshes.'''
import numpy as np
from cnld import abstract, util
from cnld.h2lib import *
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf

eps = np.finfo(np.float64).eps


class Mesh:
    '''
    2D triangular mesh class using H2Lib datastructures.

    Returns
    -------
    [type]
        [description]
    '''
    _surface = None

    def __init__(self):
        '''
        [summary]
        '''
        self._surface = None

    @classmethod
    def from_surface3d(cls, surf):
        '''
        [summary]

        Parameters
        ----------
        surf : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        obj = cls()
        obj._surface = surf
        obj._update_properties()
        return obj

    @classmethod
    def from_macrosurface3d(cls, ms, center=(0, 0, 0), refn=2):
        '''
        [summary]

        Parameters
        ----------
        ms : [type]
            [description]
        center : tuple, optional
            [description], by default (0, 0, 0)
        refn : int, optional
            [description], by default 2

        Returns
        -------
        [type]
            [description]
        '''
        # mesh must be refined at least once, otherwise h2lib throws exception
        assert refn > 1

        obj = cls.from_surface3d(build_from_macrosurface3d_surface3d(ms, refn))
        obj.translate(center)
        return obj

    @classmethod
    def from_geometry(cls, vertices, edges, triangles, triangle_edges):
        '''
        [summary]

        Parameters
        ----------
        vertices : [type]
            [description]
        edges : [type]
            [description]
        triangles : [type]
            [description]
        triangle_edges : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        surf = Surface3d(len(vertices), len(edges), len(triangles))
        surf.x[:] = vertices
        surf.e[:] = edges
        surf.t[:] = triangles
        surf.s[:] = triangle_edges

        return cls.from_surface3d(surf)

    @classmethod
    def from_abstract(cls, array, refn=1, **kwargs):
        '''
        [summary]

        Parameters
        ----------
        array : [type]
            [description]
        refn : int, optional
            [description], by default 1

        Returns
        -------
        [type]
            [description]
        '''
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
    def triangle_areas(self):
        return np.asarray(self._surface.g) / 2

    @property
    def hmin(self):
        return self._surface.hmin

    @property
    def hmax(self):
        return self._surface.hmax

    @property
    def nvertices(self):
        return len(self.vertices)

    @property
    def nedges(self):
        return len(self.edges)

    @property
    def ntriangles(self):
        return len(self.triangles)

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
        plt.plot(vertices[:, 0], vertices[:, 1], '.')

        for e in edges:
            x1, y1, z1 = vertices[e[0], :]
            x2, y2, z2 = vertices[e[1], :]
            plt.plot([x1, x2], [y1, y2], 'b-')

        plt.axis('equal')
        plt.show()

    def _memoize():
        return (self.nvertices, self.triangles.tostring(),
                self.edges.tostring(), self.triangle_edges.tostring())


def _from_abstract(cls, array, refn=1, **kwargs):
    '''
    Generate mesh from abstract representation of an array.

    Parameters
    ----------
    array : [type]
        [description]
    refn : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    TypeError
        [description]
    TypeError
        [description]
    '''
    # generate geometry in terms of vertices, edges, and triangles with refinement
    # (much faster to construct mesh from entire geometry once instead of membrane by
    # membrane)
    verts, edges, tris, tri_edges = [], [], [], []
    vidx = 0
    eidx = 0

    for elem in array.elements:
        for mem in elem.membranes:

            if isinstance(mem, abstract.SquareCmutMembrane):
                v, e, t, s = geometry_square(mem.length_x,
                                             mem.length_y,
                                             refn=refn)
            elif isinstance(mem, abstract.CircularCmutMembrane):
                v, e, t, s = geometry_circle(mem.radius, n=4, refn=refn)
            else:
                raise TypeError

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
    mesh = cls.from_geometry(verts, edges, tris, tri_edges)

    # assign mesh vertices to patches, membranes, and elements
    nverts = len(mesh.vertices)
    # patch_counter = np.zeros(nverts, dtype=np.int32) # keeps track of current patch idx for each vertex
    # patch_ids = np.ones((nverts, 4), dtype=np.int32) * np.nan
    membrane_ids = np.ones(nverts, dtype=np.int32) * np.nan
    element_ids = np.ones(nverts, dtype=np.int32) * np.nan
    mesh.on_boundary = np.zeros(nverts, dtype=np.bool)
    x, y, z = mesh.vertices.T

    for elem in array.elements:
        for mem in elem.membranes:
            # for pat in mem.patches:
            # determine vertices which belong to each patch, using
            # eps for buffer to account for round-off error
            # pat_x, pat_y, pat_z = pat.position
            # length_x, length_y = pat.length_x, pat.length_y
            # xmin = pat_x - length_x / 2  - 2 * eps
            # xmax = pat_x + length_x / 2 + 2 * eps
            # ymin = pat_y - length_y / 2 - 2 * eps
            # ymax = pat_y + length_y / 2 + 2 * eps
            # mask_x = np.logical_and(x >= xmin, x <= xmax)
            # mask_y = np.logical_and(y >= ymin, y <= ymax)
            # mask = np.logical_and(mask_x, mask_y)

            # patch_ids[mask, patch_counter[mask]] = pat.id
            # patch_counter[mask] += 1 # increment patch idx
            # membrane_ids[mask] = mem.id
            # element_ids[mask] = elem.id

            if isinstance(mem, abstract.SquareCmutMembrane):
                # determine vertices which belong to each membrane
                mem_x, mem_y, mem_z = mem.position
                length_x, length_y = mem.length_x, mem.length_y
                xmin = mem_x - length_x / 2  # - 2 * eps
                xmax = mem_x + length_x / 2  # + 2 * eps
                ymin = mem_y - length_y / 2  # - 2 * eps
                ymax = mem_y + length_y / 2  # + 2 * eps
                mask_x = np.logical_and(x >= xmin - 2 * eps,
                                        x <= xmax + 2 * eps)
                mask_y = np.logical_and(y >= ymin - 2 * eps,
                                        y <= ymax + 2 * eps)
                mem_mask = np.logical_and(mask_x, mask_y)
                membrane_ids[mem_mask] = mem.id
                element_ids[mem_mask] = elem.id

                # check and flag boundary vertices
                mask1 = np.abs(x[mem_mask] - xmin) <= 2 * eps
                mask2 = np.abs(x[mem_mask] - xmax) <= 2 * eps
                mask3 = np.abs(y[mem_mask] - ymin) <= 2 * eps
                mask4 = np.abs(y[mem_mask] - ymax) <= 2 * eps
                mesh.on_boundary[mem_mask] = np.any(np.c_[mask1, mask2, mask3,
                                                          mask4],
                                                    axis=1)

            elif isinstance(mem, abstract.CircularCmutMembrane):
                # determine vertices which belong to each membrane
                mem_x, mem_y, mem_z = mem.position
                radius = mem.radius
                rmax = radius + 2 * eps
                r = np.sqrt((x - mem_x)**2 + (y - mem_y)**2)
                mem_mask = r <= rmax
                membrane_ids[mem_mask] = mem.id
                element_ids[mem_mask] = elem.id

                # check and flag boundary vertices
                mask1 = r[mem_mask] <= radius + 2 * eps
                mask2 = r[mem_mask] >= radius - 2 * eps
                mesh.on_boundary[mem_mask] = np.logical_and(mask1, mask2)

            else:
                raise TypeError

    # check that no vertices were missed
    # assert ~np.any(np.isnan(patch_ids[:,0])) # check that each vertex is assigned to at least one patch
    # assert ~np.any(np.isnan(membrane_ids))
    # assert ~np.any(np.isnan(element_ids))

    # mesh.patch_ids = patch_ids
    mesh.membrane_ids = membrane_ids
    mesh.element_ids = element_ids

    return mesh


@util.memoize
def geometry_square(xl, yl, refn=1, type=1):
    '''
    Creates a square mesh geometry (vertices, triangles etc.) which can be used to
    construct a mesh object.

    Parameters
    ----------
    xl : [type]
        [description]
    yl : [type]
        [description]
    refn : int, optional
        [description], by default 1
    type : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    '''
    if type == 1:
        # vertices
        v = np.zeros((5, 3), dtype=np.float64)
        v[0, :] = -xl / 2, -yl / 2, 0.0  # bottom left
        v[1, :] = xl / 2, -yl / 2, 0.0  # bottom right
        v[2, :] = xl / 2, yl / 2, 0.0  # top right
        v[3, :] = -xl / 2, yl / 2, 0.0  # top left
        v[4, :] = 0.0, 0.0, 0.0  # center
        #  edges
        e = np.zeros((8, 2), dtype=np.uint32)
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

    elif type == 2:
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

    else:
        raise ValueError('incorrect type')

    # refine geometry using h2lib macrosurface3d -> surface3d procedure
    if refn > 1:
        msurf = Macrosurface3d(len(v), len(e), len(t))
        msurf.x[:] = v
        msurf.e[:] = e
        msurf.t[:] = t
        msurf.s[:] = s
        msurf.set_parametrization('square')
        surf = build_from_macrosurface3d_surface3d(msurf, refn)

        # copy arrays from surf
        v = np.array(surf.x, copy=True)
        e = np.array(surf.e, copy=True)
        t = np.array(surf.t, copy=True)
        s = np.array(surf.s, copy=True)

    # translate geometry
    # v += np.array(center)
    return v, e, t, s


@util.memoize
def geometry_circle(rl, n=4, refn=1):
    '''
    Creates a circle mesh geometry (vertices, triangles etc.) which can be used to
    construct a mesh object.

    Parameters
    ----------
    rl : [type]
        [description]
    n : int, optional
        [description], by default 4
    refn : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    ''' '''

    '''
    #  vertices
    v = np.zeros((n + 1, 3), dtype=np.float64)

    for i in range(n):
        theta = 2 * np.pi / n * i - np.pi
        # p = rl / (np.abs(np.sin(theta)) + np.abs(np.cos(theta)))
        x = rl * np.cos(theta)
        y = rl * np.sin(theta)

        v[i, :] = x, y, 0.0

    v[n, :] = 0.0, 0.0, 0.0
    v[np.isclose(v, 0)] = 0.0

    #  edges
    e = np.zeros((2 * n, 2), dtype=np.uint32)

    for i in range(n):
        e[i, :] = i, np.mod(i + 1, n)

    for i in range(n):
        e[n + i, :] = i, n

    #  triangles and triangle edges
    t = np.zeros((n, 3), dtype=np.uint32)
    s = np.zeros((n, 3), dtype=np.uint32)

    first = list(np.mod(np.arange(0, n) + 1, n) + n)
    second = list(np.mod(np.arange(0, n), n) + n)
    third = list(np.arange(0, n))

    for i in range(n):
        t[i, :] = i, np.mod(i + 1, n), n
        s[i, :] = first[i], second[i], third[i]

    # refine geometry using h2lib macrosurface3d -> surface3d procedure
    if refn > 1:
        msurf = Macrosurface3d(len(v), len(e), len(t))
        msurf.x[:] = v
        msurf.e[:] = e
        msurf.t[:] = t
        msurf.s[:] = s
        msurf.set_parametrization('circle')
        surf = build_from_macrosurface3d_surface3d(msurf, refn)

        # copy arrays from surf
        v = np.array(surf.x, copy=True)
        e = np.array(surf.e, copy=True)
        t = np.array(surf.t, copy=True)
        s = np.array(surf.s, copy=True)

    # translate geometry
    # v += np.array(center)
    return v, e, t, s


def square(xl, yl, refn=1, type=1, center=(0, 0, 0)):
    '''
    [summary]

    Parameters
    ----------
    xl : [type]
        [description]
    yl : [type]
        [description]
    refn : int, optional
        [description], by default 1
    type : int, optional
        [description], by default 1
    center : tuple, optional
        [description], by default (0, 0, 0)

    Returns
    -------
    [type]
        [description]
    '''
    v, e, t, s = geometry_square(xl, yl, refn=refn, type=type)
    v += np.array(center)
    mesh = Mesh.from_geometry(v, e, t, s)

    # check and flag boundary vertices
    mask1 = np.abs(mesh.vertices[:, 0] - center[0] + xl / 2) <= 2 * eps
    mask2 = np.abs(mesh.vertices[:, 0] - center[0] - xl / 2) <= 2 * eps
    mask3 = np.abs(mesh.vertices[:, 1] - center[1] + yl / 2) <= 2 * eps
    mask4 = np.abs(mesh.vertices[:, 1] - center[1] - yl / 2) <= 2 * eps
    mesh.on_boundary = np.any(np.c_[mask1, mask2, mask3, mask4], axis=1)

    return mesh


def circle(rl, refn=1, center=(0, 0, 0)):
    '''
    [summary]

    Parameters
    ----------
    rl : [type]
        [description]
    refn : int, optional
        [description], by default 1
    center : tuple, optional
        [description], by default (0, 0, 0)

    Returns
    -------
    [type]
        [description]
    '''
    v, e, t, s = geometry_circle(rl, n=4, refn=refn)
    v += np.array(center)
    mesh = Mesh.from_geometry(v, e, t, s)

    x, y, z = (mesh.vertices).T
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask = np.abs(r - rl) <= 2 * eps
    mesh.on_boundary = mask

    return mesh


def geometry_square3(xl, yl):
    '''
    Prototype mesh (type 3) for square membranes; suitable for 3 by 3 patches.

    Parameters
    ----------
    xl : [type]
        [description]
    yl : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    # vertices
    v = np.zeros((25, 3), dtype=np.float64)
    v[0, :] = -xl / 2, -yl / 2, 0.0
    v[1, :] = -xl / 6, -yl / 2, 0.0
    v[2, :] = xl / 6, -yl / 2, 0.0
    v[3, :] = xl / 2, -yl / 2, 0.0
    v[4, :] = -xl / 3, -yl / 3, 0.0
    v[5, :] = 0.0, -yl / 3, 0.0
    v[6, :] = xl / 3, -yl / 3, 0.0
    v[7, :] = -xl / 2, -yl / 6, 0.0
    v[8, :] = -xl / 6, -yl / 6, 0.0
    v[9, :] = xl / 6, -yl / 6, 0.0
    v[10, :] = xl / 2, -yl / 6, 0.0
    v[11, :] = -xl / 3, 0.0, 0.0
    v[12, :] = 0.0, 0.0, 0.0
    v[13, :] = xl / 3, 0.0, 0.0
    v[14, :] = -xl / 2, yl / 6, 0.0
    v[15, :] = -xl / 6, yl / 6, 0.0
    v[16, :] = xl / 6, yl / 6, 0.0
    v[17, :] = xl / 2, yl / 6, 0.0
    v[18, :] = -xl / 3, yl / 3, 0.0
    v[19, :] = 0.0, yl / 3, 0.0
    v[20, :] = xl / 3, yl / 3, 0.0
    v[21, :] = -xl / 2, yl / 2, 0.0
    v[22, :] = -xl / 6, yl / 2, 0.0
    v[23, :] = xl / 6, yl / 2, 0.0
    v[24, :] = xl / 2, yl / 2, 0.0

    #  edges
    e = np.zeros((60, 2), dtype=np.uint32)
    e[0, :] = 0, 1
    e[1, :] = 1, 2
    e[2, :] = 2, 3
    e[3, :] = 0, 4
    e[4, :] = 1, 4
    e[5, :] = 1, 5
    e[6, :] = 2, 5
    e[7, :] = 2, 6
    e[8, :] = 3, 6
    e[9, :] = 0, 7
    e[10, :] = 1, 8
    e[11, :] = 2, 9
    e[12, :] = 3, 10
    e[13, :] = 4, 7
    e[14, :] = 4, 8
    e[15, :] = 5, 8
    e[16, :] = 5, 9
    e[17, :] = 6, 9
    e[18, :] = 6, 10
    e[19, :] = 7, 8
    e[20, :] = 8, 9
    e[21, :] = 9, 10

    e[22, :] = 7, 11
    e[23, :] = 8, 11
    e[24, :] = 8, 12
    e[25, :] = 9, 12
    e[26, :] = 9, 13
    e[27, :] = 10, 13
    e[28, :] = 7, 14
    e[29, :] = 8, 15
    e[30, :] = 9, 16
    e[31, :] = 10, 17
    e[32, :] = 11, 14
    e[33, :] = 11, 15
    e[34, :] = 12, 15
    e[35, :] = 12, 16
    e[36, :] = 13, 16
    e[37, :] = 13, 17
    e[38, :] = 14, 15
    e[39, :] = 15, 16
    e[40, :] = 16, 17

    e[41, :] = 14, 18
    e[42, :] = 15, 18
    e[43, :] = 15, 19
    e[44, :] = 16, 19
    e[45, :] = 16, 20
    e[46, :] = 17, 20
    e[47, :] = 14, 21
    e[48, :] = 15, 22
    e[49, :] = 16, 23
    e[50, :] = 17, 24
    e[51, :] = 18, 21
    e[52, :] = 18, 22
    e[53, :] = 19, 22
    e[54, :] = 19, 23
    e[55, :] = 20, 23
    e[56, :] = 20, 24
    e[57, :] = 21, 22
    e[58, :] = 22, 23
    e[59, :] = 23, 24

    #  triangles and triangle edges
    t = np.zeros((36, 3), dtype=np.uint32)
    t[0, :] = 0, 1, 4
    t[1, :] = 1, 8, 4
    t[2, :] = 8, 7, 4
    t[3, :] = 7, 0, 4
    t[4, :] = 1, 2, 5
    t[5, :] = 2, 9, 5
    t[6, :] = 9, 8, 5
    t[7, :] = 8, 1, 5
    t[8, :] = 2, 3, 6
    t[9, :] = 3, 10, 6
    t[10, :] = 10, 9, 6
    t[11, :] = 9, 2, 6

    t[12, :] = 7, 8, 11
    t[13, :] = 8, 15, 11
    t[14, :] = 15, 14, 11
    t[15, :] = 14, 7, 11
    t[16, :] = 8, 9, 12
    t[17, :] = 9, 16, 12
    t[18, :] = 16, 15, 12
    t[19, :] = 15, 8, 12
    t[20, :] = 9, 10, 13
    t[21, :] = 10, 17, 13
    t[22, :] = 17, 16, 13
    t[23, :] = 16, 9, 13

    t[24, :] = 14, 15, 18
    t[25, :] = 15, 22, 18
    t[26, :] = 22, 21, 18
    t[27, :] = 21, 14, 18
    t[28, :] = 15, 16, 19
    t[29, :] = 16, 23, 19
    t[30, :] = 23, 22, 19
    t[31, :] = 22, 15, 19
    t[32, :] = 16, 17, 20
    t[33, :] = 17, 24, 20
    t[34, :] = 24, 23, 20
    t[35, :] = 23, 16, 20

    s = triangle_edges_from_triangles(t, e)

    return v, e, t, s


def triangle_edges_from_triangles(triangles, edges):
    '''
    Assign edges to triangles based on triangle vertices. Edges must be on opposite side of
    their corresponding vertex.

    Parameters
    ----------
    triangles : [type]
        [description]
    edges : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    '''
    triangle_edges = np.zeros_like(triangles)
    for t, tri in enumerate(triangles):
        a, b, c = tri

        e0 = np.where(
            np.logical_and(np.any(edges == b, axis=1),
                           np.any(edges == c, axis=1)))[0]
        if len(e0) == 0 or len(e0) > 1:
            raise RuntimeError(
                f'could not determine corresponding edge for triangle {tri}')

        e1 = np.where(
            np.logical_and(np.any(edges == c, axis=1),
                           np.any(edges == a, axis=1)))[0]
        if len(e1) == 0 or len(e1) > 1:
            raise RuntimeError(
                f'could not determine corresponding edge for triangle {tri}')

        e2 = np.where(
            np.logical_and(np.any(edges == a, axis=1),
                           np.any(edges == b, axis=1)))[0]
        if len(e2) == 0 or len(e2) > 1:
            raise RuntimeError(
                f'could not determine corresponding edge for triangle {tri}')

        triangle_edges[t, :] = e0, e1, e2

    return triangle_edges


def matrix_array(nx, ny, pitchx, pitchy, xl, yl, refn=1, **kwargs):
    '''
    Convenience function for a mesh representing a matrix array.

    Parameters
    ----------
    nx : [type]
        [description]
    ny : [type]
        [description]
    pitchx : [type]
        [description]
    pitchy : [type]
        [description]
    xl : [type]
        [description]
    yl : [type]
        [description]
    refn : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    '''
    lengthx, lengthy = pitchx * (nx - 1), pitchy * (ny - 1)
    xv = np.linspace(-lengthx / 2, lengthx / 2, nx)
    yv = np.linspace(-lengthy / 2, lengthy / 2, ny)
    zv = 0
    centers = np.stack(np.meshgrid(xv, yv, zv), axis=-1).reshape((-1, 3))

    verts, edges, tris, tri_edges = [], [], [], []
    vidx = 0
    eidx = 0

    for c in centers:
        v, e, t, s = geometry_square(xl, yl, refn=refn)

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

    mesh = Mesh.from_geometry(verts, edges, tris, tri_edges)
    mesh.on_boundary = np.zeros(len(mesh.vertices), dtype=np.bool)

    # check and flag boundary vertices
    x, y, z = mesh.vertices.T
    for cx, cy, cz in centers:
        xmin = cx - xl / 2 - 2 * eps
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
        mesh.on_boundary[mem_mask] = np.any(np.c_[mask1, mask2, mask3, mask4],
                                            axis=1)

    return mesh


def linear_array():
    '''
    Convenience function for a mesh representing a linear array.

    Raises
    ------
    NotImplementedError
        [description]
    '''
    raise NotImplementedError


def interpolator(mesh, func, function='cubic'):
    '''
    Returns an interpolator for function f defined on the nodes of the given mesh.

    Parameters
    ----------
    mesh : [type]
        [description]
    func : [type]
        [description]
    function : str, optional
        [description], by default 'cubic'

    Returns
    -------
    [type]
        [description]
    '''
    if isinstance(mesh, Mesh):
        return Rbf(mesh.vertices[:, 0],
                   mesh.vertices[:, 1],
                   func,
                   function=function,
                   smooth=0)
    else:
        x, y = mesh
        return Rbf(x, y, func, function=function, smooth=0)


def integrator(mesh, func, function='linear'):
    '''
    [summary]

    Parameters
    ----------
    mesh : [type]
        [description]
    func : [type]
        [description]
    function : str, optional
        [description], by default 'linear'

    Raises
    ------
    NotImplementedError
        [description]
    '''
    raise NotImplementedError

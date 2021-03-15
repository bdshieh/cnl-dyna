''''''
import numpy as np
from matplotlib import pyplot as plt
from cnld import mesh
from itertools import cycle


class BaseGrid:

    @property
    def vertices(self):
        return np.asarray(self._mesh.vertices)

    @property
    def edges(self):
        return np.asarray(self._mesh.edges)

    @property
    def triangles(self):
        return np.asarray(self._mesh.triangles)

    @property
    def triangle_edges(self):
        return np.asarray(self._mesh.triangle_edges)

    @property
    def normals(self):
        return np.asarray(self._mesh.normals)

    @property
    def g(self):
        return np.asarray(self._mesh.g)

    @property
    def triangle_areas(self):
        return np.asarray(self._mesh.triangle_areas)

    @property
    def on_boundary(self):
        return np.asarray(self._mesh.on_boundary)

    @property
    def triangle_neighbors(self):
        return self._triangle_neighbors

    @property
    def hmin(self):
        return self._mesh.hmin

    @property
    def hmax(self):
        return self._mesh.hmax

    @property
    def nvertices(self):
        return len(self.vertices)

    @property
    def nedges(self):
        return len(self.edges)

    @property
    def ntriangles(self):
        return len(self.triangles)

    def refine(self, n=1):
        self._mesh.refine(n)

    def translate(self, r):
        self._mesh.translate(r)

    def plot(self):
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

    def _find_triangle_neighbors(self):
        # determine list of neighbors for each triangle
        # None indicates neighbor doesn't exist for that edge (boundary edge)
        triangle_neighbors = [None] * self.ntriangles
        for i, tt in enumerate(range(self.ntriangles)):
            neighbors = []
            for te in self.triangle_edges[tt, :]:
                mask = np.any(self.triangle_edges == te, axis=1)
                args = np.nonzero(mask)[0]
                if len(args) > 1:
                    neighbors.append(args[args != tt][0])
                else:
                    neighbors.append(None)
            triangle_neighbors[i] = neighbors

        self._triangle_neighbors = triangle_neighbors


class BemGrid(BaseGrid):

    def __init__(self, layout, refn, square_type=1):

        mapping = layout.membrane_to_geometry_mapping
        if mapping is None:
            gid = cycle(range(len(layout.geometries)))
            mapping = [next(gid) for i in range(len(layout.membranes))]

        verts, edges, tris, tri_edges = [], [], [], []
        vidx = 0
        eidx = 0

        for i, mem in enumerate(layout.membranes):
            g = layout.geometries[mapping[i]]

            if g.shape == 'circle':
                v, e, t, s = mesh.geometry_circle(g.radius, n=4, refn=refn)
            elif g.shape == 'square':
                v, e, t, s = mesh.geometry_square(g.length_x,
                                                  g.length_y,
                                                  refn=refn)
            else:
                raise ValueError

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
        amesh = mesh.Mesh.from_geometry(verts, edges, tris, tri_edges)

        # assign mesh vertices to patches, membranes, and elements
        nverts = len(amesh.vertices)
        # patch_counter = np.zeros(nverts, dtype=np.int32) # keeps track of current patch idx for each vertex
        # patch_ids = np.ones((nverts, 4), dtype=np.int32) * np.nan
        membrane_ids = np.ones(nverts, dtype=np.int32) * np.nan
        # element_ids = np.ones(nverts, dtype=np.int32) * np.nan
        amesh.on_boundary = np.zeros(nverts, dtype=np.bool)
        x, y, z = amesh.vertices.T

        for i, mem in enumerate(layout.membranes):
            g = layout.geometries[mapping[i]]

            if g.shape == 'square':
                # determine vertices which belong to each membrane
                mem_x, mem_y, mem_z = mem.position
                length_x, length_y = g.length_x, g.length_y
                xmin = mem_x - length_x / 2  # - 2 * mesh.eps
                xmax = mem_x + length_x / 2  # + 2 * mesh.eps
                ymin = mem_y - length_y / 2  # - 2 * mesh.eps
                ymax = mem_y + length_y / 2  # + 2 * mesh.eps
                mask_x = np.logical_and(x >= xmin - 2 * mesh.eps,
                                        x <= xmax + 2 * mesh.eps)
                mask_y = np.logical_and(y >= ymin - 2 * mesh.eps,
                                        y <= ymax + 2 * mesh.eps)
                mem_mask = np.logical_and(mask_x, mask_y)
                membrane_ids[mem_mask] = mem.id
                # element_ids[mem_mask] = elem.id

                # check and flag boundary vertices
                mask1 = np.abs(x[mem_mask] - xmin) <= 2 * mesh.eps
                mask2 = np.abs(x[mem_mask] - xmax) <= 2 * mesh.eps
                mask3 = np.abs(y[mem_mask] - ymin) <= 2 * mesh.eps
                mask4 = np.abs(y[mem_mask] - ymax) <= 2 * mesh.eps
                amesh.on_boundary[mem_mask] = np.any(np.c_[mask1, mask2, mask3,
                                                           mask4],
                                                     axis=1)

            elif g.shape == 'circle':
                # determine vertices which belong to each membrane
                mem_x, mem_y, mem_z = mem.position
                radius = g.radius
                rmax = radius + 2 * mesh.eps
                r = np.sqrt((x - mem_x)**2 + (y - mem_y)**2)
                mem_mask = r <= rmax
                membrane_ids[mem_mask] = mem.id
                # element_ids[mem_mask] = elem.id

                # check and flag boundary vertices
                mask1 = r[mem_mask] <= radius + 2 * mesh.eps
                mask2 = r[mem_mask] >= radius - 2 * mesh.eps
                amesh.on_boundary[mem_mask] = np.logical_and(mask1, mask2)

            else:
                raise ValueError

        # check that no vertices were missed
        # assert ~np.any(np.isnan(patch_ids[:,0])) # check that each vertex is
        # assigned to at least one patch
        # assert ~np.any(np.isnan(membrane_ids))
        # assert ~np.any(np.isnan(element_ids))

        # amesh.patch_ids = patch_ids
        amesh.membrane_ids = membrane_ids
        # amesh.element_ids = element_ids

        self._mesh = amesh
        self._find_triangle_neighbors()


class FemGrid(BaseGrid):

    def __init__(self, geom, refn, square_type=1):

        if geom.shape == 'circle':
            amesh = mesh.circle(geom.radius, refn=refn)
        elif geom.shape == 'square':
            amesh = mesh.square(geom.length_x, geom.length_y, refn=refn)
        else:
            raise ValueError

        # construct mesh from geometry
        self._mesh = amesh
        self._find_triangle_neighbors()


class Grids:

    def __init__(self, bem, fem):
        self._bem = bem
        self._fem = fem

    @property
    def bem(self):
        return self._bem

    @property
    def fem(self):
        return self._fem


def generate_grids_from_layout(layout, refn, square_type=1):
    bem_grid = BemGrid(layout, refn, square_type)

    fem_grids = []
    for geom in layout.geometries:
        fem_grids.append(FemGrid(geom, refn, square_type))

    return Grids(bem_grid, fem_grids)


def import_grid():
    pass


def export_grid():
    pass

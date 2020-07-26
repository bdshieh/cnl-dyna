''''''


class Grid(object):
    _surface = None

    def __init__(self):
        self._surface = None

    @classmethod
    def from_abstract(cls, array, refn=1, **kwargs):
        return _from_abstract(cls, array, refn, **kwargs)

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

    def _memoize():
        return (self.nvertices, self.triangles.tostring(),
                self.edges.tostring(), self.triangle_edges.tostring())


def _from_abstract(cls, array, refn=1, **kwargs):

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


def import_grid():
    pass


def export_grid():
    pass

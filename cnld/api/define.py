''''''
import numpy as np
from namedlist import namedlist, FACTORY
from collections import OrderedDict
from itertools import cycle
from cnld.datatypes import register_mapping, register_list
from cnld.util import distance

GeometryData = register_mapping(
    'GeometryData',
    OrderedDict(
        id=None,
        thickness=None,
        shape=None,
        lengthx=None,
        lengthy=None,
        radius=None,
        rho=None,
        ymod=None,
        prat=None,
        isol_thickness=None,
        eps_r=None,
        gap=None,
        electrode_x=None,
        electrode_y=None,
        electrode_r=None,
    ))

Geometries = register_list('Geometries', GeometryData)

MembraneData = register_mapping('MembraneData',
                                OrderedDict(
                                    id=None,
                                    position=None,
                                ))

Membranes = register_list('Membranes', MembraneData)

ElementData = register_mapping(
    'ElementData',
    OrderedDict(
        id=None,
        position=None,
        membrane_ids=FACTORY(list),
    ))

Elements = register_list('Elements', ElementData)

ControlDomainData = register_mapping(
    'ControlDomainData',
    OrderedDict(
        id=None,
        position=None,
        shape=None,
        lengthx=None,
        lengthy=None,
        radius_min=None,
        radius_max=None,
        theta_min=None,
        theta_max=None,
        area=None,
    ))

ControlDomains = register_list('ControlDomains', ControlDomainData)

Layout = register_mapping(
    'Layout',
    OrderedDict(
        membranes=FACTORY(Membranes),
        elements=FACTORY(Elements),
        controldomains=FACTORY(ControlDomains),
    ))

# class Layout(object):
#     def __init__(self, membranes=None, elements=None, controldomains=None):
#         self._membranes = Membranes(membranes)
#         self._elements = Elements(elements)
#         self._controldomains = ControlDomains(controldomains)

#     @classmethod
#     def empty(cls, nmembrane, nelement, ncontroldomain):
#         return cls(membranes=[None] * nmembrane,
#                    elements=[None] * nelement,
#                    controldomains=[None] * ncontroldomain)

#     @property
#     def Membranes(self):
#         return self._membranes

#     @property
#     def Elements(self):
#         return self._elements

#     @property
#     def ControlDomains(self):
#         return self._controldomains

#     @property
#     def nmembrane(self):
#         return len(self.Membranes)

#     @property
#     def nelement(self):
#         return len(self.Elements)

#     @property
#     def ncontroldomain(self):
#         return len(self.ControlDomains)

#     @Membranes.setter
#     def Membranes(self, val):
#         self._membranes = Membranes(val)

#     @Elements.setter
#     def Elements(self, val):
#         self._elements = Elements(val)

#     @ControlDomains.setter
#     def ControlDomains(self, val):
#         self._controldomains = ControlDomains(val)

#     def plot(self):
#         pass


def generate_control_domains(layout,
                             geometry,
                             nx=3,
                             ny=3,
                             nr=3,
                             ntheta=4,
                             mapping=None):

    if mapping is None:
        gid = cycle(range(len(geometry)))
        mapping = [next(gid) for i in range(len(layout.membranes))]

    data = []
    cid = 0
    for i, mem in enumerate(layout.membranes):
        g = geometry[mapping[i]]

        if g.shape == 'square':
            pitchx = g.lengthx / nx
            pitchy = g.lengthy / ny
            cx, cy, cz = np.meshgrid(
                np.arange(nx) * pitchx,
                np.arange(ny) * pitchy, 0)
            cx -= (nx - 1) * pitchx / 2
            cy -= (ny - 1) * pitchy / 2
            centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

            for c in centers:
                data.append(
                    ControlDomainData(
                        id=cid,
                        position=list(mem.position + c),
                        shape='square',
                        lengthx=pitchx,
                        lengthy=pitchy,
                        area=pitchx * pitchy,
                    ))
                cid += 1

        elif g.shape == 'circle':

            r = np.linspace(0, g.radius, nr + 1)
            theta = np.linspace(-np.pi, np.pi, ntheta + 1)
            rmin = [r[i] for i in range(nr) for j in range(ntheta)]
            rmax = [r[i + 1] for i in range(nr) for j in range(ntheta)]
            thetamin = [theta[j] for i in range(nr) for j in range(ntheta)]
            thetamax = [theta[j + 1] for i in range(nr) for j in range(ntheta)]
            c = np.array([0, 0, 0])

            for j in range(nr * ntheta):
                data.append(
                    ControlDomainData(
                        id=cid,
                        position=list(mem.position + c),
                        shape='circle',
                        radius_min=rmin[j],
                        radius_max=rmax[j],
                        theta_min=thetamin[j],
                        theta_max=thetamax[j],
                        area=(rmax[j]**2 - rmin[j]**2) *
                        (thetamax[j] - thetamin[j]) / 2,
                    ))
                cid += 1
        else:
            raise TypeError

    layout.controldomains = ControlDomains(data)


# # def import_layout(file):
# #     return Layout(json.load(open(file, 'r')))

# # def export_layout(layout, file, mode='w'):
# #     json.dump(open(file, mode), layout.json)


def matrix_layout(nx, ny, pitchx, pitchy):

    cx, cy, cz = np.meshgrid(np.arange(nx) * pitchx, np.arange(ny) * pitchy, 0)
    cx -= (nx - 1) * pitchx / 2
    cy -= (ny - 1) * pitchy / 2
    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

    membranes = Membranes()
    elements = Elements()
    for id, pos in enumerate(centers):
        membranes.append(id=id, position=list(pos))
        elements.append(id=id, position=list(pos), membrane_ids=[id])

    return Layout(membranes=membranes, elements=elements)


def hexagonal_layout(nx, ny, pitch):

    pitchx = np.sqrt(3) / 2 * pitch
    pitchy = pitch
    offsety = pitch / 2

    cx, cy, cz = np.meshgrid(np.arange(nx) * pitchx, np.arange(ny) * pitchy, 0)
    cy[:, ::2, :] += offsety / 2
    cy[:, 1::2, :] -= offsety / 2
    cx -= (nx - 1) * pitchx / 2
    cy -= (ny - 1) * pitchy / 2

    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

    membranes = Membranes()
    elements = Elements()
    for id, pos in enumerate(centers):
        membranes.append(id=id, position=list(pos))
        elements.append(id=id, position=list(pos), membrane_ids=[id])

    return Layout(membranes=membranes, elements=elements)


def linear_matrix_layout(nelem, pitch, nmem, mempitch):

    cx, cy, cz = np.meshgrid(np.arange(nelem) * pitch, 0, 0)
    cx -= (nelem - 1) * pitch / 2
    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]
    mx, my, mz = np.meshgrid(0, np.arange(nmem) * mempitch, 0)
    my -= (nmem - 1) * mempitch / 2
    mem_centers = np.c_[mx.ravel(), my.ravel(), mz.ravel()]

    membranes = Membranes()
    elements = Elements()
    mid = 0
    for id, pos in enumerate(centers):
        membrane_ids = []
        for mpos in mem_centers:
            membranes.append(id=mid, position=list(pos + mpos))
            membrane_ids.append(mid)
            mid += 1

        elements.append(id=id, position=list(pos), membrane_ids=membrane_ids)

    return Layout(membranes=membranes, elements=elements)


def linear_hexagonal_layout(nx, ny, pitch):
    pass


def square_cmut_1mhz_geometry(**kwargs):
    data = GeometryData(id=0,
                        thickness=1e-6,
                        shape='square',
                        lengthx=35e-6,
                        lengthy=35e-6,
                        rho=2040,
                        ymod=110e9,
                        prat=0.2,
                        isol_thickness=100e-9,
                        eps_r=1.2,
                        gap=50e-9,
                        electrode_x=35e-6,
                        electrode_y=35e-6)

    for k, v in kwargs.items():
        if k in data:
            data[k] = v

    return Geometries([data])


def circle_cmut_1mhz_geometry(**kwargs):
    data = GeometryData(id=0,
                        thickness=1e-6,
                        shape='circle',
                        radius=35e-6,
                        rho=2040,
                        ymod=110e9,
                        prat=0.2,
                        isol_thickness=100e-9,
                        eps_r=1.2,
                        gap=50e-9,
                        electrode_r=20e-6)

    for k, v in kwargs.items():
        if k in data:
            data[k] = v

    return Geometries([data])


BeamformData = register_mapping('BeamformData',
                                OrderedDict(
                                    id=None,
                                    apod=1,
                                    delay=0,
                                ))

Beamforms = register_list('Beamforms', BeamformData)

WaveformData = register_mapping(
    'WaveformData', OrderedDict(
        id=None,
        time=None,
        voltage=None,
        fs=None,
    ))

Waveforms = register_list('Waveforms', WaveformData)

Transmit = register_mapping('Transmit',
                            OrderedDict(
                                waveforms=None,
                                beamforms=None,
                            ))


def calculate_delays(transmit, layout, focus, c=1500, fs=None, offset=True):

    for elem, bf in zip(layout.Elements, transmit.Beamforms):
        d = float(distance(elem.center, focus))

        if fs is None:
            t = d / c
        else:
            t = round(d / c * fs) / fs

        bf.delay = -t

    if offset:
        tmin = np.min(transmit.Beamforms.delays)
        new_delays = np.array(transmit.Beamforms.delays) - tmin
        transmit.Beamforms.delays = new_delays
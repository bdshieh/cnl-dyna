''''''
import numpy as np
from namedlist import FACTORY
from collections import OrderedDict
from itertools import cycle
from cnld.datatypes import register_mapping, register_list
from cnld.util import distance

Geometry = register_mapping(
    'Geometry',
    OrderedDict(
        id=None,
        thickness=None,
        shape=None,
        length_x=None,
        length_y=None,
        radius=None,
        density=None,
        y_modulus=None,
        p_ratio=None,
        isol_thickness=None,
        eps_r=None,
        gap=None,
        electrode_x=None,
        electrode_y=None,
        electrode_r=None,
        controldomain_nx=None,
        controldomain_ny=None,
        controldomain_nr=None,
        controldomain_ntheta=None,
        damping_mode1=None,
        damping_mode2=None,
        damping_freq1=None,
        damping_freq2=None,
        damping_ratio1=None,
        damping_ratio2=None,
        contact_k=None,
        contact_n=None,
        contact_z0=None,
        contact_lmda=None,
    ))

GeometryList = register_list('GeometryList', Geometry)

Membrane = register_mapping('Membrane', OrderedDict(
    id=None,
    position=None,
))

MembraneList = register_list('MembraneList', Membrane)

Element = register_mapping(
    'Element', OrderedDict(
        id=None,
        position=None,
        membrane_ids=FACTORY(list),
    ))

ElementList = register_list('ElementList', Element)

ControlDomain = register_mapping(
    'ControlDomain',
    OrderedDict(
        id=None,
        membrane_id=None,
        element_id=None,
        position=None,
        shape=None,
        length_x=None,
        length_y=None,
        radius_min=None,
        radius_max=None,
        theta_min=None,
        theta_max=None,
        area=None,
    ))

ControlDomainList = register_list('ControlDomainList', ControlDomain)

Layout = register_mapping(
    'Layout',
    OrderedDict(
        geometries=FACTORY(GeometryList),
        membranes=FACTORY(MembraneList),
        elements=FACTORY(ElementList),
        controldomains=FACTORY(ControlDomainList),
        membrane_to_geometry_mapping=None,
    ))


def controldomainlist_from_geometry(geom):
    '''
    '''
    ctrldomlist = []
    cid = 0
    if geom.shape == 'square':

        nx = geom.controldomain_nx
        ny = geom.controldomain_ny
        pitch_x = geom.length_x / nx
        pitch_y = geom.length_y / ny
        cx, cy, cz = np.meshgrid(
            np.arange(nx) * pitch_x,
            np.arange(ny) * pitch_y, 0)
        cx -= (nx - 1) * pitch_x / 2
        cy -= (ny - 1) * pitch_y / 2
        centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

        for c in centers:
            ctrldomlist.append(
                ControlDomain(
                    id=cid,
                    position=list(c),
                    shape='square',
                    length_x=pitch_x,
                    length_y=pitch_y,
                    area=pitch_x * pitch_y,
                ))
            cid += 1

    elif geom.shape == 'circle':

        nr = geom.controldomain_nr
        ntheta = geom.controldomain_ntheta
        r = np.linspace(0, geom.radius, nr + 1)
        theta = np.linspace(-np.pi, np.pi, ntheta + 1)
        rmin = [r[i] for i in range(nr) for j in range(ntheta)]
        rmax = [r[i + 1] for i in range(nr) for j in range(ntheta)]
        thetamin = [theta[j] for i in range(nr) for j in range(ntheta)]
        thetamax = [theta[j + 1] for i in range(nr) for j in range(ntheta)]
        c = np.array([0, 0, 0])

        for j in range(nr * ntheta):
            ctrldomlist.append(
                ControlDomain(
                    id=cid,
                    position=list(c),
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

    return ControlDomainList(ctrldomlist)


def generate_controldomainlist(layout, mapping=None):
    '''
    '''
    geometries = layout.geometries
    elements = layout.elements

    if mapping is None:
        mapping = layout.membrane_to_geometry_mapping

    if mapping is None:
        gid = cycle(range(len(geometries)))
        mapping = [next(gid) for i in range(len(layout.membranes))]

    mem_ids = np.array(elements.membrane_ids)
    ctrldomlist = []
    cid = 0

    for i, mem in enumerate(layout.membranes):
        g = geometries[mapping[i]]

        idx = np.argwhere(mem_ids == mem.id)
        assert len(idx) == 1
        eid = idx[0, 0]

        if g.shape == 'square':

            nx = g.controldomain_nx
            ny = g.controldomain_ny
            pitch_x = g.length_x / nx
            pitch_y = g.length_y / ny
            cx, cy, cz = np.meshgrid(
                np.arange(nx) * pitch_x,
                np.arange(ny) * pitch_y, 0)
            cx -= (nx - 1) * pitch_x / 2
            cy -= (ny - 1) * pitch_y / 2
            centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

            for c in centers:
                ctrldomlist.append(
                    ControlDomain(
                        id=cid,
                        membrane_id=mem.id,
                        element_id=eid,
                        position=list(mem.position + c),
                        shape='square',
                        length_x=pitch_x,
                        length_y=pitch_y,
                        area=pitch_x * pitch_y,
                    ))
                cid += 1

        elif g.shape == 'circle':

            nr = g.controldomain_nr
            ntheta = g.controldomain_ntheta
            r = np.linspace(0, g.radius, nr + 1)
            theta = np.linspace(-np.pi, np.pi, ntheta + 1)
            rmin = [r[i] for i in range(nr) for j in range(ntheta)]
            rmax = [r[i + 1] for i in range(nr) for j in range(ntheta)]
            thetamin = [theta[j] for i in range(nr) for j in range(ntheta)]
            thetamax = [theta[j + 1] for i in range(nr) for j in range(ntheta)]
            c = np.array([0, 0, 0])

            for j in range(nr * ntheta):
                ctrldomlist.append(
                    ControlDomain(
                        id=cid,
                        membrane_id=mem.id,
                        element_id=eid,
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

    return ControlDomainList(ctrldomlist)


# # def import_layout(file):
# #     return Layout(json.load(open(file, 'r')))

# # def export_layout(layout, file, mode='w'):
# #     json.dump(open(file, mode), layout.json)


def matrix_layout(nx, ny, pitch_x, pitch_y):
    '''
    [summary]

    Parameters
    ----------
    nx : [type]
        [description]
    ny : [type]
        [description]
    pitch_x : [type]
        [description]
    pitch_y : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    cx, cy, cz = np.meshgrid(
        np.arange(nx) * pitch_x,
        np.arange(ny) * pitch_y, 0)
    cx -= (nx - 1) * pitch_x / 2
    cy -= (ny - 1) * pitch_y / 2
    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

    membranes = MembraneList()
    elements = ElementList()
    for id, pos in enumerate(centers):
        membranes.append(id=id, position=list(pos))
        elements.append(id=id, position=list(pos), membrane_ids=[id])

    return Layout(membranes=membranes, elements=elements)


def hexagonal_layout(nx, ny, pitch):
    '''
    [summary]

    Parameters
    ----------
    nx : [type]
        [description]
    ny : [type]
        [description]
    pitch : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    pitch_x = np.sqrt(3) / 2 * pitch
    pitch_y = pitch
    offsety = pitch / 2

    cx, cy, cz = np.meshgrid(
        np.arange(nx) * pitch_x,
        np.arange(ny) * pitch_y, 0)
    cy[:, ::2, :] += offsety / 2
    cy[:, 1::2, :] -= offsety / 2
    cx -= (nx - 1) * pitch_x / 2
    cy -= (ny - 1) * pitch_y / 2

    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

    membranes = MembraneList()
    elements = ElementList()
    for id, pos in enumerate(centers):
        membranes.append(id=id, position=list(pos))
        elements.append(id=id, position=list(pos), membrane_ids=[id])

    return Layout(membranes=membranes, elements=elements)


def linear_matrix_layout(nelem, pitch, nmem, mempitch):
    '''
    [summary]

    Parameters
    ----------
    nelem : [type]
        [description]
    pitch : [type]
        [description]
    nmem : [type]
        [description]
    mempitch : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    cx, cy, cz = np.meshgrid(np.arange(nelem) * pitch, 0, 0)
    cx -= (nelem - 1) * pitch / 2
    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]
    mx, my, mz = np.meshgrid(0, np.arange(nmem) * mempitch, 0)
    my -= (nmem - 1) * mempitch / 2
    mem_centers = np.c_[mx.ravel(), my.ravel(), mz.ravel()]

    membranes = MembraneList()
    elements = ElementList()
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
    '''
    [summary]

    Parameters
    ----------
    nx : [type]
        [description]
    ny : [type]
        [description]
    pitch : [type]
        [description]
    '''
    pass


# def square_cmut_1mhz_geometry(**kwargs):
#     '''
#     '''
#     data = Geometry(id=0,
#                     thickness=1e-6,
#                     shape='square',
#                     length_x=35e-6,
#                     length_y=35e-6,
#                     density=2040,
#                     y_modulus=110e9,
#                     p_ratio=0.2,
#                     isol_thickness=100e-9,
#                     eps_r=1.2,
#                     gap=50e-9,
#                     electrode_x=35e-6,
#                     electrode_y=35e-6,
#                     controldomain_nx=3,
#                     controldomain_ny=3,
#                     damping_mode1=1,
#                     damping_mode2=2,
#                     damping_freq1=1e6,
#                     damping_freq2=10e6,
#                     damping_ratio1=0.03,
#                     damping_ratio2=0.03)

#     for k, v in kwargs.items():
#         if k in data:
#             data[k] = v

#     return data


def circular_cmut_5mhz_geometry(**kwargs):
    '''
    '''
    data = Geometry(id=0,
                    thickness=1e-6,
                    shape='circle',
                    radius=26e-6,
                    density=3000,
                    y_modulus=160e9,
                    p_ratio=0.28,
                    isol_thickness=100e-9,
                    eps_r=7.5,
                    gap=300e-9,
                    electrode_r=50e-6,
                    controldomain_nr=3,
                    controldomain_ntheta=4,
                    damping_mode1=0,
                    damping_mode2=5,
                    damping_freq1=0,
                    damping_freq2=0,
                    damping_ratio1=0.1,
                    damping_ratio2=0.1)

    for k, v in kwargs.items():
        if k in data:
            data[k] = v

    return data


def circular_cmut_1mhz_geometry(**kwargs):
    '''
    '''
    data = Geometry(id=0,
                    thickness=1e-6,
                    shape='circle',
                    radius=50e-6,
                    density=3000,
                    y_modulus=160e9,
                    p_ratio=0.28,
                    isol_thickness=100e-9,
                    eps_r=7.5,
                    gap=500e-9,
                    electrode_r=50e-6,
                    controldomain_nr=3,
                    controldomain_ntheta=4,
                    damping_mode1=0,
                    damping_mode2=5,
                    damping_freq1=0,
                    damping_freq2=0,
                    damping_ratio1=0.1,
                    damping_ratio2=0.1)

    for k, v in kwargs.items():
        if k in data:
            data[k] = v

    return data


# BeamformData = register_mapping('BeamformData',
#                                 OrderedDict(
#                                     id=None,
#                                     apod=1,
#                                     delay=0,
#                                 ))

# Beamforms = register_list('Beamforms', BeamformData)

Waveform = register_mapping('Waveform', OrderedDict(
    id=None,
    voltage=None,
))

WaveformList = register_list('WaveformList', Waveform)

Transmit = register_mapping(
    'Transmit',
    OrderedDict(
        waveforms=FACTORY(WaveformList),
        focus=None,
        apod=None,
        delays=None,
        element_to_waveform_mapping=None,
        fs=None,
    ))


def generate_delays(layout, transmit, c=1500, fs=None, offset=True):
    '''
    [summary]

    Parameters
    ----------
    transmit : [type]
        [description]
    layout : [type]
        [description]
    c : int, optional
        [description], by default 1500
    fs : [type], optional
        [description], by default None
    offset : bool, optional
        [description], by default True
    '''
    if transmit.focus is None:
        delays = [0] * len(layout.elements)
    else:
        centers = np.array(layout.elements.center)
        d = float(distance(centers, transmit.focus))

        if fs is None:
            delays = -d / c
        else:
            delays = round(-d / c * fs) / fs

    if offset:
        delays -= delays.min()

    return list(delays)

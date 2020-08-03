''''''
import numpy as np
from namedlist import namedlist
import json
from collections import OrderedDict
from itertools import cycle

ControlDomainData = namedlist(
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


class ControlDomain(object):
    def __init__(self, data):

        if not isinstance(data, list):
            raise TypeError

        _data = [None] * len(data)
        for v in data:

            if not isinstance(v, ControlDomainData):
                _v = ControlDomainData(**v)
            else:
                _v = v

            _data[int(_v.id)] = _v

        self._data = data

    def __getitem__(self, id):
        return self._data[id]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def id(self):
        return [v.id for v in self]

    @property
    def position(self):
        return [v.position for v in self]

    @property
    def center(self):
        return [v.position for v in self]

    @property
    def shape(self):
        return [v.shape for v in self]

    @property
    def lengthx(self):
        return [v.lengthx for v in self]

    @property
    def lengthy(self):
        return [v.lengthy for v in self]

    @property
    def radius_min(self):
        return [v.radius_min for v in self]

    @property
    def radius_max(self):
        return [v.radius_max for v in self]

    @property
    def theta_min(self):
        return [v.theta_min for v in self]

    @property
    def theta_max(self):
        return [v.theta_max for v in self]

    @property
    def area(self):
        return [v.area for v in self]

    def plot(self):
        pass


def autogenerate_control_domain(layout,
                                geometry,
                                nx=3,
                                ny=3,
                                nr=3,
                                ntheta=4,
                                mapping=None):

    if mapping is None:
        gid = cycle(range(len(geometry)))
        mapping = [next(gid) for i in range(len(layout))]

    data = []
    cid = 0
    for i, l in enumerate(layout):
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
                        position=list(l.position + c),
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
                        position=list(l.position + c),
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

    return ControlDomain(data)

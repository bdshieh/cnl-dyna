''''''
import numpy as np
from namedlist import namedlist
import json
from collections import OrderedDict

LayoutData = namedlist(
    'LayoutData',
    OrderedDict(
        id=None,
        position=None,
        shape=None,
        lengthx=None,
        lengthy=None,
        radius=None,
    ))


class Layout(object):
    def __init__(self, data):

        if not isinstance(data, list):
            raise TypeError

        _data = [None] * len(data)
        for v in data:

            if not isinstance(v, LayoutData):
                _v = LayoutData(**v)
            else:
                _v = v

            _data[int(_v.id)] = _v

        self._data = data

    def __getitem__(self, id):
        return self._data[id]

    def __iter__(self):
        return iter(self._data)

    @property
    def center(self):
        return [v.position for v in self]

    @property
    def id(self):
        return [v.id for v in self]

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
    def radius(self):
        return [v.radius for v in self]

    @property
    def json(self):
        return json.dumps([v._asdict() for v in self])

    def plot(self):
        pass


def import_layout(file):
    return Layout(json.load(open(file, 'r')))


def export_layout(layout, file, mode='w'):
    json.dump(open(file, mode), layout.json)


def matrix_layout(nx, ny, pitchx, pitchy, **kwargs):

    cx, cy, cz = np.meshgrid(np.arange(nx) * pitchx, np.arange(ny) * pitchy, 0)
    cx -= (nx - 1) * pitchx / 2
    cy -= (ny - 1) * pitchy / 2
    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

    layout = []
    for id, pos in enumerate(centers):
        layout.append(LayoutData(id=id, position=list(pos), **kwargs))

    return Layout(layout)


def hexagonal_layout(nx, ny, pitch, **kwargs):

    pitchx = np.sqrt(3) / 2 * pitch
    pitchy = pitch
    offsety = pitch / 2

    cx, cy, cz = np.meshgrid(np.arange(nx) * pitchx, np.arange(ny) * pitchy, 0)
    cy[:, ::2, :] += offsety / 2
    cy[:, 1::2, :] -= offsety / 2
    cx -= (nx - 1) * pitchx / 2
    cy -= (ny - 1) * pitchy / 2

    centers = np.c_[cx.ravel(), cy.ravel(), cz.ravel()]

    layout = []
    for id, pos in enumerate(centers):
        layout.append(LayoutData(id=id, position=list(pos), **kwargs))

    return Layout(layout)


def linear_layout(nelem, pitch):
    pass

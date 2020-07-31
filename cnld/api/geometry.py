''''''
import numpy as np
from namedlist import namedlist
import json
from collections import OrderedDict

GeometryData = namedlist(
    'GeomData',
    dict(id=None,
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
         electrode_r=None))


class Geometry(object):
    def __init__(self, data):

        if not isinstance(data, list):
            raise TypeError

        _data = [None] * len(data)
        for v in data:

            if not isinstance(v, GeometryData):
                _v = GeometryData(**v)
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
    def thickness(self):
        return [v.thickness for v in self]

    @property
    def rho(self):
        return [v.rho for v in self]

    @property
    def ymod(self):
        return [v.ymod for v in self]

    @property
    def prat(self):
        return [v.prat for v in self]

    @property
    def isol_thickness(self):
        return [v.isol_thickness for v in self]

    @property
    def eps_r(self):
        return [v.eps_r for v in self]

    @property
    def gap(self):
        return [v.gap for v in self]

    @property
    def electrode_x(self):
        return [v.electrode_x for v in self]

    @property
    def electrode_y(self):
        return [v.electrode_y for v in self]

    @property
    def electrode_r(self):
        return [v.electrode_r for v in self]

    @property
    def json(self):
        return json.dumps([v._asdict() for v in self])

    def plot(self):
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

    return Geometry([data])


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
                        electrode_x=35e-6,
                        electrode_y=35e-6)

    return Geometry([data])

''''''
from namedlist import namedlist

GeometryData = namedlist(
    'GeomData',
    dict(id=None,
         thickness=None,
         rho=None,
         ymod=None,
         prat=None,
         isol_thickness=None,
         eps_r=None,
         gap=None,
         electrode_x=None,
         electrode_y=None,
         electrode_r=None))


class Geometry():
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

    @property
    def id(self):
        return [v.id for v in self]

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


def cmut1_geometry(**kwargs):
    pass


def cmut5_geometry(**kwargs):
    pass


def cmut20_geometry(**kwargs):
    pass

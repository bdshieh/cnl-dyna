''''''
from namedlist import namedlist, FACTORY
import json
from collections import MutableMapping, Sequence, OrderedDict


class BaseMapping(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._data = self._type(*args, **kwargs)

    def __repr__(self):
        if 'id' in self:
            repr = [
                '<',
                type(self).__name__, '(id=',
                str(self.id), ')', ' with ',
                str(len(self)), ' items', '>'
            ]
        else:
            repr = [
                '<',
                type(self).__name__, ' with ',
                str(len(self)), ' items', '>'
            ]
        return ''.join(repr)

    def __str__(self):
        indent = 2
        strings = []
        strings += [' ' * indent, type(self).__name__, '\n']
        for key, val in self.items():
            strings += [' ' * (indent + 1), str(key), ': ', repr(val), '\n']
        return ''.join(strings)

    def keys(self):
        return self._data._asdict().keys()

    def __contains__(self, key):
        return key in self._data._asdict()

    def __getitem__(self, key):
        return self._data._asdict()[key]

    def __setitem__(self, key, val):
        self._data._asdict()[key] = val

    def __delitem__(self, key):
        self._data._asdict()[key] = None

    def __iter__(self):
        return iter(self._data._asdict())

    def __len__(self):
        return len(self._data)

    def __getattr__(self, attr):
        return getattr(self._data, attr)

    def __setattr__(self, attr, val):
        if attr == '_data':
            super(BaseMapping, self).__setattr__(attr, val)
        else:
            setattr(self._data, attr, val)

    def clear(self):
        for k in self:
            self[k] = None

    def update(self, val):
        self._data._update(val)


def register_type(typename, fieldnames):

    t = type(typename, (BaseMapping, ), {})
    t._type = namedlist(typename, fieldnames)
    MutableMapping.register(t)
    return t


GeometryData = register_type(
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


class BaseList(Sequence):
    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, **kwargs):
        self._data[key] = self._dtype(**kwargs)

    def __delitem__(self, key):
        del self._data[key]
        self._align_id_with_index()

    def __iter__(self):
        return iter(self._data)

    def __add__(self, arg):
        self.__init__(self._data + arg._data)
        return self

    def _align_id_with_index(self):
        for i, g in enumerate(self):
            g.id = i

    def append(self, *args, **kwargs):
        if args:
            if len(args) != 1:
                raise ValueError
            self._data.append(args[0])
        elif kwargs:
            self._data.append(self._dtype(**kwargs))
            self._align_id_with_index()

    @property
    def json(self):
        return json.dumps([v._asdict() for v in self])

    def __str__(self):
        BaseList._pretty(self, self._dtype)


# def pretty(obj, indent=0):

#     strings = []
#     if isinstance(obj, BaseMapping):
#         strings += [' ' * indent, type(obj).__name__, '\n']
#         for key, val in obj.items():
#             if isinstance(val, BaseMapping):
#                 strings += [' ' * (indent + 1), str(key), ': ', '\n']
#                 strings += [pretty(val, indent + 1)]
#             elif isinstance(val, (list, tuple)):
#                 strings += [' ' * (indent + 1), str(key), ': ']
#                 strings += [pretty(val, indent + 2)]
#             else:
#                 strings += [' ' * (indent + 1), str(key), ': ', str(val), '\n']
#     elif isinstance(obj, (list, tuple)):
#         if len(obj) == 0:
#             strings += [' ' * indent, '[]', '\n']
#         elif isinstance(obj[0], BaseMapping):
#             for val in obj:
#                 strings += [pretty(val, indent + 1)]
#         elif isinstance(obj[0], (list, tuple)):
#             for val in obj:
#                 strings += [pretty(val, indent + 1)]
#         else:
#             strings += [' ' * indent, str(obj), '\n']
#     else:
#         pass
#     return ''.join(strings)

# def pretty(obj, indent=0):
#     strings = []
#     if isinstance(obj, BaseMapping):
#         strings += [' ' * indent, type(obj).__name__, '\n']
#         for key, val in obj.items():
#             strings += [' ' * (indent + 1), str(key), ': ', repr(val), '\n']
#     elif isinstance(obj, BaseList):
#         pass
#     else:
#         pass
#     return ''.join(strings)

# Data = namedlist('Data', [])

# class BaseList(object):
#     _dtype = Data

#     def __init__(self, data=None):

#         if data is None:
#             data = []
#         elif isinstance(data, self._dtype):
#             data = [data]

#         if not isinstance(data, (list, BaseList)):
#             raise TypeError

#         _data = [None] * len(data)
#         for v in data:

#             if not isinstance(v, self._dtype):
#                 _v = self._dtype(**v)
#             else:
#                 _v = v

#             _data[int(_v.id)] = _v

#         self._data = data
#         self._align_id_with_index()

#     def __len__(self):
#         return len(self._data)

#     def __getitem__(self, key):
#         return self._data[key]

#     def __setitem__(self, key, **kwargs):
#         self._data[key] = self._dtype(**kwargs)

#     def __delitem__(self, key):
#         del self._data[key]
#         self._align_id_with_index()

#     def __iter__(self):
#         return iter(self._data)

#     def __add__(self, arg):
#         self.__init__(self._data + arg._data)
#         return self

#     def _align_id_with_index(self):
#         for i, g in enumerate(self):
#             g.id = i

#     def append(self, *args, **kwargs):
#         if args:
#             if len(args) != 1:
#                 raise ValueError
#             self._data.append(args[0])
#         elif kwargs:
#             self._data.append(self._dtype(**kwargs))
#             self._align_id_with_index()

#     @property
#     def json(self):
#         return json.dumps([v._asdict() for v in self])

#     def __str__(self):
#         BaseList._pretty(self, self._dtype)

#     @staticmethod
#     def _pretty(obj, dtype, indent=0):
#         strings = []
#         if isinstance(obj, dtype):
#             strings += [' ' * indent, type(obj).__name__, '\n']
#             for key, val in obj._asdict().items():
#                 if isinstance(val, (list, tuple)):
#                     strings += [' ' * (indent + 1), str(key), ': ', '\n']
#                     strings += [BaseList._pretty(val, indent + 2)]
#                 else:
#                     strings += [
#                         ' ' * (indent + 1),
#                         str(key), ': ',
#                         str(val), '\n'
#                     ]
#         elif isinstance(obj, BaseList):
#             if len(obj) == 0:
#                 strings += [' ' * indent, '[]', '\n']
#             else:
#                 for val in obj:
#                     strings += [BaseList._pretty(val, indent + 1)]
#             # else:
#             #     strings += [' ' * indent, str(obj), '\n']
#         else:
#             pass
#         return ''.join(strings)
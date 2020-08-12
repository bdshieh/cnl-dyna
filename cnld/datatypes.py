''''''
from namedlist import namedlist, FACTORY
import json
import numpy as np
from collections import MutableMapping, MutableSequence, OrderedDict
from collections.abc import Iterable


class BaseMapping(MutableMapping):
    def __init__(self, *args, **kwargs):
        if args:
            if isinstance(args[0], (dict, BaseMapping)):
                self._data = self._nl(**args[0])
                return
        self._data = self._nl(*args, **kwargs)

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

    def values(self):
        return self._data._asdict().values()

    def items(self):
        return self._data._asdict().items()

    def __contains__(self, key):
        return key in self._data._asdict()

    def __getitem__(self, key):
        if key in self:
            return getattr(self._data, key)
        raise KeyError

    def __setitem__(self, key, val):
        if key in self:
            setattr(self._data, key, val)
        else:
            raise KeyError

    def __delitem__(self, key):
        if key in self:
            setattr(self._data, key, None)
        else:
            raise KeyError

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

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def clear(self):
        for k in self:
            self[k] = None

    def popitem(self):
        return NotImplemented

    def setdefault(self, key, default=None):
        return NotImplemented

    @property
    def json(self):
        return json.loads(json.dumps(self, cls=Encoder))


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, BaseMapping):
            return obj._data._asdict()
        elif isinstance(obj, BaseList):
            return obj._data
        else:
            return super(Encoder, self).default(obj)


def register_mapping(typename, fieldnames):

    t = type(typename, (BaseMapping, ), {})
    t._nl = namedlist(typename, fieldnames)
    MutableMapping.register(t)
    return t


class BaseList(MutableSequence):
    def __init__(self, *args, **kwargs):
        self._data = []
        if args:
            if len(args) != 1:
                raise ValueError
            if isinstance(args[0], int):
                for i in range(args[0]):
                    self._data.append(self._nl(**kwargs))
            elif isinstance(args[0], (list, BaseList)):
                for d in args[0]:
                    self._data.append(self._nl(**d))
            else:
                raise ValueError

        self._align_id_with_index()

    def __repr__(self):
        repr = [
            '<',
            type(self).__name__, '(', self._nl.__name__, ')', ' with ',
            str(len(self)), ' items', '>'
        ]
        return ''.join(repr)

    def __str__(self):
        strings = []
        strings += [type(self).__name__, '\n', '[']
        for i, val in enumerate(self):
            strings += [repr(val)]
            if i != len(self) - 1:
                strings += [',\n ']
        strings += [']', '\n']
        return ''.join(strings)

    def _align_id_with_index(self):
        for i, g in enumerate(self):
            g.id = i

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        if not isinstance(val, self._nl):
            raise ValueError
        self._data[idx] = val
        self._align_id_with_index()

    def __delitem__(self, idx):
        del self._data[idx]
        self._align_id_with_index()

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, idx):
        return idx in self._data

    def __reversed__(self):
        return reversed(self._data)

    def __add__(self, arg):
        self.__init__(self._data + arg._data)
        return self

    def __getattr__(self, attr):
        if attr in self._nl():
            return [getattr(item, attr) for item in self]

    def __setattr__(self, attr, val):
        if attr == '_data':
            super(BaseList, self).__setattr__(attr, val)
        else:
            if attr in self._nl():
                if isinstance(val, Iterable):
                    for item, v in zip(self, val):
                        setattr(item, attr, v)
                else:
                    for item in self:
                        setattr(item, attr, val)

    def append(self, *args, **kwargs):
        if args:
            if len(args) != 1:
                raise ValueError
            self._data.append(self._nl(**args[0]))
            self._align_id_with_index()
        elif kwargs:
            self._data.append(self._nl(**kwargs))
            self._align_id_with_index()

    def insert(self, idx, *args, **kwargs):
        if args:
            if len(args) != 1:
                raise ValueError
            self._data.insert(idx, self._nl(**args[0]))
            self._align_id_with_index()
        elif kwargs:
            self._data.insert(idx, self._nl(**kwargs))
            self._align_id_with_index()

    @property
    def json(self):
        return json.loads(json.dumps(self, cls=Encoder))


def register_list(typename, nl):

    t = type(typename, (BaseList, ), {})
    t._nl = nl
    MutableSequence.register(t)
    return t

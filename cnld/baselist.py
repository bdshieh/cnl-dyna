''''''
from namedlist import namedlist, FACTORY
import json
# from collections import OrderedDict

Data = namedlist('Data', [])


class BaseList(object):
    _dtype = Data

    def __init__(self, data=None):

        if data is None:
            data = []
        elif isinstance(data, self._dtype):
            data = [data]

        if not isinstance(data, (list, BaseList)):
            raise TypeError

        _data = [None] * len(data)
        for v in data:

            if not isinstance(v, self._dtype):
                _v = self._dtype(**v)
            else:
                _v = v

            _data[int(_v.id)] = _v

        self._data = data
        self._align_id_with_index()

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
'''Abstract representation and manipulation of CMUT and PMUT arrays
'''
# __all__ = ['SquareCmutMembrane', 'CircularCmutMembrane', 'Patch', 'Element', 'Array', 'SimulationOptions', 
#            'move_membrane', 'translate_membrane', 'rotate_membrane', 'move_element', 'translate_element',
#            'rotate_element', 'element_position_from_membranes', 'focus_element', 'dump', 'dumps',
#            'bias_element', 'activate_element', 'deactivate_element', 'move_array', 'load', 'loads',
#            'translate_array', 'rotate_array', 'get_element_positions_from_array', 
#            'get_membrane_positions_from_array', 'focus_array', 'get_element_count', 'abstracttype',
#            'classes']

from namedlist import namedlist, FACTORY
from collections import OrderedDict
import sys
import inspect
import json
import numpy as np
import math

from . import util


def _generate_dict_with_name_attr(obj):
    '''Converts abstract object into a nested dictionary with __name__ attribute'''
    name = type(obj).__name__
    # for abstract objects and dicts, add __name__ attr
    if name in names or name is 'dict':
        d = {}
        d['__name__'] = name
        for k, v in obj._asdict().items():
            d[k] = _generate_dict_with_name_attr(v)
        return d
    elif isinstance(obj, (list, tuple)):
        l = []
        for i in obj:
            l.append(_generate_dict_with_name_attr(i))
        return l
    else:
        return obj


def _generate_object_from_json(js):
    '''Convert json object to abstract object'''
    if isinstance(js, dict):
        name = js.pop('__name__')
        d = {}
        for key, val in js.items():
            d[key] = _generate_object_from_json(val)
        # attempt to instantiate abstract object with fallback to dict
        try:
            return ObjectFactory.create(name, **d)
        except KeyError:
            return d
    elif isinstance(js, (list, tuple)):
        l = []
        for i in js:
            l.append(_generate_object_from_json(i))
        return l
    else:
        return js


class ObjectFactory:
    '''Instantiates abstract types defined in module namespace'''
    @staticmethod
    def create(name, *args, **kwargs):
        return globals()[name](*args, **kwargs)


def _repr(self):
    return self.__str__()


def _str(self):
    return pretty(self)


def _contains(self, key):
    return key in self._fields


def dump(obj, fp, indent=1, mode='w+', *args, **kwargs):
    ''''''
    json.dump(_generate_dict_with_name_attr(obj), open(fp, mode), indent=indent, *args, **kwargs)


def dumps(obj, indent=1, *args, **kwargs):
    ''''''
    return json.dumps(_generate_dict_with_name_attr(obj), indent=indent, *args, **kwargs)


def load(fp, *args, **kwargs):
    ''' '''
    return _generate_object_from_json(json.load(open(fp, 'r'), *args, **kwargs))


def loads(s, *args, **kwargs):
    ''''''
    return _generate_object_from_json(json.loads(s, *args, **kwargs))


names = []
def register_type(*args, **kwargs):
    ''''''
    cls = namedlist(*args, **kwargs)
    cls.__repr__ = _repr
    cls.__str__ = _str
    cls.__contains__ = _contains
    names.append(cls.__name__)
    return cls


def pretty(obj, indent=0):
    '''Pretty print abstract objects'''
    strings = []
    if type(obj).__name__ in names:
        strings += [' ' * indent, type(obj).__name__, '\n']
        for key, val in obj._asdict().items():
            if type(val).__name__ in names:
                strings += [' ' * (indent + 1), str(key), ': ', '\n']
                strings += [pretty(val, indent + 1)]
            elif isinstance(val, (list, tuple)):
                strings += [' ' * (indent + 1), str(key), ': ', '\n']
                strings += [pretty(val, indent + 2)]
            else:
                strings += [' ' * (indent + 1), str(key), ': ', str(val), '\n']
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            strings += [' ' * indent , '[]', '\n']
        elif type(obj[0]).__name__ in names:
            for val in obj:
                strings += [pretty(val, indent + 1)]
        elif isinstance(obj[0], (list, tuple)):
            for val in obj:
                strings += [pretty(val, indent + 1)]
        else:
            strings += [' ' * indent, str(obj), '\n']
    else:
        pass
    return ''.join(strings)


_SquareCmutMembrane = OrderedDict()
_SquareCmutMembrane['id'] = None
_SquareCmutMembrane['position'] = None
_SquareCmutMembrane['length_x'] = 40e-6
_SquareCmutMembrane['length_y'] = 40e-6
_SquareCmutMembrane['electrode_x'] = 40e-6
_SquareCmutMembrane['electrode_y'] = 40e-6
_SquareCmutMembrane['thickness'] = (2e-6,)
_SquareCmutMembrane['density'] = (2040,)
_SquareCmutMembrane['y_modulus'] = (110e9,)
_SquareCmutMembrane['p_ratio'] = (0.22,)
_SquareCmutMembrane['isolation'] = 200e-9
_SquareCmutMembrane['permittivity'] = 6.3
_SquareCmutMembrane['gap'] = 100e-9
_SquareCmutMembrane['att_mech'] = 0
_SquareCmutMembrane['npatch_x'] = 3
_SquareCmutMembrane['npatch_y'] = 3
_SquareCmutMembrane['k_matrix_comsol_file'] = None
_SquareCmutMembrane['patches'] = FACTORY(list)
SquareCmutMembrane = register_type('SquareCmutMembrane', _SquareCmutMembrane)

_CircularCmutMembrane = OrderedDict()
_CircularCmutMembrane['id'] = None
_CircularCmutMembrane['position'] = None
_CircularCmutMembrane['radius'] = 20e-6
_CircularCmutMembrane['electrode_r'] = 20e-6 / 2
_CircularCmutMembrane['thickness'] = (2e-6,)
_CircularCmutMembrane['density'] = (2040,)
_CircularCmutMembrane['y_modulus'] = (110e9,)
_CircularCmutMembrane['p_ratio'] = (0.22,)
_CircularCmutMembrane['isolation'] = 200e-9
_CircularCmutMembrane['permittivity'] = 6.3
_CircularCmutMembrane['gap'] = 100e-9
_CircularCmutMembrane['att_mech'] = 0
_CircularCmutMembrane['npatch_r'] = 2
_CircularCmutMembrane['npatch_theta'] = 4
_CircularCmutMembrane['k_matrix_comsol_file'] = None
_CircularCmutMembrane['patches'] = FACTORY(list)
CircularCmutMembrane = register_type('CircularCmutMembrane', _CircularCmutMembrane)

_Patch = OrderedDict()
_Patch['id'] = None
_Patch['position'] = None
_Patch['length_x'] = None
_Patch['length_y'] = None
Patch = register_type('Patch', _Patch)

_Element = OrderedDict()
_Element['id'] = None
_Element['position'] = None
_Element['kind'] = None
_Element['active'] = True
_Element['apodization'] = 1
_Element['delay'] = 0
_Element['dc_bias'] = 0
_Element['membranes'] = FACTORY(list)
Element = register_type('Element', _Element)

_Array = OrderedDict()
_Array['id'] = None
_Array['position'] = None
_Array['delay'] = 0
_Array['elements'] = FACTORY(list)
Array = register_type('Array', _Array)

# _SimulationOptions = OrderedDict()
# _SimulationOptions['fluid_density'] = 1000.
# _SimulationOptions['sound_speed'] = 1500.
# SimulationOptions = register_type('SimulationOptions', _SimulationOptions)

# classes = tuple(name for name, value in inspect.getmembers(sys.modules[__name__], inspect.isclass))


## DECORATORS ##

def vectorize(f):
    '''Allows function to be called with either a single abstract object or list/tuple of them'''
    def decorator(m, *args, **kwargs):

        if isinstance(m, (list, tuple)):
            res = list()
            for i in m:
                res.append(f(i, *args, **kwargs))
            return res
        else:

            return f(m, *args, **kwargs)
    return decorator

# def distance(x, y):
    # return math.sqrt(math.fsum([(i - j) ** 2 for i, j in zip(x, y)]))


## MEMBRANE MANIPLUATIONS ##

@vectorize
def move_membrane(m, pos):
    '''
    '''
    m.position = pos


@vectorize
def translate_membrane(m, vec):
    '''
    '''
    m.position = [i + j for i, j in zip(m.position, vec)]


@vectorize
def rotate_membrane(m, origin, vec, angle):
    '''
    '''
    org = np.array(origin)
    pos = np.array(m.position)

    # determine new membrane position
    newpos = util.rotation_matrix(vec, angle).dot(pos - org) + org
    m.position = newpos.tolist()

    # update membrane rotation list
    if 'rotations' in m:
        m.rotations.append([vec, angle])
    else:
        m.rotations = [[vec, angle]]


## ELEMENT MANIPULATIONS ##

@vectorize
def move_element(e, pos):
    '''
    '''
    vec = [i - j for i, j in zip(pos, e.position)]
    translate_element(e, vec)


@vectorize
def translate_element(e, vec):
    '''
    '''
    e.position = [i + j for i, j in zip(e.position, vec)]

    for m in e.membranes:
        translate_membrane(m, vec)


@vectorize
def rotate_element(e, origin, vec, angle):
    '''
    '''
    org = np.array(origin)
    pos = np.array(e.position)

    # determine new element position
    newpos = util.rotation_matrix(vec, angle).dot(pos - org) + org
    e.position = newpos.tolist()

    # rotate membranes
    for m in e.membranes:
        rotate_membrane(m, origin, vec, angle)


@vectorize
def element_position_from_membranes(e):
    '''
    '''
    membranes = e.membranes

    x = [m.position[0] for m in membranes]
    y = [m.position[1] for m in membranes]
    z = [m.position[2] for m in membranes]

    e.position = [np.mean(x), np.mean(y), np.mean(z)]


@vectorize
def focus_element(e, pos, sound_speed, quantization=None):
    '''
    '''
    d = util.distance(e.position, pos)
    if quantization is None or quantization == 0:
        t = d / sound_speed
    else:
        t = round(d / sound_speed / quantization) * quantization

    e.delay = -t


@vectorize
def defocus_element(e, pos):
    '''
    '''
    raise NotImplementedError


@vectorize
def bias_element(e, bias):
    '''
    '''
    e.dc_bias = bias


@vectorize
def activate_element(e):
    '''
    '''
    e.active = True


@vectorize
def deactivate_element(e):
    '''
    '''
    e.active = False


## ARRAY MANIPLUATIONS ##

@vectorize
def move_array(a, pos):
    '''
    '''
    vec = [i - j for i, j in zip(pos, a.position)]
    translate_array(a, vec)


@vectorize
def translate_array(a, vec):
    '''
    '''
    a.position = [i + j for i, j in zip(a.position, vec)]

    for e in a.elements:
        translate_element(e, vec)


@vectorize
def rotate_array(a, vec, angle, origin=None):
    '''
    '''
    org = np.array(origin)
    pos = np.array(a.position)

    # determine new array position
    newpos = util.rotation_matrix(vec, angle).dot(pos - org) + org
    a.position = newpos.tolist()


    # rotate elements
    for e in a.elements:
        util.rotate_element(e, origin, vec, angle)


@vectorize
def get_element_positions_from_array(a):
    '''
    '''
    return np.array([e.position for e in a.elements])


@vectorize
def get_membrane_positions_from_array(a):
    '''
    '''
    return np.array([m.position for e in a.elements for m in e.membranes])


@vectorize
def focus_array(a, pos, sound_speed, quantization=None, kind=None):
    '''
    '''
    if kind.lower() in ['tx', 'transmit']:
        elements = [e for e in a.elements if e['kind'].lower() in ['tx', 'transmit', 'both', 'txrx']]
    elif kind.lower() in ['rx', 'receive']:
        elements = [e for e in a.elements if e['kind'].lower() in ['rx', 'receive', 'both', 'txrx']]
    elif kind.lower() in ['txrx', 'both'] or kind is None:
        elements = a.elements

    for e in elements:
        focus_element(e, pos, sound_speed, quantization)


@vectorize
def reset_focus_array(a):
    '''
    '''
    for e in a.elements:
        e.delay = 0


@vectorize
def get_element_count(a, kind=None):
    '''
    '''
    if kind is None:
        return len(a.elements)
    elif kind.lower() in ['tx', 'transmit']:
        return len([e for e in a.elements if e['kind'].lower() in ['tx', 'transmit', 'both', 'txrx']])
    elif kind.lower() in ['rx', 'receive']:
        return len([e for e in a.elements if e['kind'].lower() in ['rx', 'receive', 'both', 'txrx']])
    elif kind.lower() in ['txrx', 'both']:
        return len([e for e in a.elements if e['kind'].lower() in ['both', 'txrx']])


@vectorize
def get_membrane_count(a):
    '''
    '''
    return sum([len(e.membranes) for e in a.elements])

@vectorize
def get_patch_count(a):
    '''
    '''
    return sum([len(m.patches) for e in a.elements for m in e.membranes])
    

if __name__ == '__main__':

    pass
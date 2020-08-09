''''''
import numpy as np
from namedlist import namedlist, FACTORY
import json
from collections import OrderedDict
from itertools import cycle
from cnld.baselist import BaseList
from cnld.util import distance

BeamformData = namedlist('BeamformData', OrderedDict(
    id=None,
    apod=1,
    delay=0,
))

WaveformData = namedlist(
    'WaveformData', OrderedDict(
        id=None,
        time=None,
        voltage=None,
        fs=None,
    ))


class Beamforms(BaseList):
    _dtype = BeamformData

    @property
    def id(self):
        return [v.id for v in self]

    @property
    def apod(self):
        return [v.apod for v in self]

    @property
    def delay(self):
        return [v.delay for v in self]

    @apod.setter
    def apod(self, val):
        if isinstance(val, list):
            for bf, v in zip(self, val):
                bf.apod = v
        else:
            raise TypeError

    @delay.setter
    def delay(self, val):
        if isinstance(val, list):
            for bf, v in zip(self, val):
                bf.delay = v
        else:
            raise TypeError


class Waveforms(BaseList):
    _dtype = WaveformData

    @property
    def id(self):
        return [v.id for v in self]

    @property
    def time(self):
        return [v.time for v in self]

    @property
    def voltage(self):
        return [v.voltage for v in self]

    @property
    def fs(self):
        return [v.fs for v in self]


class Transmit(object):
    def __init__(self, beamforms=None, waveforms=None):
        self._beamforms = Beamforms(beamforms)
        self._waveforms = Waveforms(waveforms)

    @classmethod
    def empty(cls, nbeamform, nwaveform):
        return cls(beamforms=[None] * nbeamform, waveforms=[None] * nwaveform)

    @property
    def Beamforms(self):
        return self._beamforms

    @property
    def Waveforms(self):
        return self._waveforms

    @property
    def nbeamform(self):
        return len(self.Beamforms)

    @property
    def nwaveform(self):
        return len(self.Waveforms)

    @Beamforms.setter
    def Beamforms(self, val):
        self._beamforms = Beamforms(val)

    @Waveforms.setter
    def Waveforms(self, val):
        self._waveforms = Waveforms(val)


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

# classes to for the overall population structure

import numpy as np
from sklearn.model_selection import train_test_split
from .functions import chain


class Clade:

    def __init__(self, model, descs=[], size_max=16, len_max=4, train_frac=.5):
        self._descs = descs
        self._gen = 0
        self._size_max = size_max
        self._len_max = len_max
        self._model = model
        self._train_frac = train_frac

    @property
    def alive(self):
        return (self.size > 0)

    @property
    def size(self):
        return (self._size_descs())

    @property
    def fertile(self):
        for desc in self.descs:
            if desc.fertile:
                return true
        return false

    @property
    def descs(self):
        descs = [d for d in self._descs if d.alive]
        return(descs)

    @property
    def clade(self):
        clade = []
        for d in self.descs:
            for i in d.clade:
                clade.append(i)
        return(clade)

    def _size_descs(self, init = 0):
        tally = sum([d.size for d in self.descs])
        return (init + tally)

    def _len_descs(self, init = 0):
        tally = len(self.descs)
        return (init + tally)

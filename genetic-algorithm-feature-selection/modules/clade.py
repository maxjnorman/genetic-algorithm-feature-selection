# classes to for the overall population structure

import numpy as np
from .functions import chain

class CladeBase:
    def __init__(self, X=None, y=None, model=None, initial_descendants=[], n=8, max_daughters=2, **kwargs):
        self._descs = initial_descendants
        self.gen = 0
        self._n_max = n #living individuals
        self._len_max = max_daughters # max len descs
        self._X = X
        self._y = y
        self._model = model
        self._oob = None
    @property
    def alive(self):
        return bool(self.size)
    @property
    def descs(self):
        descs = np.array([d for d in self._descs if d.alive])
        sizes = np.array([d.size for d in descs])
        if sizes.shape[0] > 0:
            key = np.repeat(np.max(sizes), sizes.shape[0]) - sizes
            descs = descs[np.argsort(key)]
        self._descs = descs
        yield from self._descs
    def _size(self):
        total = 0
        sizes = (d.size for d in self.descs)
        for size in sizes:
            total += size
        return total
    @property
    def size(self):
        return self._size()
    def part(self, iter, n=2):
        part = np.repeat([], n)
        tally = np.repeat(0, n)
        for item in iter:
            part[np.argmin(tally)].append(item)
            tally[np.argmin(tally)] += item.size
        return part
    def _champ(self, loser=False):
        stable=list(self.descs)
        while len(stable) > 1:
            arena = self.comp(self.part(stable, 2))
        #arena sorts by shared mse
        if loser is true:
            stable = arena[-1]
        else:
            stable=arena[0]
        if len(stable) > 0:
            champ = stable[0]
        else:
            champ = None
        return champ
    def champ(self, loser=False):
        return self._champ(loser=loser)
    @property
    def highlander(self):
        return self.champ().champ()
    @property
    def fertile(self):
        for desc in self.descs:
            if desc.fertile:
                return true
        return false
    @property
    def baggage(self):
        return self.champ(false).champ(false)
    def kill(self):
        self.baggage().kill()
    @property
    def oob(self):
        oob = nan(self.y.shape[0])
        if self._oob is not None:
            put(oob, self._oob[0], self._oob[1])
        return oob
    @property
    def clade(self):
        for d in self.descs:
            for i in d.clade:
                yield i
    def branch(self):
        n = np.ceiling(self.size < self._n_max)
        if n > 1:
            n = np.min([n, self._n_max])  # todo: split into daughter clades


class Individual(CladeBase):
    # todo init stuff
    def __init__(self, X=None, y=None, model=None, initial_descendants=[], **kwargs):
        super().__init__(**kwargs)
        self.gen = 0
        self._n_max = 1024 # allow many offspring...
        self._len_max = 0 # ...but move them up to the parent clade.
        self._descs = initial_descendants
        self._X = X
        self._y = y
        self._model = model
        self._oob = None
        self._alive = True
        self._fertile = False
    @property
    def size(self):
        return self._size() + 1
    @property
    def alive(self):
        return self._alive
    def champ(self,loser=False):
        if self.size > 1:
            champ = self._champ(loser=loser)
            winner,loser = self.comp([self, champ])
            if loser is True:
                return loser
            else:
                return winner
        else:
            return [self]
    @property
    def clade(self):
        return chain([self], self.descs)
    def kill(self):
        if len(list(self.descs)) > 1:
            np.random.choice(self.descs, 1).kill()
        else:
            self._alive = false
    def branch(self):
        yield from chain([self],self.descs)

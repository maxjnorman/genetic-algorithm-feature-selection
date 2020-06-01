# classes to for the overall population structure

import numpy as np
from .functions import chain

class CladeBase:

    def __init__(self, X=None, y=None, model=None, initial_descendants=[], n=8,
                 max_daughters=2, **kwargs):
        self._descs = initial_descendants
        self.gen = 0
        self._n_max = n # living individuals
        self._X = X
        self._y = y
        self._model = model  # only to be used to build individuals
        self._oob = None

    @property
    def alive(self):
        return (self.size > 0)

    @property
    def n_max(self):
        return self._n_max

    @n_max.setter
    def n_max(self, n_max):
        self._n_max = n_max

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def descs(self):
        descs = np.array([d for d in self._descs if d.alive])
        if len(descs) > 0:
            sizes = np.array([d.size for d in descs])
            key = np.repeat(np.max(sizes), sizes.shape[0]) - sizes
            descs = descs[np.argsort(key)]
        self._descs = descs
        yield from self._descs

    def _size_descs(self):
        total = 0
        sizes = (d.size for d in self.descs)
        for size in sizes:
            total += size
        return total

    def _len_descs(self):
        total = 0
        for d in self.descs:
            total += 1
        return total

    @property
    def size(self):
        return self._size_descs()

    def part(self, iter, n=2):
        part = [list() for _ in range(n)]
        # print("part: {}".format(part))
        tally = np.repeat(0, n)
        # print("tally: {}".format(tally))
        for item in iter:
            # print("item: {}".format(item))
            # print("np.argmin(tally): {}".format(np.argmin(tally)))
            part[np.argmin(tally)].append(item)
            tally[np.argmin(tally)] += item.size
            # print("tally: {}".format(tally))
        return part

    def _champ(self, loser=False):
        stable = list(self.descs)
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


class Clade(CladeBase):

    def __init__(self, X=None, y=None, model=None, initial_descendants=[], n=8,
                 max_daughters=2, **kwargs):
        super().__init__(**kwargs)
        self._descs = initial_descendants
        self.gen = 0
        self.n_max = n # living individuals
        self.len_max = max_daughters # max len descs
        self._X = X
        self._y = y
        self._model = model  # only to be used to build individuals
        self._oob = None

    @property
    def len_max(self):
        return self._len_max

    @len_max.setter
    def len_max(self, len_max):
        assert(isinstance(len_max, int))
        assert(len_max > 1)
        self._len_max = len_max

    def branch(self):
        n = np.ceil(self._len_descs() / self.len_max)
        n = np.min([n, self.len_max])
        n = int(n)
        print("n: {}".format(n))
        if n > 1:
            part_descs = self.part(self.descs)
            print("part_descs: {}".format(part_descs))
            clades = []
            for part_desc in part_descs:
                clades.append(Clade(X=self.X, y=self.y, model=self.model, n=self.n_max, initial_descendants=part_desc))
            self._descs = clades
        for desc in self._descs:
            desc.branch()

    def flatten(self):
        pass


class Individual(CladeBase):

    def __init__(self, X=None, y=None, model=None, initial_descendants=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.gen = 0
        self.n_max = 1024 # allow many offspring...
        self.len_max = 0 # ...but move them up to the parent clade.
        self._descs = initial_descendants
        self._X = X
        self._y = y
        self._model = model
        self._oob = None
        self._alive = True
        self._fertile = False

    @property
    def len_max(self):
        return self._len_max

    @len_max.setter
    def len_max(self, len_max):
        assert(isinstance(len_max, int))
        assert(len_max == 0)
        self._len_max = len_max

    @property
    def size(self):
        return self._size_descs() + 1

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
        if self._len_descs() == 0:
            return None
        else:
            return None
            # return Clade of [self] + list(self.descs)

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

    def branch(self):
        pass

    def breed(self):
        pass

    def collapse(self):
        pass

    def simplify(self):
        pass


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

    def _branch_descs(self):
        for desc in self.descs:
            desc.branch()

    def branch(self):
        n = np.ceil(self._len_descs() / self.len_max)
        n = np.min([n, self.len_max])
        n = int(n)
        if n > 1:
            print("n: {}".format(n))
            part_descs = self.part(self.descs)
            print("part_descs: {}".format(part_descs))
            clades = []
            for part_desc in part_descs:
                clades.append(Clade(X=self.X, y=self.y, model=self.model,
                              n=self.n_max, initial_descendants=part_desc))
            self._descs = clades
        self._branch_descs()

    def collapse(self):
        """
        want to remove clades with single descendants
        """
        print("len(list(self.descs)): {}".format(len(list(self.descs))))
        for desc in self.descs:
            desc.collapse()
            # print("len(list(desc.descs)): {}".format(len(list(desc.descs))))
            if desc._len_descs() == 1:
                self._descs = list(self.descs) + list(desc.descs)
                desc._descs = []

    def simplify(self):
        """
        want to combine descs if len_max allows

        https://stackoverflow.com/
            questions/43313531/python-merging-list-elements-with-a-condition-in-a-bit-tricky-way
        def join_while_too_short(it, length):
            it = iter(it)
            while True:
                current = next(it)
                while len(current) < length:
                    current += ' ' + next(it)
                yield current
        """
        tidy = []
        untidy = []
        # print()
        # print([self])
        for desc in self.descs:
            desc.simplify()
            # print("desc._len_descs(): {}".format(desc._len_descs()))
            # print("desc.len_max: {}".format(desc.len_max))
            if desc._len_descs() < desc.len_max:
                untidy.append(desc)
            else:
                tidy.append(desc)
        # print("tidy: {}".format(tidy))
        print("len(tidy): {}".format(len(tidy)))
        print("len(untidy): {}".format(len(untidy)))
        if len(untidy) == 1:
            tidy = tidy + untidy
            untidy = []
        elif len(untidy) > 1:
            lens = [d._len_descs() for d in untidy]
            # print("lens: {}".format(lens))
            key = np.argsort(lens)
            key = key.max() - key
            # print("key: {}".format(key))
            untidy = np.array(untidy)[key]
            # print("untidy: {}".format(untidy))
            lens = np.array(lens)[key]
            # print("lens: {}".format(lens))
            for i, desc in enumerate(untidy):
                idx = np.arange(0, untidy.shape[0])
                idx_i = idx[idx != i]
                lens_i = lens[idx_i]
                untidy_i = untidy[idx_i]
                while (len(lens_i) > 0) and (np.min(lens_i) + desc._len_descs() < desc.len_max):
                        print("len(lens_i): {}".format(len(lens_i)))
                        print("np.min(lens_i): {}".format(np.min(lens_i)))
                        print("desc._len_descs(): {}".format(desc._len_descs()))
                        print("desc.len_max: {}".format(desc.len_max))
                        print("lens_i: {}".format(lens_i))
                        print("untidy_i: {}".format(untidy_i))
                        print()
                        idx_argmin = np.argmin(lens_i)
                        problem = untidy_i[idx_argmin]
                        desc._descs = list(desc.descs) + list(problem.descs)
                        problem._descs = []
                        # try:
                        untidy_i = np.delete(untidy_i, idx_argmin)
                        lens_i = np.delete(lens_i, idx_argmin)
                        # except:
                        #     untidy_i = []
                        #     lens_i = []



class Individual(CladeBase):

    def __init__(self, X=None, y=None, model=None, **kwargs):
        super().__init__(**kwargs)
        self.gen = 0
        self.n_max = 1024 # allow many offspring...
        self.len_max = 0 # ...but move them up to the parent clade.
        self._descs = []
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
        return int(self.alive)

    @property
    def alive(self):
        return self._alive

    @property
    def clade(self):
        return [self]

    def champ(self, **kwargs):
        return self.clade

    def kill(self):
        self._alive = False

# classes to for the overall population structure

import numpy as np
from .functions import chain

class CladeBase:

    def __init__(self, X=None, y=None, model=None, initial_descendants=[],
                 n_max=8, n_decay_factor=.5, max_daughters=2,
                 train_fraction=.666667, oob_power=2., **kwargs):
        self._descs = initial_descendants
        self.gen = 0
        self._n_max = n_max # living individuals
        self._n_decay_factor = n_decay_factor # how much to reduce subsequent n_max by
        self._X = X
        self._y = y
        self._model = model  # only to be used to build individuals
        self._oob = None
        self._train_fraction = train_fraction
        self._oob_power = oob_power

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
    def n_decay_factor(self):
        return self._n_decay_factor

    @n_decay_factor.setter
    def n_decay_factor(self, n_decay_factor):
        self._n_decay_factor = n_decay_factor

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
        # # print("part: {}".format(part))
        tally = np.repeat(0, n)
        # # print("tally: {}".format(tally))
        for item in iter:
            # # print("item: {}".format(item))
            # # print("np.argmin(tally): {}".format(np.argmin(tally)))
            part[np.argmin(tally)].append(item)
            tally[np.argmin(tally)] += item.size
            # # print("tally: {}".format(tally))
        return part

    def compete(self, iter, loser=False):
        iter = np.random.choice(iter, len(iter), replace=False).tolist()
        teams = self.part(iter, n=2)
        try:
            blue_team = teams[1]
        except:  # TODO: catch specific exception.
            return teams
        red_team = teams[0]
        red_error = np.concatenate([i.oob for i in red_team], axis=0)
        blue_error = np.concatenate([i.oob for i in blue_team], axis=0)
        red_error = np.nanmean(np.power(red_error, 2.), axis=0)
        blue_error = np.nanmean(np.power(blue_error, 2.), axis=0)
        diffs = red_error - blue_error

        if not np.all(np.isnan(diffs)):
            score = np.nanmean(diffs)
        else:
            score = np.random.random() - .5

        if score < 0:
            teams = teams[::-1]

        return teams

    def _champ(self, loser=False):
        stable = list(self.descs)
        if len(stable) > 0:
            while len(stable) > 1:
                # arena = None # TODO:
                # arena sorts by shared mse
                arena = self.compete(stable)
                # # print(arena)
                if loser is True:
                    stable = arena[0]
                else:
                    stable = arena[-1]
            champ = stable[0]
        else:
            champ = None
        return champ

    def champ(self, loser=False):
        return self._champ(loser=loser)

    def highlander(self):
        return self.champ(loser=False).highlander()

    def baggage(self):
        return self.champ(loser=True).baggage()

    @property
    def fertile(self):
        for desc in self.descs:
            if desc.fertile:
                return true
        return false

    def kill(self):
        self.baggage().kill()

    @property
    def clade(self):
        for d in self.descs:
            for i in d.clade:
                yield i

    def branch(self):
        pass

    def collapse(self):
        pass

    def simplify(self):
        pass


class Clade(CladeBase):

    def __init__(self, X=None, y=None, model=None, initial_descendants=[], n=8,
                 max_daughters=2,
                 train_fraction=.666667, oob_power=2., **kwargs):
        super().__init__(**kwargs)
        self._descs = initial_descendants
        self.gen = 0
        self.n_max = n # living individuals
        self.len_max = max_daughters # max len descs
        self._X = X
        self._y = y
        self._model = model  # only to be used to build individuals
        self._oob = None
        self._train_fraction = train_fraction
        self._oob_power = oob_power

    @property
    def _mask(self):
        return [desc._mask for desc in self.descs]

    @property
    def _idx(self):
        return [desc._idx for desc in self.descs]

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

    def _breed_descs(self):
        for desc in self.descs:
            desc.breed()

    def breed(self):
        self._breed_descs()

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None, replace=True):
        self._X = X
        self._y = y
        for desc in self.descs:
            desc.fit(X, y, sample_weight, check_input, X_idx_sorted, replace=replace)

    def predict(self, X, check_input=True):
        stack = []
        concat = []
        for desc in self.descs:
            prediction = desc.predict(X, check_input=True)
            if prediction is not None:
                if len(prediction.shape) == 1:
                    stack.append(prediction)
                elif len(prediction.shape) == 2:
                    concat.append(prediction)
                else:
                    pass
        predictions = []
        try:
            predictions.append(np.stack(stack, axis=0))
        except:
            pass
        try:
            predictions.append(np.concatenate(concat, axis=0))
        except:
            pass

        if len(predictions) > 0:
            return np.concatenate(predictions, axis=0)

    def predict_proba(self, X, check_input=True):
        stack = []
        concat = []
        for desc in self.descs:
            prediction = desc.predict_proba(X, check_input=True)
            if prediction is not None:
                if len(prediction.shape) == 2:
                    stack.append(prediction)
                elif len(prediction.shape) == 3:
                    concat.append(prediction)
                else:
                    pass
        predictions = []
        try:
            predictions.append(np.stack(stack, axis=0))
        except:
            pass
        try:
            predictions.append(np.concatenate(concat, axis=0))
        except:
            pass

        if len(predictions) > 0:
            return np.concatenate(predictions, axis=0)

    @property
    def oob(self):
        stack = []
        concat = []
        for desc in self.descs:
            oob = desc.oob
            if oob is not None:
                if len(oob.shape) == 1:
                    stack.append(oob)
                elif len(oob.shape) == 2:
                    concat.append(oob)
                else:
                    pass
        oobs = []
        try:
            oobs.append(np.stack(stack, axis=0))
        except: # TODO: catch specific exception
            pass
        try:
            oobs.append(np.concatenate(concat, axis=0))
        except: # TODO: catch specific exception
            pass
        if len(oobs) > 0:
            return np.concatenate(oobs, axis=0)
        else:
            return None

    @property
    def error(self):
        stack = []
        concat = []
        for desc in self.descs:
            error = desc.error
            if error is not None:
                if len(error.shape) == 1:
                    stack.append(error)
                elif len(error.shape) == 2:
                    concat.append(error)
                else:
                    pass
        errors = []
        try:
            errors.append(np.stack(stack, axis=0))
        except: # TODO: catch specific exception
            pass
        try:
            errors.append(np.concatenate(concat, axis=0))
        except: # TODO: catch specific exception
            pass
        if len(errors) > 0:
            return np.concatenate(errors, axis=0)
        else:
            return None

    @property
    def genotype(self):
        stack = []
        concat = []
        for desc in self.descs:
            genotype = desc.genotype
            if genotype is not None:
                if len(genotype.shape) == 1:
                    stack.append(genotype)
                elif len(genotype.shape) == 2:
                    concat.append(genotype)
                else:
                    pass
        genotypes = []
        try:
            genotypes.append(np.stack(stack, axis=0))
        except: # TODO: catch specific exception
            pass
        try:
            genotypes.append(np.concatenate(concat, axis=0))
        except: # TODO: catch specific exception
            pass
        if len(genotypes) > 0:
            return np.concatenate(genotypes, axis=0)
        else:
            return None


    def branch(self):
        n = np.ceil(self._len_descs() / self.len_max)
        n = np.min([n, self.len_max])
        n = int(n)
        if n > 1:
            part_descs = self.part(self.descs)
            clades = []
            for part_desc in part_descs:
                clades.append(Clade(X=self.X, y=self.y, model=self.model,
                              n_max=self.n_max * self.n_decay_factor,
                              initial_descendants=part_desc,
                              n_decay_factor=self.n_decay_factor,
                              max_daughters=self._len_max))
            self._descs = clades
        self._branch_descs()

    def collapse(self):
        """
        want to remove clades with single descendants
        """
        if self._len_descs() == 1:
            if list(self.descs)[0]._len_descs() > 0:  # it is not an Individual
                desc = list(self.descs)[0]  # the descendant object
                self._descs = list(desc.descs)

        for desc in self.descs:
            desc.collapse()
            if desc._len_descs() == 1:
                self._descs = list(self.descs) + list(desc.descs)
                desc._descs = [] # this kills the clade

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
        for desc in self.descs:
            desc.simplify()
            if desc._len_descs() < desc.len_max:
                untidy.append(desc)
            else:
                tidy.append(desc)
        if len(untidy) == 1:
            tidy = tidy + untidy
            untidy = []
        elif len(untidy) > 1:
            lens = [d._len_descs() for d in untidy]
            key = np.argsort(lens)
            key = key.max() - key
            untidy = np.array(untidy)[key]
            lens = np.array(lens)[key]
            for i, desc in enumerate(untidy):
                idx = np.arange(0, untidy.shape[0])
                idx_i = idx[idx != i]
                lens_i = lens[idx_i]
                untidy_i = untidy[idx_i]
                while (len(lens_i) > 0) and (np.min(lens_i) + desc._len_descs() < desc.len_max):
                    gap = desc.len_max - desc._len_descs()
                    idx_transfer = np.argmax(lens_i[lens_i <= gap])
                    # idx_transfer = np.argmin(lens_i)
                    problem = untidy_i[idx_transfer]
                    desc._descs = list(desc.descs) + list(problem.descs)
                    problem._descs = []
                    untidy_i = np.delete(untidy_i, idx_transfer)
                    lens_i = np.delete(lens_i, idx_transfer)



class Individual(CladeBase):

    def __init__(self, genes, model, expression_function, X=None, y=None,
                 train_fraction=.666667, oob_power=2., **kwargs):
        super().__init__(**kwargs)
        self.gen = 0
        self.n_max = 0 # allow many offspring...
        self._n_decay_factor = .0 # how much to reduce subsequent n_max by
        self.len_max = 0 # ...but move them up to the parent clade.
        self._descs = []
        self._X = X
        self._y = y
        self._model = model
        self._oob = None
        self._oob_score = None
        self._oob_truth = None
        self._mask = None
        self._idx = None
        self._alive = True
        self._fertile = False
        self._train_fraction = train_fraction
        self._oob_power = oob_power
        self._genes = genes
        self._expression_function = expression_function

    @property
    def phenotype(self):
        return self._expression_function(self._genes)

    @property
    def genotype(self):
        return self._genes.reshape((1,-1))

    @genotype.setter
    def genotype(self, genes):
        assert(len(genes.shape) == 1)
        self._genes = genes

    def breed(self):
        pass

    @property
    def train_fraction(self):
        return self._train_fraction

    @train_fraction.setter
    def train_fraction(self, train_fraction):
        assert(isinstance(train_fraction, float))
        # assert(train_fraction <= 1.)
        assert(train_fraction > 0.)
        self._train_fraction = train_fraction

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

    @property
    def error(self):
        if self._oob is None:
            return None
        else:
            oob = np.repeat(np.nan, self._mask.shape[0])
            np.put(oob, self._idx, self._oob)
            oob = np.abs(oob)
            oob = np.power(oob, self._oob_power)
            return oob

    @property
    def oob(self):
        if self._oob is None:
            return None
        else:
            mask = np.logical_not(self._mask)
            oob = np.repeat(np.nan, mask.shape[0])
            idx = np.where(mask)[0]
            np.put(oob, idx, self._oob_score)
            return oob.reshape((1,-1))


    def champ(self, **kwargs):
        return self

    def highlander(self, **kwargs):
        return self

    def baggage(self, **kwargs):
        return self

    def kill(self):
        self._alive = False

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None, replace=True):
        n = y.shape[0]
        # TODO: split into train and test and bootstrap
        self._idx = np.random.choice(
            np.arange(n),
            int(np.ceil(n * self.train_fraction)),
            replace=replace
        )
        mask = np.repeat(False, n)
        np.put(mask, np.unique(self._idx), True)
        self._mask = mask  # mask for rows, not cols
        # TODO: apply genes to X
        self._X = X
        self._y = y
        # TODO: apply genes to model
        self._model.set_params()
        self._model.fit(self._X[self._mask], self._y[self._mask],
                        sample_weight, check_input, X_idx_sorted)
        self._oob_score = self.predict_proba(self._X[np.logical_not(self._mask)])[:,1]
        self._oob_truth = self._y[np.logical_not(self._mask)]
        self._oob = self._oob_score - self._oob_truth

    def predict(self, X, check_input=True):
        predictions = self._model.predict(X, check_input=True)
        return predictions

    def predict_proba(self, X, check_input=True):
        predictions = self._model.predict_proba(X, check_input=True)
        return predictions

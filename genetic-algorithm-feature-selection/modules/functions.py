import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG)


def chain_safe(*iterables):
  for iterable in iterables:
    yield from iterable


def chain(*iterables):
    for iterable in iterables:
        try:
            yield from iterable
        except TypeError as e:
            print(e)


def express_vals(genes, cuts, labels):
    max = np.max(cuts)
    excess = genes[max:]
    genes = genes[0:max]
    vals = np.array([
        int("".join(arr.astype(str)), 2)
        for arr
        in np.split(genes, cuts)
        if arr.shape[0] > 0
        ])
    assert(len(vals) == len(labels))
    vals = dict(zip(labels, vals))
    return vals, excess


def sample_dict(params, selected):
    keys = []
    vals = []
    for key in selected.keys():
        try:
            val = params[key][selected[key]]
        except KeyError:
            val = selected[key]
        finally:
            keys.append(key)
            vals.append(val)
    return dict(zip(keys, vals))


def express_factory(cuts, labels, params):

    def express(genes):
        vals, excess = express_vals(genes, cuts, labels)
        vals = sample_dict(params, vals)
        return vals, excess

    return express

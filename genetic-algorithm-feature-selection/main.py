import numpy as np
import pandas as pd

from modules.functions import chain
from modules.clade import Clade, Individual

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

max_daughters = 8
train_fraction = .95
replace = True

# clade = Clade(
#     y="A",
#     max_daughters=max_daughters,
#     initial_descendants=[

#         Individual(y=0),

#         Individual(y=1),

#         Clade(
#             y="B",
#             max_daughters=max_daughters,
#             initial_descendants=[

#                 Individual(y=2),

#                 Individual(y=3),

#                 Individual(y=4)
#             ]),

#         Clade(
#             y="C",
#             max_daughters=max_daughters,
#             initial_descendants=[

#                 Individual(y=5),

#                 Clade(
#                     y="D",
#                     max_daughters=max_daughters,
#                     initial_descendants=[

#                         Individual(y=6),

#                         Clade(
#                             y="E",
#                             max_daughters=max_daughters,
#                             initial_descendants=[

#                                 Individual(y=7),

#                                 Individual(y=8),

#                                 Individual(y=9)

#                             ])
#                     ])
#             ])

#     ])


dat = pd.read_csv("/Users/maxjnorman/OneDrive - HEOR Ltd/Projects/Main/HEOR00324 - ML INR/dataset_20/baseline_20_simple.csv")
colnames = dat.columns[dat.columns != "ttr_out_ros"]
y = (dat["ttr_out_ros"] > dat["ttr_out_ros"].mean()).astype(int).values
X = dat.loc[:,colnames].values

splits = np.concatenate((
    np.arange(1, colnames.shape[0]),
    colnames.shape[0] + np.array([1,2,7])
    ))

def express_hyps(vals):
    assert(len(vals) == 2)
    criterion = ("gini", "entropy")[vals[0]]
    max_depth = vals[1] + 1
    return {"criterion":criterion, "max_depth":max_depth}

# def express(genes, colnames=colnames, splits=splits):
#     vals = express_values(genes, splits)
#     mask = vals[np.arange(0, colnames.shape[0])]
#     hyps = vals[np.arange(colnames.shape[0], vals.shape[0])]
#     return {"mask":mask.astype(bool), "hyps":express_hyps(hyps)}

def express_factory(colnames, splits, express_hyps):

    def express_values(genes, splits=np.array([])):
        genes = genes.round().astype(bool).astype(int).astype(str)
        bin_arrs = np.split(genes, np.unique(splits))  # binary arrays
        ints = np.array([
            int("".join(gene), 2)
            for gene
            in bin_arrs
            if gene.shape[0] > 0
            ])
        return ints

    def express(genes):
        vals = express_values(genes, splits)
        mask = vals[np.arange(0, colnames.shape[0])]
        hyps = vals[np.arange(colnames.shape[0], vals.shape[0])]
        return {"mask":mask.astype(bool), "hyps":express_hyps(hyps)}

    return express


genes = np.random.random(splits.max()).round().astype(bool).astype(int)
express = express_factory(colnames, splits, express_hyps)
phenotype = express(genes)
mask = phenotype["mask"]
hyps = phenotype["hyps"]

tree = DecisionTreeClassifier
root = Clade(
    initial_descendants=[
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)),
        Individual(model=tree(), train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int))],
    n_max=128,
    max_daughters=max_daughters,
    train_fraction=train_fraction, expression_function=express, genes = np.random.random(splits.max()).round().astype(bool).astype(int)
    )

root.fit(X,y,replace=replace)
root.branch()
root.simplify()
root.collapse()
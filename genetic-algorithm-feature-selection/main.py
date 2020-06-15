import numpy as np
import pandas as pd

from modules.functions import chain
from modules.clade import Clade, Individual

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

max_daughters = 8

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
tree = DecisionTreeClassifier
root = Clade(
    initial_descendants=[
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),
        Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3)),Individual(model=tree(max_depth=3))],
    n_max=128, max_daughters=max_daughters, train_fraction=.75
    )
# vine = Clade(
#     initial_descendants=[Clade(
#         initial_descendants=[Clade(
#             initial_descendants=[Individual(model=tree(max_depth=3))]
#             )]
#         )]
#     )


dat = pd.read_csv("/Users/maxjnorman/OneDrive - HEOR Ltd/Projects/Main/HEOR00324 - ML INR/dataset_20/baseline_20_simple.csv")
y = (dat["ttr_out_ros"] > 0.3).astype(int).values
X = dat.loc[:,dat.columns != "ttr_out_ros"].values

root.fit(X,y)



train_fraction = .75
_model = tree(max_depth=3)

n = y.shape[0]
_idx = np.random.choice(
            np.arange(n),
            int(np.ceil(n * train_fraction)),
            replace=False
        )
mask = np.repeat(False, n)
np.put(mask, _idx, True)
_mask = mask
_X = X
_y = y
_model.fit(_X[_mask], _y[_mask])
_oob_score = _model.predict_proba(_X[np.logical_not(_mask)])[:,1]
_oob_truth = _y[np.logical_not(_mask)]
_oob = _oob_score - _oob_truth



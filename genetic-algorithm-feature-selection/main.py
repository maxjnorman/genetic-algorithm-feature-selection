import numpy as np
import pandas as pd

from modules.functions import chain
from modules.clade import Clade, Individual

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

max_daughters = 8
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

dat = pd.read_csv(
    "/home/max/projects/toy/ga-feat-selection/data/splice/Source/splice.data",
    header=None
)
y = dat[0].str.match("^EI$").astype(int).values
X = dat[2].str.strip().str.split('', expand=True).values
X = OneHotEncoder().fit_transform(X)
root.fit(X,y)

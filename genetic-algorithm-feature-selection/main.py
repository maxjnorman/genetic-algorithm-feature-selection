import numpy as np
import pandas as pd

from modules.functions import chain, express_factory
from modules.clade import Clade, Individual

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder



dat = pd.read_csv(
    "/home/max/projects/toy/ga-feat-selection/data/splice/Source/splice.data",
    header=None
)
y = dat[0].str.match("^EI$").astype(int).values
X = dat[2].str.strip().str.split('', expand=True).values
X = OneHotEncoder().fit_transform(X)

max_daughters = 8
tree = DecisionTreeClassifier
express_meta = express_factory(
    params = {
        "mutation_rate": np.arange(2., 34., 1.) / 34.
    },
    labels = ("mutation_rate",),
    cuts = (5,)
)
express_params = express_factory(
    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": np.arange(1, 65)
    },
    labels = ("criterion", "max_depth"),
    cuts = (1,6)
)
def express(genes):
    meta, genes = express_meta(genes)
    hyps, genes = express_params(genes)
    phenotype = {
        "meta": meta,
        "hyps": hyps,
        "mask": genes.astype(bool)
    }
    return phenotype

print("EXPRESS")
print(
    express(
        np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int)
    )
)

root = Clade(
    initial_descendants=[
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express),
        Individual(model=tree(), genes=np.random.random(5 + 1 + 5 + X.shape[1]).round().astype(int), expression_function=express)],
        n_max=128,
        max_daughters=max_daughters,
        train_fraction=.75
    )
root.fit(X,y)

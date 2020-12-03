import numpy as np
import pandas as pd
import warnings

from modules.functions import chain, express_factory
from modules.clade import Clade, Individual

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# dat = pd.read_csv(
#     # "/home/max/projects/toy/ga-feat-selection/data/splice/Source/splice.data",
#     "data/diabetes.csv",
#     header=0
# )
# rands = np.round(np.random.random(dat.shape) * .2, 1) * 5.
# rands = pd.DataFrame(rands)
# dat = pd.concat((dat, rands), axis=1)
# rands = np.round(np.random.random(dat.shape) * .2, 1) * 5.
# rands = pd.DataFrame(rands)
# dat = pd.concat((dat, rands), axis=1)
# train, test = train_test_split(dat, test_size=0.25)
# encoder = OneHotEncoder()
# encoder.fit(dat.drop('class', axis=1).values)
# y = train['class'].str.match("^Positive$").astype(int).values
# X = train.drop('class', axis=1).values
# X = encoder.transform(X)
# y_test = test['class'].str.match("^Positive$").astype(int).values
# X_test = test.drop('class', axis=1).values
# X_test = encoder.transform(X_test)

dat = pd.read_csv("/Users/maxjnorman/Desktop/baseline.csv")
tst = pd.read_csv("/Users/maxjnorman/Desktop/testset.csv")
cols = dat.columns
cols = cols[cols.str.startswith("flag")] 

X = dat[cols[cols.str.startswith("flag") | cols.str.startswith("value")]].values
X_test = tst[cols[cols.str.startswith("flag") | cols.str.startswith("value")]].values
y = dat["ttr_out_ros_good_control"].values
y_test = tst["ttr_out_ros_good_control"].values

max_daughters = 32
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
root.branch()
root.simplify()
root.collapse()
counter = 0

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    while counter < 96:
        counter += 1
        # root.fit(X,y)
        while root._size_descs() <= (128 + counter):
            root.fit(X,y)
            root.breed(X, y)
            root.branch()
        root.fit(X,y)
        while root._size_descs() > (96 + counter):
            # root.fit(X,y)
            root.kill()
            root.simplify()
            root.collapse()
            a = np.nanmean(np.nanmean(root.oob, axis=0)).round(3)
            b = np.round(roc_auc_score(y_test, np.nanmean(root.predict_proba(X_test)[:,:,1], axis=0)), 3)
            print(f"oob:{a} auc:{b}")
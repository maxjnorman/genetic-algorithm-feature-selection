import numpy as np
import pandas as pd

from modules.functions import chain
from modules.clade import Clade, Individual

from sklearn.tree import DecisionTreeClassifier

max_daughters = 8

clade = Clade(
    y="A",
    max_daughters=max_daughters,
    initial_descendants=[

        Individual(y=0),

        Individual(y=1),

        Clade(
            y="B",
            max_daughters=max_daughters,
            initial_descendants=[

                Individual(y=2),

                Individual(y=3),

                Individual(y=4)
            ]),

        Clade(
            y="C",
            max_daughters=max_daughters,
            initial_descendants=[

                Individual(y=5),

                Clade(
                    y="D",
                    max_daughters=max_daughters,
                    initial_descendants=[

                        Individual(y=6),

                        Clade(
                            y="E",
                            max_daughters=max_daughters,
                            initial_descendants=[

                                Individual(y=7),

                                Individual(y=8),

                                Individual(y=9)

                            ])
                    ])
            ])

    ])
tree = DecisionTreeClassifier
root = Clade(
    initial_descendants=[
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree()),
        Individual(model=tree()),Individual(model=tree()),Individual(model=tree())],
    n_max=8, max_daughters=max_daughters
    )
vine = Clade(
    initial_descendants=[Clade(
        initial_descendants=[Clade(
            initial_descendants=[Individual(model=tree())]
            )]
        )]
    )
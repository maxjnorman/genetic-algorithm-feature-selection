import numpy as np
import pandas as pd

from modules.functions import chain
from modules.clade import CladeBase, Individual

clade = CladeBase(initial_descendants=[
    Individual(y=0),
    CladeBase(initial_descendants=[
        Individual(y=1),
        Individual(y=2)
        ]),
    CladeBase(initial_descendants=[
        Individual(y=3),
        CladeBase(initial_descendants=[
            Individual(y=4),
            CladeBase(initial_descendants=[
                Individual(y=5,
                           initial_descendants=[
                    Individual(y=6)
                    ]),
                Individual(y=7)
                ])
            ])
        ])
    ])
for desc in clade.descs:
    print(desc)
print()

for indiv in clade.clade:
    print(indiv._y, indiv)
print()
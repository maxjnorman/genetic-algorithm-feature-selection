import numpy as np
import pandas as pd

from modules.functions import chain
from modules.clade import Clade, Individual

root = Clade(initial_descendants=[Individual(),Individual(),Individual(),
                                  Individual(), Individual()],
             n_max=8, max_daughters=2)
vine = Clade(initial_descendants=[
    Clade(initial_descendants=[
        Clade(initial_descendants=[
            Individual()
            ])])])
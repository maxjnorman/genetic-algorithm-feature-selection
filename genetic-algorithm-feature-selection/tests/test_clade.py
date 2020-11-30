import unittest
import logging
import numpy as np
# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from modules.clade import Clade, Individual
        # from modules.functions import express_factory
    else:
        from ..modules.clade import Clade, Individual
        # from ..modules.functions import express_factory

logging.basicConfig(level=logging.DEBUG)

# express_meta = express_factory(
#     params = {
#         "mutation_rate": np.arange(2., 34., 1.) / 34.
#     },
#     labels = ("mutation_rate",),
#     cuts = (5,)
# )
# express_params = express_factory(
#     params = {
#         "criterion": ["gini", "entropy"],
#         "max_depth": np.arange(1, 65)
#     },
#     labels = ("criterion", "max_depth"),
#     cuts = (1,6)
# )
# def express(genes):
#     meta, genes = express_meta(genes)
#     hyps, genes = express_params(genes)
#     phenotype = {
#         "meta": meta,
#         "hyps": hyps,
#         "mask": genes.astype(bool)
#     }
#     return phenotype


class TestCladeStructureMethods (unittest.TestCase):

    def test_clade8(self):
        """
        Test that Clade class clade method returns all individuals
        """
        clade = Clade(
            initial_descendants=[
                Individual(y=0, model=None, genes=None,
                           expression_function=None),
                Clade(
                    initial_descendants=[
                        Individual(y=1, model=None, genes=None,
                                   expression_function=None),
                        Individual(y=2, model=None, genes=None,
                                   expression_function=None)
                        ]
                    ),
            Clade(initial_descendants=[
                Individual(y=3, model=None, genes=None,
                           expression_function=None),
                Clade(initial_descendants=[
                    Individual(y=4, model=None, genes=None,
                               expression_function=None),
                    Clade(initial_descendants=[
                        Individual(y=5, model=None, genes=None,
                                   expression_function=None),
                        Individual(y=6, model=None, genes=None,
                                   expression_function=None),
                        Individual(y=7, model=None, genes=None,
                                   expression_function=None)
                    ])
                ])
            ])
        ])
        check = np.array([i._y for i in clade.clade])
        np.testing.assert_array_equal(check[np.argsort(check)], np.arange(8))

    def test_collapse11(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Individual(y="A", model=None, genes=None, expression_function=None)
            ])
        ])
        clade.collapse()
        self.assertEqual(list(clade.descs)[0].y, "A")

    def test_collapse21(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Clade(initial_descendants=[
                    Individual(y="A", model=None, genes=None, expression_function=None)
                ])
            ])
        ])
        clade.collapse()
        self.assertEqual(list(clade.descs)[0].y, "A")

    def test_collapse31(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Clade(initial_descendants=[
                    Clade(initial_descendants=[
                        Individual(y="A", model=None, genes=None, expression_function=None)
                    ])
                ])
            ])
        ])
        clade.collapse()
        self.assertEqual(list(clade.descs)[0].y, "A")

    def test_collapse32(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Clade(y="C", initial_descendants=[
                    Clade(
                        initial_descendants=[
                            Individual(y="A", model=None, genes=None, expression_function=None),
                            Individual(y="B", model=None, genes=None, expression_function=None)
                        ]
                    )
                ])
            ])
        ])
        clade.collapse()
        checkC = list([i.y for i in clade.descs])
        checkAB = list([i.y for i in list(clade.descs)[0].descs])
        np.testing.assert_array_equal(checkAB + checkC, ["A", "B", "C"])

    def test_collapse121(self):
        clade = Clade(initial_descendants=[
            Individual(y="A", model=None, genes=None, expression_function=None),
            Clade(initial_descendants=[
                Individual(y="B", model=None, genes=None, expression_function=None)
            ])
        ])
        clade.collapse()
        check = [d.y for d in list(clade.descs)]
        np.testing.assert_array_equal(check, ["A", "B"])

    def test_branch13_descs(self):
        clade = Clade(y="A", initial_descendants=[
            Individual(y=1, model=None, genes=None, expression_function=None),
            Individual(y=2, model=None, genes=None, expression_function=None),
            Individual(y=3, model=None, genes=None, expression_function=None)
            ])
        clade.branch()
        check = list([d.y for d in clade.descs])
        np.testing.assert_array_equal(check, ["A", "A"])

    def test_branch13_branch(self):
        clade = Clade(y="A", initial_descendants=[
            Individual(y=1, model=None, genes=None, expression_function=None),
            Individual(y=2, model=None, genes=None, expression_function=None),
            Individual(y=3, model=None, genes=None, expression_function=None)
            ])
        clade.branch()
        check = np.array([d.y for d in clade.clade])
        check = check[np.argsort(check)]
        np.testing.assert_array_equal(check, np.array([1, 2, 3]))

if __name__ == '__main__':
    unittest.main()

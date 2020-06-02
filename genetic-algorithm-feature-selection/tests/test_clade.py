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
    else:
        from ..modules.clade import Clade, Individual

logging.basicConfig(level=logging.DEBUG)


class TestCladeStructureMethods (unittest.TestCase):

    def test_clade8(self):
        """
        Test that Clade class clade method returns all individuals
        """
        clade = Clade(initial_descendants=[
            Individual(y=0),
            Clade(initial_descendants=[
                Individual(y=1),
                Individual(y=2)
            ]),
            Clade(initial_descendants=[
                Individual(y=3),
                Clade(initial_descendants=[
                    Individual(y=4),
                    Clade(initial_descendants=[
                        Individual(y=5),
                        Individual(y=6),
                        Individual(y=7)
                    ])
                ])
            ])
        ])
        check = np.array([i._y for i in clade.clade])
        np.testing.assert_array_equal(check[np.argsort(check)], np.arange(8))

    def test_collapse11(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Individual(y="A")
            ])
        ])
        clade.collapse()
        self.assertEqual(list(clade.descs)[0].y, "A")

    def test_collapse21(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Clade(initial_descendants=[
                    Individual(y="A")
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
                        Individual(y="A")
                    ])
                ])
            ])
        ])
        clade.collapse()
        self.assertEqual(list(clade.descs)[0].y, "A")

    def test_collapse32(self):
        clade = Clade(initial_descendants=[
            Clade(initial_descendants=[
                Clade(initial_descendants=[
                    Clade(
                        y="C",
                        initial_descendants=[
                            Individual(y="A"),
                            Individual(y="B")
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
            Individual(y="A"),
            Clade(initial_descendants=[
                Individual(y="B")
            ])
        ])
        clade.collapse()
        check = [d.y for d in list(clade.descs)]
        np.testing.assert_array_equal(check, ["A", "B"])

if __name__ == '__main__':
    unittest.main()

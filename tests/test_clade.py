import unittest
import logging

if __name__ == '__main__':
    # https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from modules.clade import CladeBase, Individual
    else:
        from ..modules.clade import CladeBase, Individual

logging.basicConfig(level=logging.DEBUG)


class TestModulesCladeClade(unittest.TestCase):

    def test_clade_method_clade(self):
        """
        Test that Clade class clade method returns correctly
        """
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

if __name__ == '__main__':
    unittest.main()

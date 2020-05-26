import unittest
# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from modules.functions import chain
    else:
        from ..modules.functions import chain


class TestModulesFunctionsChain(unittest.TestCase):

    def test_two_gens_list_range(self):
        """
        Test that it can chain two generators
        """
        gens = [
            (x for x in range(0, 6)),
            (x for x in range(6, 12))
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 12)))

    def test_three_gens_list_range(self):
        """
        Test that it can chain two generators
        """
        gens = [
            (x for x in range(0, 6)),
            (x for x in range(6, 12)),
            (x for x in range(12, 18))
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 18)))

    def test_two_lists_list_range(self):
        """
        Test that it can chain two generators
        """
        gens = [
            [x for x in range(0, 6)],
            [x for x in range(6, 12)]
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 12)))

    def test_three_lists_list_range(self):
        """
        Test that it can chain two generators
        """
        gens = [
            [x for x in range(0, 6)],
            [x for x in range(6, 12)],
            [x for x in range(12, 18)]
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 18)))

if __name__ == '__main__':
    unittest.main()

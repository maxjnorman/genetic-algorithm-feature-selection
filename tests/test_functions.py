import unittest
import logging
# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from modules.functions import chain
    else:
        from ..modules.functions import chain

logging.basicConfig(level=logging.DEBUG)


class TestModulesFunctionsChain(unittest.TestCase):

    def test_one_gen_range(self):
        """
        Test that it can chain a single list
        """
        gens = [
            (x for x in range(0, 6))
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 6)))

    def test_two_gen_range(self):
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

    def test_three_gen_range(self):
        """
        Test that it can chain three generators
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

    def test_one_list_range(self):
        """
        Test that it can chain a single list
        """
        gens = [
            [x for x in range(0, 6)]
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 6)))

    def test_two_list_range(self):
        """
        Test that it can chain two lists
        """
        gens = [
            [x for x in range(0, 6)],
            [x for x in range(6, 12)]
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 12)))

    def test_three_list_range(self):
        """
        Test that it can chain three lists
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

    def test_one_listgen_range(self):
        """
        Test that it can chain a list and a generator
        """
        gens = [
            [x for x in range(0, 6)],
            (x for x in range(6, 12))
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 12)))

    def test_three_listgen_range(self):
        """
        Test that it can chain alternating lists and generators
        """
        gens = [
            [x for x in range(0, 6)],
            (x for x in range(6, 12)),
            [x for x in range(12, 18)],
            (x for x in range(18, 24)),
            [x for x in range(24, 30)],
            (x for x in range(30, 36))
        ]
        result = []
        for item in chain(*gens):
            result.append(item)
        self.assertEqual(result, list(range(0, 36)))

if __name__ == '__main__':
    unittest.main()

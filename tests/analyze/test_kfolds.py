import unittest

from src.analyze import k_folds


class KFoldsTets(unittest.TestCase):
    def test_configure_k_folds(self):
        groups = k_folds.configure_k_folds()
        print(groups)

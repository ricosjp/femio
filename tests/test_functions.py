import unittest

import numpy as np

import femio.functions as functions


class TestFunctions(unittest.TestCase):

    def test_normalize(self):
        array = np.random.rand(10, 3)
        normed = functions.normalize(array)
        for n in normed:
            np.testing.assert_almost_equal(np.linalg.norm(n), 1.)

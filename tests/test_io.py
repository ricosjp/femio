import unittest

import numpy as np

from femio.fem_data import FEMData
from femio import io


FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestIO(unittest.TestCase):

    def test_io_read_files(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        io_fem_data = io.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        np.testing.assert_almost_equal(
            io_fem_data.nodes.data, fem_data.nodes.data)
        np.testing.assert_array_equal(
            io_fem_data.elements.data, fem_data.elements.data)
        for key in fem_data.nodal_data.keys():
            np.testing.assert_almost_equal(
                io_fem_data.nodal_data.get_attribute_data(key),
                fem_data.nodal_data.get_attribute_data(key))
        for key in fem_data.elemental_data.keys():
            np.testing.assert_almost_equal(
                io_fem_data.elemental_data.get_attribute_data(key),
                fem_data.elemental_data.get_attribute_data(key))

    def test_io_read_directory(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False, save=False)
        io_fem_data = io.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            io_fem_data.nodes.data, fem_data.nodes.data)
        np.testing.assert_array_equal(
            io_fem_data.elements.data, fem_data.elements.data)
        for key in fem_data.nodal_data.keys():
            np.testing.assert_almost_equal(
                io_fem_data.nodal_data.get_attribute_data(key),
                fem_data.nodal_data.get_attribute_data(key))
        for key in fem_data.elemental_data.keys():
            np.testing.assert_almost_equal(
                io_fem_data.elemental_data.get_attribute_data(key),
                fem_data.elemental_data.get_attribute_data(key))

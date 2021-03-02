"""
FEM I/O tool
"""

from .fem_data import FEMData  # NOQA
from .fem_attribute import FEMAttribute  # NOQA
from .fem_attributes import FEMAttributes  # NOQA
from .fem_elemental_attribute import FEMElementalAttribute  # NOQA
from .io import read_directory, read_files  # NOQA
from .util.brick_generator import generate_brick  # NOQA
from .util.random_generator import generate_random_mesh  # NOQA

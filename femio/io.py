
from .fem_data import FEMData


def read_directory(*args, **kwargs):
    """Initialize FEMData object from directory.

    See FEMData.read_directory for more detail.
    """
    return FEMData.read_directory(*args, **kwargs)


def read_files(*args, **kwargs):
    """Initialize FEMData object from files.

    See FEMData.read_files for more detail.
    """
    return FEMData.read_files(*args, **kwargs)

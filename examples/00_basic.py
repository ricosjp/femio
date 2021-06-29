"""
Basic Usage of FEMIO
====================

FEMIO can be used as a converter, load data in a format, and write data in
another format.
"""

###############################################################################
# Import numpy and :mod:`femio`.

import numpy as np
import femio

###############################################################################
# First, download the
# `VTK file <https://github.com/ricosjp/femio/examples/hex.vtk>`_ and
# place it in the same directory as that of this script.
# If you load the file with
# `ParaView <https://www.paraview.org/>`_, it should look as follows.
#
# .. image:: ../../examples/00_basic_fig/hex_raw.png
#   :width: 400

###############################################################################
# Then, read the file using femio.read_files function.
# It returns a :class:`~femio.fem_data.FEMData` object that contains mesh data.

fem_data = femio.read_files('vtk', 'hex.vtk')

###############################################################################
# Elsewise, one can generate a simple mesh using FEMIO's function.

# fem_data = femio.generate_brick('hex', 1, 1, 2)

###############################################################################
# A FEMData object has various attributes, e.g., nodes, elements.
#
# The attribute :code:`nodes` has ids and data that means node positions.

print(f"Node IDs:\n{fem_data.nodes.ids}\n")
print(f"Node data (positions):\n{fem_data.nodes.data}")

###############################################################################
# The attribute :code:`elements` has ids and data that means node
# connectivities based on node IDs.

print(f"Element IDs:\n{fem_data.elements.ids}\n")
print(f"Element data (positions):\n{fem_data.elements.data}")

###############################################################################
# Here, please note that the term 'ID' differs from the array's index.
# Array's index always starts from zero and is consecutive.
# However, ID does not necessarily start from zero and is consecutive.
# By default, ID starts from one even if the original format's ID starts
# from zero (like VTK).
#
# Please be aware that they correspond to :code:`loc` and :code:`iloc` in
# `pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html>`_,
# and one can actually use them to access data.

print(fem_data.nodes.loc[3].values)  # Access with ID
print(fem_data.nodes.iloc[2].values)  # Access with index

###############################################################################
# Now, let's add analysis conditions to perform heat analysis using
# `FrontISTR <https://github.com/FrontISTR/FrontISTR>`_.
#
# To add boundary conditions, we first create :class:`~FEMAttribute` objects,
# then add them to the :code:`fem_data`.

fixtemp = femio.FEMAttribute('fixtemp', np.array([1, 12]), np.array([0., 1.]))
fem_data.constraints.update({'fixtemp': fixtemp})

###############################################################################
# If you want to add several data with the same IDs, you can use the
# :meth:`femio.fem_attributes.FEMAttributes.update_data` method.
# Here, :code:`'MAT_ALL'` is the ID (can be multiple).

fem_data.materials.update_data(
    'MAT_ALL', {
        'density': np.array([[1., 0.]]),
        'specific_heat': np.array([[1., 0.]]),
        'thermal_conductivity': np.array([[1., 0.]])})
fem_data.settings['solution_type'] = 'HEAT'

###############################################################################
# Next, we add the section's information to connect the material defined above
# to the element group (:code:`'ALL'` here means all elements in the mesh).
fem_data.sections.update_data(
    'MAT_ALL', {'TYPE': 'SOLID', 'EGRP': 'ALL'})

###############################################################################
# Then, we write a FrontISTR data directory.
fem_data.write('fistr', '00_basic_out/mesh', overwrite=True)

###############################################################################
# Finally, run FrontISTR like a bash script shown below (Docker required).
#
# .. code-block:: bash
#
#   cd 00_basic_out
#   docker pull registry.gitlab.com/frontistr-commons/frontistr/fistr1:master
#   docker run -it --sig-proxy=false --rm -u $UID -v $PWD:$PWD -w $PWD \
#     registry.gitlab.com/frontistr-commons/frontistr/fistr1:master fistr1 -t 1
#
# If you load the resultant file :code:`00_basic_out/mesh_vis_psf.0001.inp`
# in ParaView, it will look as follows.
#
# .. image:: ../../examples/00_basic_fig/res.png
#   :width: 400
#
# In addition, you can load that file and analyze the data.

res_fem_data = femio.read_files('ucd', '00_basic_out/mesh_vis_psf.0001.inp')
temperature = res_fem_data.nodal_data['TEMPERATURE'].data
print(f"\nCalculated temperature:\n{temperature}\n")
print(f"Mean temperature:\n{np.mean(temperature)}")

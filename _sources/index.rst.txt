.. femio documentation master file, created by
   sphinx-quickstart on Sat Dec 26 20:26:08 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: fig/femio_logo.svg
  :width: 400
  :alt: FEMIO

FEMIO: FEM and mesh I/O tool
============================
*FEMIO* is a tool to handle discretized geometry data and simulation data
associated with them, e.g., meshes for finite element method, point clouds
for particle method.
Using FEMIO, one can perform preprocessing and postprocessing using FEMIO's
unified python interface.
Also, FEMIO provides various data processing tools,
e.g., geometry processing and graph signal processing.
So far, FEMIO supports the following file formats:

.. list-table::
  :widths: 40 20 20 10 10
  :header-rows: 1

  * - File format name
    - Extension
    - Name in FEMIO
    - Read
    - Write
  * - VTK legacy
    - .vtk
    - :code:`'vtk'`
    - yes
    - yes
  * - AVS UCD
    - .inp
    - :code:`'ucd'`
    - yes
    - yes
  * - FrontISTR
    - .msh, .cnt, .res
    - :code:`'fistr'`
    - yes
    - yes
  * - Wavefront OBJ
    - .obj
    - :code:`'obj'`
    - yes
    - yes
  * - Stereolithography
    - .stl
    - :code:`'stl'`
    - yes
    - yes


Installation
============

FEMIO supports Python 3.8 or newer.
FEMIO can be installed via pip:

.. code-block:: bash

    $ pip install femio

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  examples/index
  femio

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

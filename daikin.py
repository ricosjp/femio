from mesh_compressor import *

import femio
import numpy as np

fem_data = femio.read_files(
    'polyvtk',
    '/home/group/ricos/data/daikin/step1_202203/prediction/In_all/In_all_H/In_all_H_030.vtu'
)
# 260万節点, 210万要素
fem_data = fem_data.to_polyhedron()

compressor = MeshCompressor(fem_data=fem_data)
fem_data1 = compressor.compress(
    elem=2000, cos_thresh=0.80, dist_thresh=0.00
)
fem_data1.write('polyvtk', 'out1.vtu', overwrite=True)

compressor = MeshCompressor(fem_data=fem_data1)
fem_data2 = compressor.compress(
    elem=200, cos_thresh=0.80, dist_thresh=0.25
)
fem_data2.write('polyvtk', 'out2.vtu', overwrite=True)

compressor = MeshCompressor(fem_data=fem_data2)
fem_data3 = compressor.compress(
    elem=5, cos_thresh=0.00, dist_thresh=1.40
)

fem_data3.write('polyvtk', 'out3.vtu', overwrite=True)

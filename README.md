# Femio
The FEM I/O + mesh processing tool.

Femio can:
- Read FEM data including analysis results from various formats
- Perform mesh processing
- Write FEM data to various formats


## How to install
```bash
pip install femio
```


## How to use
Usage could be something similar to this:

```python
import femio

# Read FEM data of files
fem_data = femio.FEMData.read_files(file_type='ucd', file_names=['mesh.inp'])
# Read FEM data in a directory
fem_data = femio.FEMData.read_directory(file_type='ucd', 'directory/name')

# Access FEM data
print(fem_data.nodes.ids, fem_data.entity.nodes.data)  # data means node position here
print(fem_data.elements.ids, fem_data.entity.elements.data)  # data means node ids here
print(fem_data.nodal_data['DISPLACEMENT'].ids, fem_data.entity.nodal_data['DISPLACEMENT']).data

# Output FEM data to a file format different from the input
fem_data.write(file_type='stl')
```

Supported file types:
- 'fistr': FrontISTR file format
- 'obj': Wavefront .obj file format
- 'stl': STereoLithography file format
- 'ucd': AVS UCD old format
- 'vtk': VTK format


## License

[Apache License 2.0](./LICENSE).

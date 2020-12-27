# Femio
The FEM I/O + mesh processing tool.

Femio can:
- Read FEM data including analysis results from various formats
- Perform mesh processing
- Write FEM data to various formats


## How to install
```bash
pip install poetry
poetry install
```


## How to use
Usage could be something similar to this:

```python
import femio

# Read FEM data of files
fem_data = femio.FEMData.read_files('input_type', ['mesh_file', 'results_file'])
# Read FEM data in a directory (all files excluding include files shoud be in the same directory)
fem_data = femio.FEMData.read_directory('input_type', 'directory/name')

# Access FEM data
print(fem_data.nodes.ids, fem_data.entity.nodes.data)  # data means node position here
print(fem_data.elements.ids, fem_data.entity.elements.data)  # data means node ids here
print(fem_data.nodal_data['DISPLACEMENT'].ids, fem_data.entity.nodal_data['DISPLACEMENT']).data

# Output FEM data to a file format different from the input
fem_data.write('output_type')
```

Parameters:
- input_type: Type of input file format
- list_files: List of file names (can be arbitrary order)
  - mesh_file: file containing mesh information
  - results_file: file containig results information (can be omitted)
- output_type: type of output file format

Currently, the following types are supported.

Input type:
- FrontISTR: 'fistr'
- AVS UCD old format: 'ucd'

Output type:
- FrontISTR: 'fistr'
- AVS UCD old format: 'ucd'


## Examples

```bash
# FrontISTR to UCD
python3 fistr2ucd.py frontistr/files/directory
```


## How to test
In the top directory,

```bash
./run_test.sh
```


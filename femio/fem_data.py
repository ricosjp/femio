import glob
from pathlib import Path
import re

import meshio
import networkx as nx
import numpy as np
from numba import njit

from . import config
from . import functions
from .fem_attribute import FEMAttribute
from .fem_attributes import FEMAttributes
from .fem_elemental_attribute import FEMElementalAttribute
from .geometry_processor import GeometryProcessorMixin
from .graph_processor import GraphProcessorMixin
from .signal_processor import SignalProcessorMixin
from .util import string_parser as st


class FEMData(
        GraphProcessorMixin, GeometryProcessorMixin, SignalProcessorMixin):
    """Represents FEM data generary.

    Attributes:
        file_names: util.string.StringSeries object indicating source files.
        nodes: FEMAttribute object.
            ids: Node IDs.
            data: Node positions.
        elements: FEMAttribute object or dict of FEMAttribute objects.
            ids: Element IDs.
            data: Node IDs composing the element.
        nodal_data: dict with key = string, value = FEMAttribute object. ids of
            each component corresponds to node IDs.
        elemental_data: dict with key = string, value = FEMAttribute object.
            ids of each component corresponds to node IDs.
        node_groups: dict with key = string, value = ndarray of ints of IDs.
        element_groups: dict with key = string, value = ndarray of ints of IDs.
        materials: dict with key = string, value = FEMMaterial object.
        sections: list of dicts with keys == ['EGRP', 'MATERIALS'] indicating
            material asignments to element groups.
        settings: dict with key = string, value = any object.
    """

    @classmethod
    def read_files(
            cls, file_type, file_names, *,
            read_mesh_only=False, time_series=False):
        """Initialize FEMData object from files.

        Args:
            file_type: Sting indicating type of FEM. The following formats are
                supported:
                    - 'fistr': FrontISTR
                    - 'ucd': UCD old format
                    - 'obj': Wavefront obj format
                    - 'stl': Stereolithography format
            file_names: List of strings indicating file names. Order of files
                is arbitrary.
            read_mesh_only: Bool. If true, read mesh (nodes and elements) and
                ignore material data, results and so on. Default is False.
            time_series: bool, optional [False]
                If True, parse files as time series data.
        """
        if isinstance(file_names, (str, Path)):
            file_names = [file_names]
        file_names = st.StringSeries(
            [str(file_name) for file_name in file_names])
        if file_type == 'fistr':
            from .formats.fistr import fistr
            cls_ = fistr.FrontISTRData
        elif file_type == 'ucd':
            from .formats.ucd import ucd
            cls_ = ucd.UCDData
        elif file_type == 'stl':
            from .formats.stl import stl
            cls_ = stl.STLData
        elif file_type == 'obj':
            from .formats.obj import obj
            cls_ = obj.ObjData
        elif file_type in ['polyvtk', 'vtu']:
            from .formats.polyvtk import polyvtk
            cls_ = polyvtk.PolyVTKData
        elif file_type == 'ensight':
            from .formats.ensight import ensight
            cls_ = ensight.EnsightGoldData
        elif file_type == 'vtp':
            from .formats.vtp import vtp
            cls_ = vtp.VTPData

        # File formats supported by meshio
        elif file_type == 'vtk':
            return cls.read_meshio_file(file_names)

        else:
            raise NotImplementedError(
                f"Unknown file_type: {file_type}")
        obj = cls_.read_files(
            file_names, read_mesh_only=read_mesh_only, time_series=time_series)

        if isinstance(obj, list):
            return [o._to_fem_data() for o in obj]
        else:
            return obj._to_fem_data()

    def _read_files(
            self, pattern, *,
            mandatory=True, pattern_ignore=r'(?:#|^\s*$)', strip=False):
        """Read files matched to the specified pattern.

        Args:
            pattern: str
                Pattern to be used to search for the file name.
            mandatory: bool, optional [True]
                If True, raises ValueError when no file found.
            pattern_ignore: str, optional
                Pattern to ignore lines in the file.
            strip: bool, optional [False]
                If True, stip each line in the file.
        Returns:
            StringSeries object or List[StringSeries] object.
        """
        matched_files = self.file_names.find_match(pattern)
        if len(matched_files) == 0:
            if mandatory:
                raise ValueError('No file found for: {}'.format(pattern))
            else:
                return None

        s = st.StringSeries.read_files(
            matched_files, pattern_ignore=pattern_ignore,
            separate=self.time_series)
        if strip:
            s = s.strip()

        return s

    @classmethod
    def read_directory(
            cls, file_type, dir_name, *, read_mesh_only=False,
            recursive=False, read_npy=True, save=True,
            read_res=True, stem=None, time_series=False):
        """Initialize FEMData object from directory.

        Parameters
        ----------
        file_type: Sting indicating type of FEM. The following formats are
            supported:
                - 'fistr': FrontISTR
                - 'ucd': UCD old format
        dir_name: String of directory name.
        read_mesh_only: Bool. If true, read mesh (nodes and elements) and
            ignore material data, results and so on. Default is False.
        recursive: Bool if make recursive search.
        read_npy: Bool if read npy files when exist
        stem: Stem of files to be read.
        """
        dir_name = Path(dir_name)
        if not dir_name.is_dir():
            raise ValueError(f"{dir_name} is not directory")

        if read_npy and (dir_name / 'femio_npy_saved.npy').exists():
            print('Npy files exist. Read npy files instead.')
            return cls.read_npy_directory(
                dir_name, read_mesh_only=read_mesh_only)

        if file_type == 'fistr':
            if read_res:
                extensions = ['msh', 'cnt', 'res.*']
            else:
                extensions = ['msh', 'cnt']
        elif file_type == 'ucd':
            extensions = ['inp']
        elif file_type == 'obj':
            extensions = ['obj']
        elif file_type == 'stl':
            extensions = ['stl']
        elif file_type == 'vtk':
            extensions = ['vtk']
        elif file_type == 'polyvtk':
            extensions = ['vtu']
        elif file_type == 'vtp':
            extensions = ['vtp']
        else:
            raise NotImplementedError(
                f"Unknown file_type: {file_type}")

        if stem is None:
            stem = '*'

        def find(extension):
            if recursive:
                files = glob.glob(
                    str(dir_name / (stem + '.' + extension))) \
                    + glob.glob(str(
                        dir_name / ('**/' + stem + '.' + extension)))
            else:
                files = glob.glob(
                    str(dir_name / (stem + '.' + extension)))
            if len(files) == 1:
                found_files = [files[0]]
            elif len(files) > 1:
                steps = [
                    int(re.findall(r'\d+$', f)[-1]) for f in files]
                sorted_files = list(np.array(files)[np.argsort(steps)])
                if time_series:
                    found_files = sorted_files
                else:
                    # NOTE: Use the last one
                    found_files = [sorted_files[-1]]
            else:
                found_files = [None]

            return found_files

        file_names = sum([find(ext) for ext in extensions], [])
        file_names_wo_none = [f for f in file_names if f is not None]
        if len(file_names_wo_none) == 0:
            raise ValueError(f"Files not found in {dir_name}")
        obj = cls.read_files(
            file_type, file_names_wo_none, read_mesh_only=read_mesh_only,
            time_series=time_series)
        if save and not (dir_name / 'femio_npy_saved.npy').exists():
            obj.save(dir_name)
        return obj

    @classmethod
    def read_npy_directory(cls, dir_name, *, read_mesh_only=False):
        """Initialize FEMData object from directory with npy files.
        Args:
            dir_name: String of directory name.
            read_mesh_only: Bool. If true, read mesh (nodes and elements) and
                ignore material data, results and so on. Default is False.
            recursive: Bool if make recursive search.
        """
        files = glob.glob(str(Path(dir_name) / 'femio_*.np*'))
        dict_files = {Path(f).stem: f for f in files}

        nodes = FEMAttribute.load(
            'NODE', dict_files['femio_nodes'])

        elements = FEMElementalAttribute.load(
            'ELEMENT', dict_files['femio_elements'])
        obj = cls(nodes, elements)

        # Read 'face' attribute in case of polyhedral data
        if 'femio_elemental_data' in dict_files:
            elemental_data = FEMAttributes.load(
                dict_files['femio_elemental_data'], is_elemental=True)
            if 'face' in elemental_data:
                obj.elemental_data['face'] = elemental_data['face']

        if read_mesh_only:
            return obj

        obj.settings.update({'solution_type': None})
        if 'femio_settings' in dict_files:
            obj.settings.update(
                dict(np.load(dict_files['femio_settings'], allow_pickle=True)))
            if 'solution_type' in obj.settings:
                if isinstance(obj.settings['solution_type'], np.ndarray):
                    obj.settings['solution_type'] = str(
                        obj.settings['solution_type'])
                if obj.settings['solution_type'] == 'None':
                    obj.settings['solution_type'] = None

        if 'femio_nodal_data' in dict_files:
            obj.nodal_data.update(FEMAttributes.load(
                dict_files['femio_nodal_data']))
        if 'femio_elemental_data' in dict_files:
            obj.elemental_data.update(elemental_data)
        if 'femio_constraints' in dict_files:
            obj.constraints.update(FEMAttributes.load(
                dict_files['femio_constraints']))

        if obj.settings['solution_type'] is None:
            obj.settings['solution_type'] = 'STATIC'

        return obj._to_fem_data()

    @classmethod
    def from_meshio(cls, meshio_data):
        """Construct FEMData object from meshio.Mesh object.

        Parameters
        ----------
        meshio_data: meshio.Mesh

        Returns
        -------
        fem_data: FEMData
        """
        nodes = FEMAttribute(
            'NODE', ids=np.arange(len(meshio_data.points)) + 1,
            data=meshio_data.points)

        # NOTE: So far only tetra10 is supported
        elements = FEMElementalAttribute.from_meshio(meshio_data.cells)

        nodal_data = FEMAttributes.from_meshio(
            nodes.ids, meshio_data.point_data)

        if len(meshio_data.cell_data.keys()) == 1:
            elemental_data = FEMAttributes.from_meshio(
                elements.ids, meshio_data.cell_data, is_elemental=True)
        else:
            print('Mixed cell type detected. Skip loading elemental data')
            elemental_data = None

        return cls(
            nodes=nodes, elements=elements, nodal_data=nodal_data,
            elemental_data=elemental_data)

    @classmethod
    def read_meshio_file(cls, file_names, read_mesh_only=False):
        """Read files supported with meshio and create FEMData object.

        Parameters
        ----------
        file_names: femio.util.StringSeries
            File names.
        read_mesh_only: bool, optional [False]
            If true, read mesh (nodes and elements) and ignore
            material data, results and so on.

        Returns
        -------
        fem_data: FEMData
        """
        if len(file_names) != 1:
            raise ValueError(
                f"{len(file_names)} files found. "
                'Specify file name by using read_files() instead of '
                'read_directory().')
        file_name = file_names[0]

        print('Parsing data')
        mesh = meshio.read(str(file_name))
        return cls.from_meshio(mesh)

    def __init__(
            self, nodes=None, elements=None, *,
            nodal_data=None, elemental_data=None,
            node_groups=None, element_groups=None, materials=None,
            sections=None, constraints=None, settings=None, file_names=None):
        """Initialize FEMData object.

        Args:
            nodes: FEMAttribute
                ids: Node IDs.
                data: Node positions.
            elements: FEMElementalAttribute or FEMAttribute
                ids: Element IDs.
                data: Node IDs composing the element.
            nodal_data: FEMAttributes
                Nodal data with key = string, value = FEMAttribute object.
                Ids of each component corresponds to node IDs.
            elemental_data: FEMAttributes
                Elemental data with key = string, value = FEMAttribute object.
                Ids of each component corresponds to element IDs.
            node_groups: Dict[str, numpy.ndarray[int]], optional [None]
                Node groups with key = string, value = ndarray of ints of node
                IDs.
            element_groups: Dict[str, numpy.ndarray[int]], optional [None]
                Element groups with key = string, value = ndarray of ints of
                element IDs.
            materials: FEMAttributes, optional [None]
                Material properties with key = property name,
                value = FEMAttribute object with ids = material names and
                data = material property values.
            sections: FEMAttributes, optional [None]
                Section data with key = section name (e.g. EGRP),
                value = FEMAttribute object with ids = material names and
                data = section names (e.g. Element group names).
            constraints: FEMAttributes
                Constraints data with key = constraint name,
                value = FEMAttribute constraint value with
                ids = node or element IDs and
                data = constraint property values.
            settings: dict, optional [None]
                Settings with key = string, value = any object.
            file_names: List[str], optional [None]
                File names of the data.
        """
        if nodes is None:
            self.nodes = None
        else:
            self.nodes = FEMAttribute(
                'NODE', nodes.ids, nodes.data,
                data_unit=nodes.data_unit, silent=True, generate_id2index=True)
        if elements is None:
            self.elements = None
        else:
            self.elements = FEMElementalAttribute('ELEMENT', elements)
        self.nodal_data = nodal_data or FEMAttributes([])
        self.elemental_data = elemental_data or FEMAttributes(
            [], is_elemental=True)
        self.node_groups = node_groups or {}
        self.element_groups = element_groups or {}
        self.materials = materials or FEMAttributes([])
        self.sections = sections or FEMAttributes([])
        self.constraints = constraints or FEMAttributes([])
        self.settings = settings or {}
        if file_names is None:
            self.file_names = []
        else:
            self.file_names = file_names

        self.time_series = False
        self.material_overwritten = False

        if self.nodes is None or self.elements is None:
            return

        self._update_dict_id2index()

        # Add node position to nodal data
        self.nodal_data['NODE'] = self.nodes

        return

    def _update_dict_id2index(self):
        self.dict_node_id2index = {
            _id: ind for ind, _id in enumerate(self.nodes.ids)}
        self.dict_element_id2index = {
            id_val: i for i, id_val in enumerate(self.elements.ids)}
        return

    def overwritten_material_exists(self):
        """Check if material is overwritten.

        Returns
        -------
        overwritten_material_exists: bool
        """
        return self.elemental_data.material_overwritten

    def _to_fem_data(self):
        nodes = FEMAttribute(
            'NODE', self.nodes.ids, self.nodes.data,
            data_unit=self.nodes.data_unit, silent=True,
            generate_id2index=True)
        return FEMData(
            nodes=nodes, elements=self.elements,
            nodal_data=self.nodal_data, elemental_data=self.elemental_data,
            node_groups=self.node_groups, element_groups=self.element_groups,
            materials=self.materials, sections=self.sections,
            constraints=self.constraints, settings=self.settings,
            file_names=self.file_names)

    def to_meshio(self):
        cell_info = self.elements.to_meshio(self.nodes)
        point_data = self.nodal_data.to_meshio()
        cell_data = self.elemental_data.to_meshio()

        meshio_mesh = meshio.Mesh(
            self.nodes.data, cell_info,
            point_data=point_data, cell_data=cell_data)
        return meshio_mesh

    def write(
            self, file_type, file_name=None, *, overwrite=False,
            write_msh_only=False, include_displacement=False):
        """Write FEM data into the specified format.

        Args:
            file_type: Sting indicating type of FEM. The following formats are
                supported:
                    - 'fistr': FrontISTR
                    - 'ucd': UCD old format
                    - 'stl': stereolithography format
                    - 'obj': Wavefront OBJ format
            file_name: File name of the output file. If not fed,
                input_filename.out.ext will be the output file name.
            overwrite: Bool, if True, allow averwrite files (Default: False.)
            write_msh_only: Bool, if True, omit writing cnt file
                (Default: False,) for file_type == 'fistr' case.
        """
        print('Start writing data')
        if file_name is None:
            file_name = Path(
                str(self.fem_data.file_names[0]) + '.out.'
                + config.DICT_EXT[file_type])
        else:
            file_name = Path(file_name)
        if not overwrite and file_name.exists():
            raise ValueError(f"{file_name} already exists")
        if not file_name.parent.exists():
            file_name.parent.mkdir(parents=True)

        if file_type == 'fistr':
            from .formats.fistr.write_fistr import FistrWriter
            written_files = FistrWriter(self).write(
                file_name=file_name, overwrite=overwrite,
                write_msh_only=write_msh_only)

        elif file_type == 'ucd':
            from .formats.ucd.write_ucd import UCDWriter
            written_files = UCDWriter(self).write(
                file_name=self.add_extension_if_needed(file_name, 'inp'),
                overwrite=overwrite)

        elif file_type == 'stl':
            from .formats.stl.write_stl import STLWriter
            written_files = STLWriter(
                self, include_displacement=include_displacement).write(
                    file_name=self.add_extension_if_needed(file_name, 'stl'),
                    overwrite=overwrite)

        elif file_type == 'obj':
            from .formats.obj.write_obj import OBJWriter
            written_files = OBJWriter(self).write(
                file_name=self.add_extension_if_needed(file_name, 'obj'),
                overwrite=overwrite)

        elif file_type in ['polyvtk', 'vtu']:
            from .formats.polyvtk.write_polyvtk import PolyVTKWriter
            written_files = PolyVTKWriter(self).write(
                file_name=self.add_extension_if_needed(file_name, 'vtu'),
                overwrite=overwrite)

        elif file_type == 'vtp':
            from .formats.vtp.write_vtp import VTPWriter
            written_files = VTPWriter(self).write(
                file_name=self.add_extension_if_needed(file_name, 'vtp'),
                overwrite=overwrite)

        # File formats supported by meshio
        elif file_type == 'vtk':
            meshio_mesh = self.to_meshio()
            meshio.write(file_name, meshio_mesh, file_format='vtk')
            written_files = file_name

        else:
            raise NotImplementedError

        if isinstance(written_files, (str, Path)):
            written_file = written_files
            print(f"File written in: {written_file}")
        elif isinstance(written_files, list):
            for written_file in written_files:
                print(f"File written in: {written_file}")
        else:
            raise ValueError(f"No written file found: {written_files}")
        return

    def add_extension_if_needed(self, file_name, ext):
        if str(file_name).endswith(ext):
            return file_name
        else:
            return Path(str(file_name) + '.' + ext)

    def save(self, dir_name, *, save_mesh_only=False):
        """Write FEM data into the specified format.

        Args:
            dir_name: str or pathlib.Path
                Directory name to be saved.
            save_mesh_only: Bool, optional, [False]
                If true, save only node and element information.
        """
        dir_name = Path(dir_name)
        if not dir_name.exists():
            dir_name.mkdir(parents=True)

        # Save mesh data
        self.nodes.save(dir_name / 'femio_nodes')
        self.elements.save(dir_name / 'femio_elements')
        if save_mesh_only:
            print(f"Nodes and elements saved in: {dir_name}")
            (dir_name / 'femio_npy_saved.npy').touch()
            return

        # Save nodal and elemental data
        self.nodal_data.save(dir_name / 'femio_nodal_data')
        self.elemental_data.save(dir_name / 'femio_elemental_data')
        self.constraints.save(dir_name / 'femio_constraints')
        np.savez(dir_name / 'femio_settings', **self.settings)
        (dir_name / 'femio_npy_saved.npy').touch()
        return

    def remove_useless_nodes(self):
        """Remove useless nodes which are not referenced from elements.

        Args:
            None
        Returns:
            resultant_nodes: FEMAttribute object
        """
        if isinstance(self.elements, dict):
            useful_node_ids = np.unique(np.concatenate([
                np.ravel(v.data) for v in self.elements.values()]))
        else:
            useful_node_ids = np.unique(self.elements.data)
        original_sorted_indices = np.argsort(self.nodes.ids)
        original_node_ids = self.nodes.ids[original_sorted_indices]
        if len(original_node_ids) == len(useful_node_ids):
            if np.all(useful_node_ids == original_node_ids):
                return
            else:
                raise ValueError('Node IDs are inconsistent with elements')
        print('Nodes not used in elements found. Removing.')

        filter_useful_nodes = np.ones(len(original_node_ids), dtype=bool)
        original_node_index = 0
        useful_node_index = 0
        while useful_node_index < len(useful_node_ids):
            if original_node_ids[original_node_index] != useful_node_ids[
                    useful_node_index]:
                filter_useful_nodes[original_node_index] = False
                original_node_index += 1
                continue

            original_node_index += 1
            useful_node_index += 1
        filter_useful_nodes[original_node_index:] = False
        useful_indices = original_sorted_indices[filter_useful_nodes]

        # Overwrite data
        self.nodes = FEMAttribute(
            self.nodes.name, self.nodes.ids[useful_indices],
            self.nodes.data[useful_indices], generate_id2index=True)
        self._update_dict_id2index()
        for key, value in self.nodal_data.items():
            self.nodal_data[key] = FEMAttribute(
                value.name, self.nodes.ids, value.data[useful_indices])
        return

    def to_first_order(self):
        """Convert the FEMData object to the first order data.

        Returns
        -------
        FEMData:
            First order FEMData object.
        """
        if np.all(['2' not in key for key in self.elements.keys()]):
            return self

        filter_ = self.filter_first_order_nodes()
        nodes = FEMAttribute(
            'NODE', self.nodes.ids[filter_], self.nodes.loc[filter_].values)
        elements = self.elements.to_first_order()
        nodal_data = FEMAttributes({
            k: FEMAttribute(
                k, v.ids[filter_], v.loc[filter_].values,
                time_series=v.time_series)
            for k, v in self.nodal_data.items() if len(filter_) == len(v.ids)})
        elemental_data = self.elemental_data
        fem_data = FEMData(
            nodes, elements, nodal_data=nodal_data,
            elemental_data=elemental_data)
        fem_data.materials = self.materials
        fem_data.constraints = self.constraints
        fem_data.settings = self.settings
        return fem_data

    def to_surface(self, *, remove_unnecessary_nodes=True):
        """Convert the FEMData object to the surface data.

        Parameters
        ----------
        remove_unnecessary_nodes: bool, optional
            If True, remove nodes unnecessary for surface. The default is
            True.

        Returns
        -------
        FEMData:
            Surface FEMData object.
        """
        def flatten(indices, element_type):
            if element_type == 'polygon':
                return np.concatenate(indices)
            else:
                return np.ravel(indices)

        surface_indices, _ = self.extract_surface()
        if isinstance(surface_indices, dict):
            unique_indices = np.unique(np.concatenate([
                flatten(v, k) for k, v in surface_indices.items()]))
            surface_ids = {
                t: self.nodes.ids[ids] for t, ids in surface_indices.items()
                if t != 'polygon'}
            if 'polygon' in surface_indices \
                    and len(surface_indices['polygon']) > 0:
                polygon_data = np.empty(
                    len(surface_indices['polygon']), object)
                polygon_data[:] = [
                    self.nodes.ids[i]
                    for i in surface_indices['polygon']]
                surface_ids.update({'polygon': polygon_data})
        else:
            unique_indices = np.unique(surface_indices)
            surface_ids = self.nodes.ids[surface_indices]

        if remove_unnecessary_nodes:
            node_ids = self.nodes.ids[unique_indices]
            nodes = FEMAttribute(
                'NODE', node_ids, self.nodes.iloc[unique_indices].values)
            n_node = len(self.nodes)
            nodal_data = FEMAttributes({
                k: FEMAttribute(
                    k, node_ids, v.iloc[unique_indices].values,
                    time_series=v.time_series)
                for k, v in self.nodal_data.items() if len(v) == n_node})
        else:
            nodes = self.nodes
            nodal_data = self.nodal_data
        elements = self.elements.to_surface(surface_ids)

        return FEMData(
            nodes=nodes, elements=elements, nodal_data=nodal_data,
            elemental_data={})

    def to_facets(
            self, remove_duplicates=True, return_dict_facets=False,
            dict_facets=None):
        """Convert the FEMData object to the facet data including facets inside
        the solid.

        Parameters
        ----------
        remove_duplicates: bool, optional
            If True, remove duplicated faces and remain only one. The default
            is True.
        return_dict_facets: bool, optional
            If True, also return dict_facets.

        Returns
        -------
        FEMData:
            Facets FEMData object.
        """
        if dict_facets is None:
            dict_facets = self.extract_facets(
                remove_duplicates=remove_duplicates)

        elements = self.elements.to_surface(dict_facets)
        facet_fem_data = FEMData(
            nodes=self.nodes, elements=elements, nodal_data=self.nodal_data,
            elemental_data={})

        if return_dict_facets:
            return facet_fem_data, dict_facets
        else:
            return facet_fem_data

    @staticmethod
    @njit
    def convert_polyhedron(now_ids, new_ids, poly):
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            poly[L:R] = now_ids[poly[L:R]]
            poly[L:R] = np.searchsorted(new_ids, poly[L:R])
            L = R
        return list(poly)

    def cut_with_element_ids(self, element_ids):
        node_ids = np.unique(np.concatenate([
            np.concatenate(e.data) for e
            in self.elements.filter_with_ids(element_ids).values()]))
        nodes = self.nodes.filter_with_ids(node_ids)
        nodal_data = FEMAttributes({
            k: v.filter_with_ids(node_ids) for k, v in self.nodal_data.items()
        })
        elemental_data = FEMAttributes({
            k: v.filter_with_ids(element_ids)
            for k, v in self.elemental_data.items()}, is_elemental=True)
        # convert face data
        have_face = False
        if 'face' in elemental_data:
            if 'polyhedron' in elemental_data['face']:
                have_face = True
        if have_face:
            dat = elemental_data['face']['polyhedron'].data
            n = len(dat)
            newdat = np.empty(n, object)
            for i in range(n):
                poly = np.array(dat[i])
                newdat[i] = self.convert_polyhedron(
                    self.nodes.ids, node_ids, poly)
            elemental_data['face']['polyhedron'].data = newdat
        cut_fem_data = FEMData(
            nodes=nodes,
            elements=self.elements.filter_with_ids(element_ids),
            nodal_data=nodal_data,
            elemental_data=elemental_data
        )
        return cut_fem_data

    def cut_with_element_type(self, element_type):
        element_ids = self.elements[element_type].ids
        return self.cut_with_element_ids(element_ids)

    def cut_with_node_ids(self, node_ids):
        nodes = self.nodes.filter_with_ids(node_ids)
        element_ids = self.cut_elements_with_node_ids(node_ids)
        nodal_data = FEMAttributes({
            k: v.filter_with_ids(node_ids) for k, v in self.nodal_data.items()
        })
        elemental_data = FEMAttributes({
            k: v.filter_with_ids(element_ids)
            for k, v in self.elemental_data.items()}, is_elemental=True)
        cut_fem_data = FEMData(
            nodes=nodes,
            elements=self.elements.filter_with_ids(element_ids),
            nodal_data=nodal_data,
            elemental_data=elemental_data
        )

        return cut_fem_data

    def cut_elements_with_node_ids(self, node_ids):
        return np.array([
            id_ for id_, element
            in zip(self.elements.ids, self.elements.data)
            if self._elements_exist(element, node_ids)
        ])

    def _elements_exist(self, element, node_ids):
        return np.all([np.any(e - node_ids == 0) for e in element])

    def _read_ideas_universal(self, string_series, names=None):
        """Read I-DEAS Universal formatted data. This function parses the
        content of the input StringSeries object and add the parsed data to
        nodal or elemental data.

        Args:
            string_series: femio.util.string_parser.StringSeries
                StringSeries object of file contents.
            names: list-like of str
                Variable names which will be given to the contents of the data.
        Return:
            None
        """

        written_variable_names = string_series[2].rstrip(',').split(',')
        n_variable = len(written_variable_names)
        if names is not None:
            if len(names) != n_variable:
                raise ValueError(
                    'Length of names are not consistent with data '
                    f"({len(names)} vs {n_variable}")
            variable_names = names
        else:
            variable_names = written_variable_names

        str_data_block_number = string_series[1]
        if str_data_block_number == '55':
            data_type = 'nodal'
        else:
            raise ValueError(
                f"Unknown data block number: {str_data_block_number}")

        start_line_index = 8 + n_variable
        written_data = string_series[start_line_index:-1:2].to_values(
            delimiter=r'\s+')
        for variable_name, data in zip(variable_names, written_data.T):
            if data_type == 'nodal':
                self.nodal_data[variable_name] = FEMAttribute(
                    variable_name, self.nodes.ids, data)
            else:
                raise ValueError(
                    f"Unknown data block number: {str_data_block_number}")
        return

    def add_static_material(self):
        """Add simple material data for static analysis."""
        self.materials.update_data(
            'M1',
            {'Young_modulus': np.array([1.]), 'Poisson_ratio': np.array([.3])})
        self.sections.update_data(
            'M1', {'TYPE': 'SOLID', 'EGRP': 'ALL'})
        return

    def generate_graph_fem_data(
            self, adjs, *, mode='nodal', attribute_name='data'):
        """Generate FEMData of the specified graphs.

        Parameters
        ----------
        adjs: List[scipy.sparse]
            Adjacency matrices with the same shape and the same non-zero
            profile.
        mode: str, optional
            'nodal' or 'elemental'.
        attribute_name: str, optional
            The name of the edge feature. The default is 'data'.

        Returns
        -------
        graph_fem_data: femio.FEMData
            FEMData object of the specified graph.
        """
        if mode == 'nodal':
            positions = self.nodes.data
        elif mode == 'elemental':
            positions = self.convert_nodal2elemental(
                self.nodes.data, calc_average=True)

        aligned_adjs = functions.align_nnz(adjs)
        graphs = [
            nx.from_scipy_sparse_matrix(
                adj, parallel_edges=False, create_using=nx.DiGraph)
            for adj in aligned_adjs]
        edge_attributes = np.stack([
            np.array(list(nx.get_edge_attributes(graph, 'weight').values()))
            for graph in graphs], axis=-1)

        middle_positions = np.array([
            (positions[start] + positions[end]) / 2
            for start, end in graphs[0].edges()])
        all_positions = np.concatenate(
            [positions, middle_positions], axis=0)

        positions_attribute = FEMAttribute(
            'NODE', ids=np.arange(len(all_positions)) + 1,
            data=all_positions)
        n_positions = len(positions)
        segments = np.array([
            [edge[0], n_positions + i]
            for i, edge in enumerate(graphs[0].edges())]) + 1
        segments_attribute = FEMAttribute(
            'line', ids=np.arange(len(segments)) + 1, data=segments)
        graph_fem_data = FEMData(
            nodes=positions_attribute, elements={'line': segments_attribute})
        graph_fem_data.elemental_data.update_data(
            graph_fem_data.elements.ids, {attribute_name: edge_attributes})
        return graph_fem_data

    def create_node_group(self, group_name, selected):
        """Create new node_group.

        Parameters
        ----------
        group_name: str
            The name of new node group.
        selected: np.ndarray, bool
            1D mask array, containing data with boolean type.
        """
        if group_name in self.node_groups:
            raise ValueError(f"node group {group_name} already exists")
        self.node_groups[group_name] = self.nodes.ids[selected]

    def create_element_group_from_node_group(
            self, element_group_name, node_group_name, kind='all'):
        """Create new surface group from a node group.

        An element is in the created group when (all / any) of the nodes
        is contained in the node group.

        Parameters
        ----------
        element_group_name: str
            The name of new element group.

        node_group_name: str
            The name of original node group.

        kind: {'all', 'any'}, default 'all'
            Type of creation rule.
        """
        element_data = self.elements.data
        target_nodes = self.node_groups[node_group_name]
        target_nodes = np.sort(target_nodes)
        count = np.searchsorted(target_nodes, element_data, 'right')
        count -= np.searchsorted(target_nodes, element_data, 'left')
        isin = count > 0

        element_ids = self.elements.ids
        if kind == 'all':
            selected = np.all(isin, axis=1)
        elif kind == 'any':
            selected = np.any(isin, axis=1)
        else:
            raise ValueError('kind must be all or any')
        elem_group = element_ids[selected]
        if element_group_name in self.element_groups:
            raise ValueError(
                f"element group {element_group_name} already exists")
        self.element_groups[element_group_name] = elem_group

    def extract_with_element_indices(self, element_indices):
        """Extract a sub-FEMData object with the specified element indices.

        Parameters
        ----------
        element_indices: numpy.ndarray[int]
            Element indices to extract the data.

        Returns
        -------
        sub_fem_data: femio.FEMData
            Extracted FEMData object.
        """
        element_data = self.elements.data[element_indices]
        element_ids = self.elements.ids[element_indices]
        node_ids = np.unique(np.concatenate(element_data))

        nodes = self.nodes.filter_with_ids(node_ids)
        elements = self.elements.filter_with_ids(element_ids)
        nodal_data = self.nodal_data.filter_with_ids(node_ids)
        elemental_data = self.elemental_data.filter_with_ids(
            element_ids)
        return FEMData(
            nodes=nodes, elements=elements, nodal_data=nodal_data,
            elemental_data=elemental_data)

    def resolve_degeneracy(self):
        """Resolve degeneracy in hex elements, and
        return resolved new FEMData."""

        fem_data = FEMData(
            nodes=self.nodes, elements=self.elements,
            nodal_data=self.nodal_data, elemental_data={}
        )
        elements = fem_data.elements

        if 'hex' not in self.elements:
            return fem_data

        hex_ids = elements['hex'].ids
        hex_data = elements['hex'].data
        equal_01 = hex_data[:, 0] == hex_data[:, 1]
        equal_12 = hex_data[:, 1] == hex_data[:, 2]
        equal_23 = hex_data[:, 2] == hex_data[:, 3]
        equal_30 = hex_data[:, 3] == hex_data[:, 0]
        if not all((
            np.all(hex_data[equal_01, 4] == hex_data[equal_01, 5]),
            np.all(hex_data[equal_12, 5] == hex_data[equal_12, 6]),
            np.all(hex_data[equal_23, 6] == hex_data[equal_23, 7]),
            np.all(hex_data[equal_30, 7] == hex_data[equal_30, 4]),
        )):
            raise ValueError("Unknown degeneracy pattern in hex")
        nondegenerate = ~(equal_01 | equal_12 | equal_23 | equal_30)

        if 'prism' in elements:
            prism_ids = elements['prism'].ids
            prism_data = elements['prism'].data
        else:
            prism_ids = np.empty(0, hex_ids.dtype)
            prism_data = np.empty((0, 6), hex_data.dtype)

        prism_ids = np.concatenate([
            prism_ids,
            hex_ids[equal_01],
            hex_ids[equal_12],
            hex_ids[equal_23],
            hex_ids[equal_30],
        ])
        prism_data = np.concatenate([
            prism_data,
            hex_data[equal_01][:, [0, 3, 2, 4, 7, 6]],
            hex_data[equal_12][:, [0, 3, 1, 4, 7, 5]],
            hex_data[equal_23][:, [0, 2, 1, 4, 6, 5]],
            hex_data[equal_30][:, [0, 2, 1, 4, 6, 5]],
        ])
        IDX = np.argsort(prism_ids)
        prism_ids = prism_ids[IDX]
        prism_data = prism_data[IDX]

        hex_ids = hex_ids[nondegenerate]
        hex_data = hex_data[nondegenerate]

        if len(hex_ids) > 0:
            hex = FEMAttribute('hex', ids=hex_ids, data=hex_data)
            elements.update({'hex': hex})
        else:
            del elements['hex']
            fem_data.elements._update_self()

        if len(prism_ids) > 0:
            prism = FEMAttribute('prism', ids=prism_ids, data=prism_data)
            elements.update({'prism': prism})

        return fem_data

    @staticmethod
    @njit
    def tet_to_polyhedron(dat, node_ids, argsort):
        a, b, c, d = argsort[np.searchsorted(node_ids, dat)]
        faces = [[a, c, b], [d, a, b], [d, c, a], [d, b, c]]
        face_dat = [len(faces)]
        for F in faces:
            face_dat.append(len(F))
            face_dat += F
        return face_dat

    @staticmethod
    @njit
    def hex_to_polyhedron(dat, node_ids, argsort):
        a, b, c, d, e, f, g, h = argsort[np.searchsorted(node_ids, dat)]
        faces = [[e, f, g, h], [f, e, a, b], [g, f, b, c],
                 [h, g, c, d], [e, h, d, a], [d, c, b, a]]
        face_dat = [len(faces)]
        for F in faces:
            face_dat.append(len(F))
            face_dat += F
        return face_dat

    @staticmethod
    @njit
    def prism_to_polyhedron(dat, node_ids, argsort):
        a, b, c, d, e, f = argsort[np.searchsorted(node_ids, dat)]
        faces = [[a, b, c], [f, e, d], [
            b, a, d, e], [c, b, e, f], [a, c, f, d]]
        face_dat = [len(faces)]
        for F in faces:
            face_dat.append(len(F))
            face_dat += F
        return face_dat

    @staticmethod
    @njit
    def pyr_to_polyhedron(dat, node_ids, argsort):
        a, b, c, d, e = np.searchsorted(node_ids, dat)
        faces = [[d, c, b, a], [a, b, e], [b, c, e], [c, d, e], [d, a, e]]
        face_dat = [len(faces)]
        for F in faces:
            face_dat.append(len(F))
            face_dat += F
        return face_dat

    def to_polyhedron(self):
        node_ids = self.nodes.ids
        argsort = node_ids.argsort()
        node_ids = node_ids[argsort]
        elements = self.elements.data

        n_elem = len(elements)
        face_dat = np.empty(n_elem, object)
        types = np.unique(self.elements.types)
        for tp in types:
            indices = np.where(self.elements.types == tp)[0]
            if tp == 'tet':
                for i in indices:
                    face_dat[i] = self.tet_to_polyhedron(
                        elements[i].astype(np.int32), node_ids, argsort)
            elif tp == 'hex':
                for i in indices:
                    face_dat[i] = self.hex_to_polyhedron(
                        elements[i].astype(np.int32), node_ids, argsort)
            elif tp == 'prism':
                for i in indices:
                    face_dat[i] = self.prism_to_polyhedron(
                        elements[i].astype(np.int32), node_ids, argsort)
            elif tp == 'pyr':
                for i in indices:
                    face_dat[i] = self.pyr_to_polyhedron(
                        elements[i].astype(np.int32), node_ids, argsort)
            elif tp == 'polyhedron':
                pass
            else:
                raise NotImplementedError(
                    f"to_polyhedron is not supported for : {tp}")

        if 'face' in self.elemental_data:
            indices = np.where(self.elements.types == 'polyhedron')[0]
            dat = self.elemental_data['face']['polyhedron'].data
            for i, x in zip(indices, dat):
                face_dat[i] = x

        nodes = self.nodes
        polyhedron = FEMAttribute(
            'polyhedron',
            ids=self.elements.ids,
            data=self.elements.data
        )
        elements = FEMElementalAttribute(
            'ELEMENT', {'polyhedron': polyhedron}
        )
        face = FEMElementalAttribute(
            'face', {
                'polyhedron':
                FEMAttribute('face', ids=self.elements.ids, data=face_dat)
            }
        )
        fem_data = FEMData(
            nodes=nodes, elements=elements,
            nodal_data=self.nodal_data, elemental_data=self.elemental_data
        )
        fem_data.elemental_data.update({'face': face})
        return fem_data

    def face_data_csr(self):
        face_data_list = self.elemental_data['face']['polyhedron'].data
        indptr = [len(row) for row in face_data_list]
        indptr = np.append(0, np.cumsum(indptr))
        return (indptr, np.concatenate(face_data_list))

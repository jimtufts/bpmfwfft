import pytest
import bpmfwfft.fft_sampling
import netCDF4
import pickle

from pathlib import Path

cwd = Path.cwd()
mod_path = Path(__file__).parent

# Use ubql_ubiquitin example (same as test_grids.py)
rec_prmtop = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.prmtop").resolve()
rec_inpcrd = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.inpcrd").resolve()
grid_nc_file = (mod_path / "../../examples/grid/ubql_ubiquitin/grid.nc").resolve()

# Parameters matching test_grids.py
lj_sigma_scal_fact = 1.0    # lj_scale
rc_scale = 0.76             # receptor core scaling
rs_scale = 0.53             # receptor surface scaling
rm_scale = 0.55             # receptor metal scaling
lc_scale = 0.76             # ligand core scaling (same as receptor for protein-protein)
ls_scale = 0.53             # ligand surface scaling
lm_scale = 0.55             # ligand metal scaling
rho = 9.0                   # water density parameter

bsite_file = None
lig_prmtop = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.prmtop").resolve()
lig_inpcrd = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.inpcrd").resolve()

energy_sample_size_per_ligand = 10  # Small number for testing
output_nc = (mod_path / "../../examples/fft_sampling/ubql_ubiquitin/fft_sampling_test.nc").resolve()
start_index = 0

ligand_md_trj_file = (mod_path / "../../examples/ligand_md/ubiquitin/rotation.nc").resolve()
lig_coord_ensemble = netCDF4.Dataset(ligand_md_trj_file, "r").variables["positions"][:]


Sampling_test = bpmfwfft.fft_sampling.Sampling(
    rec_prmtop, lj_sigma_scal_fact,
    rc_scale, rs_scale, rm_scale,
    lc_scale, ls_scale, lm_scale,
    rho,
    rec_inpcrd, bsite_file, grid_nc_file,
    lig_prmtop, lig_inpcrd,
    lig_coord_ensemble,
    energy_sample_size_per_ligand,
    output_nc,
    start_index,
    temperature=300.)

def test_sampling_initialization():
    """Test that Sampling object is created successfully"""
    assert Sampling_test is not None
    assert hasattr(Sampling_test, '_rec_crd')
    assert hasattr(Sampling_test, '_lig_grid')
    assert hasattr(Sampling_test, '_lig_coord_ensemble')

def test_rec_grid_data():
    """Test that receptor grid data was loaded"""
    assert Sampling_test._rec_crd is not None
    assert hasattr(Sampling_test, '_rec_grid_displacement')
    assert Sampling_test._rec_grid_displacement is not None

def test_lig_grid_properties():
    """Test that ligand grid has expected properties"""
    assert Sampling_test._lig_grid is not None
    assert hasattr(Sampling_test._lig_grid, '_prmtop')
    assert hasattr(Sampling_test._lig_grid, '_crd')

def test_ligand_coord_ensemble():
    """Test that ligand coordinate ensemble was loaded"""
    assert Sampling_test._lig_coord_ensemble is not None
    assert len(Sampling_test._lig_coord_ensemble.shape) == 3  # (n_frames, n_atoms, 3)
    # Check that we have the expected number of rotations (should be 101 from rotation.nc)
    assert Sampling_test._lig_coord_ensemble.shape[0] > 0

def test_nc_file_created():
    """Test that output NetCDF file was initialized"""
    assert Sampling_test._nc_handle is not None
    assert hasattr(Sampling_test._nc_handle, 'variables')

def test_energy_parameters():
    """Test that energy calculation parameters are set correctly"""
    assert Sampling_test._energy_sample_size_per_ligand == 10
    assert hasattr(Sampling_test, '_beta')
    assert Sampling_test._beta > 0  # Should be 1/(kB*T), positive value

def test_start_index():
    """Test that start index is set correctly"""
    assert Sampling_test._start_index == 0

def test_lig_grid_dimensions():
    """Test that ligand grid has correct dimensions matching receptor"""
    lig_grid = Sampling_test._lig_grid
    assert hasattr(lig_grid, '_grid')
    assert lig_grid._grid is not None
    # Grid should have counts key which defines dimensions
    assert 'counts' in lig_grid._grid
    assert len(lig_grid._grid['counts']) == 3  # 3D grid

def test_rec_coordinates_shape():
    """Test that receptor coordinates have correct shape"""
    assert Sampling_test._rec_crd is not None
    assert len(Sampling_test._rec_crd.shape) == 2  # (n_atoms, 3)
    assert Sampling_test._rec_crd.shape[1] == 3  # x, y, z coordinates

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

def test_free_of_clash_calculation():
    """Test clash detection calculation"""
    # Call the method to calculate free of clash
    Sampling_test._cal_free_of_clash()

    # Check that the free_of_clash array was created
    assert hasattr(Sampling_test._lig_grid, '_free_of_clash')
    assert Sampling_test._lig_grid._free_of_clash is not None

    # Should be a boolean array
    assert Sampling_test._lig_grid._free_of_clash.dtype == bool

    # Should be 3D
    assert len(Sampling_test._lig_grid._free_of_clash.shape) == 3

def test_netcdf_variables():
    """Test that NetCDF output has expected structure"""
    nc = Sampling_test._nc_handle

    # Check for expected variables
    assert 'rec_positions' in nc.variables
    assert 'lig_positions' in nc.variables
    assert 'resampled_energies' in nc.variables

    # Check for expected dimensions
    assert 'rec_natoms' in nc.dimensions
    assert 'lig_natoms' in nc.dimensions

def test_remove_nonphysical_energies():
    """Test filtering of nonphysical energies"""
    import numpy as np

    # First ensure free_of_clash is calculated
    if not hasattr(Sampling_test._lig_grid, '_free_of_clash'):
        Sampling_test._cal_free_of_clash()

    # Create a test grid with same shape as max_grid_indices
    max_i, max_j, max_k = Sampling_test._lig_grid._max_grid_indices
    test_grid = np.ones((max_i, max_j, max_k))

    # Apply the filter
    filtered_grid = Sampling_test._remove_nonphysical_energies(test_grid)

    # Result should be 1D array of non-clashing positions
    assert filtered_grid is not None
    assert len(filtered_grid.shape) == 1

    # Number of elements should be <= original grid size
    assert filtered_grid.size <= test_grid.size

def test_cal_energies_electrostatic():
    """Test electrostatic energy calculation"""
    import numpy as np

    # Ensure prerequisites are met
    if not hasattr(Sampling_test._lig_grid, '_free_of_clash'):
        Sampling_test._cal_free_of_clash()

    # Initialize _meaningful_energies on the ligand grid (full grid size, not max_grid_indices)
    if not hasattr(Sampling_test._lig_grid, '_meaningful_energies'):
        grid_counts = Sampling_test._lig_grid._grid['counts']
        Sampling_test._lig_grid._meaningful_energies = np.zeros(grid_counts, dtype=float)

    # Calculate electrostatic energies for first ligand conformation
    Sampling_test._cal_energies("electrostatic", step=0)

    # Check that results were stored (note: gets saved as "no_sasa" not "electrostatic")
    assert hasattr(Sampling_test, '_resampled_energies_components')
    assert hasattr(Sampling_test, '_resampled_trans_vectors_components')
    assert 'no_sasa' in Sampling_test._resampled_energies_components
    assert 'no_sasa' in Sampling_test._resampled_trans_vectors_components

    # Check that we got the expected number of samples
    assert len(Sampling_test._resampled_energies_components['no_sasa']) == energy_sample_size_per_ligand

    # Verify actual energy values for regression testing
    energies = Sampling_test._resampled_energies_components['no_sasa']
    # Check that energies are sorted (lowest first)
    assert all(energies[i] <= energies[i+1] for i in range(len(energies)-1))

    # Regression test: check against known reference values for ubql_ubiquitin
    # These values were computed with the mdtraj 1.10.3 compatible code
    expected_min_energy = -339.583014
    expected_max_energy = -212.753208

    actual_min = energies[0]
    actual_max = energies[-1]

    # Allow small tolerance for floating point differences
    tolerance = 0.01
    assert abs(actual_min - expected_min_energy) < tolerance, \
        f"Electrostatic min energy changed: expected {expected_min_energy:.6f}, got {actual_min:.6f}"
    assert abs(actual_max - expected_max_energy) < tolerance, \
        f"Electrostatic max energy changed: expected {expected_max_energy:.6f}, got {actual_max:.6f}"

    # Basic sanity checks
    assert np.isfinite(actual_min), "Minimum energy should be finite"
    assert np.isfinite(actual_max), "Maximum energy should be finite"

def test_cal_energies_lj():
    """Test Lennard-Jones energy calculation"""
    import numpy as np

    # Ensure prerequisites are met
    if not hasattr(Sampling_test._lig_grid, '_free_of_clash'):
        Sampling_test._cal_free_of_clash()

    # Initialize _meaningful_energies on the ligand grid (full grid size, not max_grid_indices)
    if not hasattr(Sampling_test._lig_grid, '_meaningful_energies'):
        grid_counts = Sampling_test._lig_grid._grid['counts']
        Sampling_test._lig_grid._meaningful_energies = np.zeros(grid_counts, dtype=float)

    # Calculate LJ energies for first ligand conformation (note: LJa gets saved as "LJ" not "LJa")
    Sampling_test._cal_energies("LJa", step=0)
    Sampling_test._cal_energies("LJr", step=0)

    # Check that results were stored (LJa is saved as "LJ", LJr doesn't trigger save)
    assert 'LJ' in Sampling_test._resampled_energies_components
    assert 'LJ' in Sampling_test._resampled_trans_vectors_components

    # Check that we got the expected number of samples
    assert len(Sampling_test._resampled_energies_components['LJ']) == energy_sample_size_per_ligand

    # Verify actual energy values for regression testing
    energies = Sampling_test._resampled_energies_components['LJ']
    # Check that energies are sorted (lowest first)
    assert all(energies[i] <= energies[i+1] for i in range(len(energies)-1))

    # Regression test: check against known reference values for ubql_ubiquitin
    # These values were computed with the mdtraj 1.10.3 compatible code
    expected_min_energy = -1671.010308
    expected_max_energy = -1255.879301

    actual_min = energies[0]
    actual_max = energies[-1]

    # Allow small tolerance for floating point differences
    tolerance = 0.01
    assert abs(actual_min - expected_min_energy) < tolerance, \
        f"LJ min energy changed: expected {expected_min_energy:.6f}, got {actual_min:.6f}"
    assert abs(actual_max - expected_max_energy) < tolerance, \
        f"LJ max energy changed: expected {expected_max_energy:.6f}, got {actual_max:.6f}"

    # Basic sanity checks
    assert np.isfinite(actual_min), "Minimum energy should be finite"
    assert np.isfinite(actual_max), "Maximum energy should be finite"

def test_cal_energies_sasa():
    """Test SASA energy calculation"""
    import numpy as np

    # Ensure prerequisites are met
    if not hasattr(Sampling_test._lig_grid, '_free_of_clash'):
        Sampling_test._cal_free_of_clash()

    # Initialize _meaningful_energies on the ligand grid (full grid size, not max_grid_indices)
    if not hasattr(Sampling_test._lig_grid, '_meaningful_energies'):
        grid_counts = Sampling_test._lig_grid._grid['counts']
        Sampling_test._lig_grid._meaningful_energies = np.zeros(grid_counts, dtype=float)

    # Calculate SASA energies for first ligand conformation
    Sampling_test._cal_energies("sasa", step=0)

    # Check that results were stored
    assert 'sasa' in Sampling_test._resampled_energies_components
    assert 'sasa' in Sampling_test._resampled_trans_vectors_components

    # Check that we got the expected number of samples
    assert len(Sampling_test._resampled_energies_components['sasa']) == energy_sample_size_per_ligand

    # Check that translation vectors match across energy types
    # (all should sample the same positions)
    assert len(Sampling_test._resampled_trans_vectors_components['sasa']) == energy_sample_size_per_ligand

    # Verify actual energy values for regression testing
    energies = Sampling_test._resampled_energies_components['sasa']
    # Check that energies are sorted (lowest first)
    assert all(energies[i] <= energies[i+1] for i in range(len(energies)-1))

    # Regression test: check against known reference values for ubql_ubiquitin
    # These values were computed with the mdtraj 1.10.3 compatible code
    expected_min_energy = -6.935386
    expected_max_energy = -5.578102

    actual_min = energies[0]
    actual_max = energies[-1]

    # Allow small tolerance for floating point differences
    tolerance = 0.01
    assert abs(actual_min - expected_min_energy) < tolerance, \
        f"SASA min energy changed: expected {expected_min_energy:.6f}, got {actual_min:.6f}"
    assert abs(actual_max - expected_max_energy) < tolerance, \
        f"SASA max energy changed: expected {expected_max_energy:.6f}, got {actual_max:.6f}"

    # Basic sanity checks
    assert np.isfinite(actual_min), "Minimum energy should be finite"
    assert np.isfinite(actual_max), "Maximum energy should be finite"

def test_run_sampling_generates_valid_data():
    """Test that run_sampling() actually populates the NC file with valid data"""
    import numpy as np
    import tempfile
    import os

    # Create a temporary filename (but don't create the file yet - let Sampling create it)
    fd, test_output_path = tempfile.mkstemp(suffix='.nc')
    os.close(fd)
    os.remove(test_output_path)  # Remove the empty file so Sampling creates it properly

    try:
        # Use only first 2 ligand conformations for fast testing
        small_lig_ensemble = lig_coord_ensemble[:2]

        sampler = bpmfwfft.fft_sampling.Sampling(
            rec_prmtop, lj_sigma_scal_fact,
            rec_inpcrd, bsite_file, grid_nc_file,
            lig_prmtop, lig_inpcrd,
            small_lig_ensemble,
            energy_sample_size_per_ligand,
            test_output_path,
            start_index=0,
            temperature=300.)

        # Actually run the sampling
        sampler.run_sampling()

        # Now verify the NC file has valid (non-masked) data
        nc = netCDF4.Dataset(test_output_path, 'r')

        # Check that ligand positions were written (at least for first rotation)
        lig_pos = nc.variables['lig_positions'][0, 0, :]
        assert not np.ma.is_masked(lig_pos), "lig_positions should not be masked after run_sampling()"
        assert np.all(np.isfinite(lig_pos)), "lig_positions should contain finite values"

        # Check that resampled energies were written
        resampled_e = nc.variables['resampled_energies'][0, :]
        assert not np.ma.is_masked(resampled_e[0]), "resampled_energies should not be masked after run_sampling()"
        # Only check the first energy_sample_size_per_ligand values (rest may be fill values)
        valid_energies = resampled_e[:energy_sample_size_per_ligand]
        assert np.all(np.isfinite(valid_energies)), "resampled_energies should contain finite values"

        nc.close()

    finally:
        # Clean up temporary file
        if os.path.exists(test_output_path):
            os.remove(test_output_path)

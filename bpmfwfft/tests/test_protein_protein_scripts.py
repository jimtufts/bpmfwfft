import pytest
import sys
import os
import netCDF4
import numpy as np
from pathlib import Path
import tempfile

# Add protein_protein_scripts to path
mod_path = Path(__file__).parent
scripts_path = (mod_path / "../../protein_protein_scripts").resolve()
sys.path.insert(0, str(scripts_path))

from _receptor_grid_cal import is_nc_grid_good, get_grid_size_from_nc, get_grid_size_from_lig_rec_crd, rec_grid_cal
from _fft_sampling import is_sampling_nc_good, parse_nr_ligand_confs, sampling

# Test data paths
cwd = Path.cwd()
rec_prmtop_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.prmtop").resolve()
rec_inpcrd_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.inpcrd").resolve()
lig_prmtop_file = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.prmtop").resolve()
lig_inpcrd_file = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.inpcrd").resolve()
grid_nc_file = (mod_path / "../../examples/grid/ubql_ubiquitin/grid.nc").resolve()
rotation_nc_file = (mod_path / "../../examples/ligand_md/ubiquitin/rotation.nc").resolve()


class TestIsNcGridGood:
    """Tests for is_nc_grid_good() function"""

    def test_valid_grid_file(self):
        """Test that a valid grid file returns True"""
        assert is_nc_grid_good(grid_nc_file) == True

    def test_nonexistent_file(self):
        """Test that a nonexistent file returns False"""
        assert is_nc_grid_good("/path/to/nonexistent/file.nc") == False

    def test_empty_file(self, tmp_path):
        """Test that an empty file returns False"""
        empty_file = tmp_path / "empty.nc"
        empty_file.touch()
        assert is_nc_grid_good(empty_file) == False

    def test_invalid_nc_file(self, tmp_path):
        """Test that a netCDF file missing required keys returns False"""
        invalid_nc = tmp_path / "invalid.nc"
        with netCDF4.Dataset(invalid_nc, 'w') as nc:
            # Create a netCDF file but don't add required grid variables
            nc.createDimension('test', 1)
            nc.createVariable('test_var', 'f', ('test',))

        assert is_nc_grid_good(invalid_nc) == False


class TestGetGridSizeFromNc:
    """Tests for get_grid_size_from_nc() function"""

    def test_get_grid_size(self):
        """Test extracting grid size from valid netCDF file"""
        size = get_grid_size_from_nc(grid_nc_file)
        assert size is not None
        # Convert to scalar for comparison (handles masked arrays and numpy arrays)
        size_val = np.asscalar(size) if hasattr(np, 'asscalar') else int(size)
        assert size_val > 0
        assert size_val == 61  # Expected value for ubql_ubiquitin grid


class TestGetGridSizeFromLigRecCrd:
    """Tests for get_grid_size_from_lig_rec_crd() function"""

    def test_calculate_grid_size(self):
        """Test grid size calculation from receptor and ligand coordinates"""
        buffer = 1.0
        box_size = get_grid_size_from_lig_rec_crd(rec_inpcrd_file, lig_inpcrd_file, buffer)

        assert box_size is not None
        assert isinstance(box_size, (int, float, np.floating))
        assert box_size > 0

    def test_different_buffers(self):
        """Test that larger buffer produces larger box size"""
        small_buffer = 1.0
        large_buffer = 10.0

        small_box = get_grid_size_from_lig_rec_crd(rec_inpcrd_file, lig_inpcrd_file, small_buffer)
        large_box = get_grid_size_from_lig_rec_crd(rec_inpcrd_file, lig_inpcrd_file, large_buffer)

        assert large_box > small_box


class TestIsSamplingNcGood:
    """Tests for is_sampling_nc_good() function"""

    def test_nonexistent_nc_file(self):
        """Test with nonexistent sampling nc file"""
        result = is_sampling_nc_good("/nonexistent/file.nc", lig_inpcrd_file)
        assert result == False

    def test_nonexistent_rotation_file(self, tmp_path):
        """Test with nonexistent rotation nc file"""
        fake_nc = tmp_path / "fake.nc"
        fake_nc.touch()
        result = is_sampling_nc_good(fake_nc, "/nonexistent/rotation.nc")
        assert result == False

    def test_both_files_nonexistent(self):
        """Test with both files nonexistent"""
        result = is_sampling_nc_good("/nonexistent1.nc", "/nonexistent2.nc")
        assert result == False


class TestParseNrLigandConfs:
    """Tests for parse_nr_ligand_confs() function"""

    def test_parse_valid_submit_file(self, tmp_path):
        """Test parsing nr_lig_conf from a valid submit file"""
        submit_file = tmp_path / "submit.job"
        submit_content = """#!/bin/bash
#SBATCH --job-name=test
python script.py --nr_lig_conf 100 --other_arg value
"""
        submit_file.write_text(submit_content)

        result = parse_nr_ligand_confs(submit_file)
        assert result == 100

    def test_parse_with_multiple_args(self, tmp_path):
        """Test parsing when nr_lig_conf is among multiple arguments"""
        submit_file = tmp_path / "submit.job"
        submit_content = """#!/bin/bash
python script.py --arg1 val1 --nr_lig_conf 250 --arg2 val2
"""
        submit_file.write_text(submit_content)

        result = parse_nr_ligand_confs(submit_file)
        assert result == 250

    def test_nonexistent_file(self):
        """Test with nonexistent submit file"""
        result = parse_nr_ligand_confs("/nonexistent/submit.job")
        assert result is None

    def test_file_without_nr_lig_conf(self, tmp_path):
        """Test with submit file that doesn't contain nr_lig_conf"""
        submit_file = tmp_path / "submit.job"
        submit_content = """#!/bin/bash
python script.py --other_arg value
"""
        submit_file.write_text(submit_content)

        result = parse_nr_ligand_confs(submit_file)
        assert result is None


class TestRecGridCalRegression:
    """Regression tests for rec_grid_cal() function"""

    def test_rec_grid_cal_produces_expected_output(self, tmp_path):
        """Verify rec_grid_cal() produces output consistent with RecGrid class"""
        # Parameters matching those used in test_grids.py
        lj_scale = 1.0
        spacing = 2.0  # larger spacing for faster test
        buffer = 1.0
        radii_type = "VDW_RADII"
        exclude_H = True

        # Output files
        grid_out = tmp_path / "test_grid.nc"
        pdb_out = tmp_path / "receptor_trans.pdb"
        box_out = tmp_path / "box.pdb"

        # Run rec_grid_cal
        rec_grid_cal(
            rec_prmtop_file, lj_scale,
            rec_inpcrd_file, lig_inpcrd_file,
            spacing, buffer,
            grid_out, pdb_out, box_out,
            radii_type, exclude_H
        )

        # Verify output files were created
        assert grid_out.exists()
        assert pdb_out.exists()
        assert box_out.exists()

        # Verify the grid nc file is valid
        assert is_nc_grid_good(grid_out) == True

        # Compare with expected grid structure
        with netCDF4.Dataset(grid_out, 'r') as test_nc:
            # Check all required grid variables are present
            required_vars = ['occupancy', 'electrostatic', 'LJr', 'LJa', 'sasa', 'water',
                           'spacing', 'counts', 'origin', 'lj_sigma_scaling_factor']
            for var in required_vars:
                assert var in test_nc.variables, f"Missing variable: {var}"

            # Verify spacing matches input
            spacing_val = test_nc.variables['spacing'][:]
            assert np.allclose(spacing_val, spacing, rtol=1e-10)

            # Verify lj_sigma_scaling_factor matches input
            lj_scale_val = test_nc.variables['lj_sigma_scaling_factor'][:]
            assert np.allclose(lj_scale_val, lj_scale, rtol=1e-10)

            # Verify grid dimensions are consistent
            counts = test_nc.variables['counts'][:]
            for grid_var in ['occupancy', 'electrostatic', 'LJr', 'LJa', 'sasa', 'water']:
                grid_shape = test_nc.variables[grid_var].shape
                assert grid_shape == tuple(counts), f"{grid_var} shape mismatch"


class TestSamplingRegression:
    """Regression tests for sampling() function"""

    def test_sampling_produces_expected_output(self, tmp_path):
        """Verify sampling() produces output consistent with Sampling class"""
        # Parameters matching those used in test_fft_sampling.py
        lj_sigma_scal_fact = 1.0
        nr_lig_conf = 1  # Just one rotation for fast test
        energy_sample_size_per_ligand = 10  # Small number for fast test
        output_nc = tmp_path / "test_sampling.nc"
        output_dir = str(tmp_path)

        # Run sampling
        sampling(
            rec_prmtop_file, lj_sigma_scal_fact,
            rec_inpcrd_file, grid_nc_file,
            lig_prmtop_file, lig_inpcrd_file,
            rotation_nc_file, nr_lig_conf,
            energy_sample_size_per_ligand,
            output_nc, output_dir
        )

        # Verify output file was created
        assert output_nc.exists()

        # Verify the sampling nc file structure
        with netCDF4.Dataset(output_nc, 'r') as test_nc:
            # Check that required variables are present
            assert 'lig_positions' in test_nc.variables
            assert 'rec_positions' in test_nc.variables
            assert 'resampled_energies' in test_nc.variables
            assert 'resampled_trans_vectors' in test_nc.variables

            # Verify we processed the expected number of rotations
            lig_positions = test_nc.variables['lig_positions'][:]
            assert lig_positions.shape[0] == nr_lig_conf

            # Verify poses were sampled
            resampled_energies = test_nc.variables['resampled_energies'][:]
            # Each rotation should have energy_sample_size_per_ligand poses
            assert resampled_energies.shape[0] == nr_lig_conf
            assert resampled_energies.shape[1] == energy_sample_size_per_ligand

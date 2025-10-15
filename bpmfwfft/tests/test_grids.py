import pytest
import bpmfwfft.grids
import netCDF4
import os
import numpy as np
from pathlib import Path

cwd = Path.cwd()
mod_path = Path(__file__).parent

# Use ubql_ubiquitin example with production parameters
rec_prmtop_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.prmtop").resolve()
rec_inpcrd_file = (mod_path / "../../examples/amber/ubql_ubiquitin/receptor.inpcrd").resolve()
grid_nc_file = (mod_path / "../../examples/grid/ubql_ubiquitin/grid.nc").resolve()

# Parameters matching production usage from run_receptor_grid_cal.py
lj_sigma_scaling_factor = 1.0  # lj_scale
rec_core_scaling = 0.76        # rc_scale
rec_surface_scaling = 0.53     # rs_scale
rec_metal_scaling = 0.55       # rm_scale
rho = 9.0                      # water density parameter
spacing = 2.0                  # grid spacing in Angstroms (larger for faster tests)
bsite_file = None

# Ligand scaling parameters (same as receptor for protein-protein)
lig_core_scaling = 0.76
lig_surface_scaling = 0.53
lig_metal_scaling = 0.55

rec_grid = bpmfwfft.grids.RecGrid(rec_prmtop_file, lj_sigma_scaling_factor,
                        rec_core_scaling, rec_surface_scaling, rec_metal_scaling,
                        rho,
                        rec_inpcrd_file,
                        bsite_file,
                        grid_nc_file,
                        new_calculation=False,
                        spacing=spacing)
lig_prmtop_file = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.prmtop").resolve()
lig_inpcrd_file = (mod_path / "../../examples/amber/ubql_ubiquitin/ligand.inpcrd").resolve()
lig_grid = bpmfwfft.grids.LigGrid(lig_prmtop_file, lj_sigma_scaling_factor,
                        lig_core_scaling, lig_surface_scaling, lig_metal_scaling,
                        lig_inpcrd_file, rec_grid)

print(lig_grid.get_initial_com())

def test_is_nc_grid_good():
    assert bpmfwfft.grids.is_nc_grid_good(grid_nc_file) == True
    assert bpmfwfft.grids.is_nc_grid_good("trash") == False
#
#
# #Grid class tests
#
def test_get_six_corner_shifts():
    print(bpmfwfft.grids.Grid._get_six_corner_shifts(rec_grid))
#     assert bpmfwfft.grids.Grid._get_six_corner_shifts()
#
def test_set_grid_key_value():
    good_key = 'sasa'
    bad_key = 'this_key_is_bad'
    original_value = rec_grid._grid[good_key]
    bpmfwfft.grids.Grid._set_grid_key_value(rec_grid, good_key, 1)
    assert rec_grid._grid[good_key] == 1
    assert bad_key not in rec_grid._grid_allowed_keys
    rec_grid._grid[good_key] = original_value
#
def test_load_prmtop():
    assert rec_grid._prmtop is not None
    assert "MASS" in rec_grid._prmtop
    assert "CHARGE_E_UNIT" in rec_grid._prmtop
    assert len(rec_grid._prmtop["MASS"]) == rec_grid.get_natoms()

def test_load_inpcrd():
    assert rec_grid._crd is not None
    assert rec_grid._crd.shape[0] == rec_grid.get_natoms()
    assert rec_grid._crd.shape[1] == 3

def test_move_molecule_to():
    original_com = rec_grid._get_molecule_center_of_mass()
    new_location = np.array([100.0, 100.0, 100.0])
    rec_grid._move_molecule_to(new_location)
    new_com = rec_grid._get_molecule_center_of_mass()
    assert np.allclose(new_com, new_location, rtol=1e-10)
    rec_grid._move_molecule_to(original_com)
#
def test_get_molecule_center_of_mass():
    masses = rec_grid._prmtop["MASS"]
    crd = rec_grid._crd
    expected_com = np.sum(masses[:, np.newaxis] * crd, axis=0) / masses.sum()
    com = bpmfwfft.grids.Grid._get_molecule_center_of_mass(rec_grid)
    assert np.allclose(com, expected_com, rtol=1e-10)
#
def test_get_molecule_sasa():
    test_sasa = bpmfwfft.grids.Grid._get_molecule_sasa(rec_grid, 0.14, 960)
    print("test_sasa shape", test_sasa.shape)
    print("test_sasa", test_sasa)
    print("sum test_sasa", test_sasa.sum())


def test_get_corner_crd():
    np.set_printoptions(precision=15)
    print(bpmfwfft.grids.Grid._get_corner_crd(rec_grid, [1,1,1]))
#    assert bpmfwfft.grids.Grid._get_corner_crd()
#
def test_get_uper_most_corner():
    corner = rec_grid._get_uper_most_corner()
    counts = rec_grid._grid['counts']
    expected = counts - 1
    assert np.array_equal(corner, expected)

def test_get_uper_most_corner_crd():
    corner_crd = rec_grid._get_uper_most_corner_crd()
    origin = rec_grid._grid['origin']
    counts = rec_grid._grid['counts']
    spacing = rec_grid._grid['spacing']
    expected = origin + (counts - 1) * spacing
    assert np.allclose(corner_crd, expected, rtol=1e-10)

def test_get_origin_crd():
    origin = rec_grid._get_origin_crd()
    expected = rec_grid._grid['origin']
    assert np.array_equal(origin, expected)

def test_initialize_convenient_para():
    assert hasattr(lig_grid, '_origin_crd')
    assert hasattr(lig_grid, '_uper_most_corner_crd')
    assert hasattr(lig_grid, '_uper_most_corner')
    assert hasattr(lig_grid, '_spacing')
    assert isinstance(lig_grid._spacing, np.ndarray)
    assert len(lig_grid._spacing) == 3

def test_is_in_grid():
    origin = rec_grid._get_origin_crd()
    upper = rec_grid._get_uper_most_corner_crd()
    center = (origin + upper) / 2.0
    assert rec_grid._is_in_grid(center) == True
    far_outside = origin - np.array([1000.0, 1000.0, 1000.0])
    assert rec_grid._is_in_grid(far_outside) == False

def test_distance():
    corner = np.array([0, 0, 0])
    coord = np.array([3.0, 4.0, 0.0])
    dist = rec_grid._distance(corner, coord)
    expected = 5.0
    assert np.isclose(dist, expected)

def test_containing_cube():
    origin = rec_grid._get_origin_crd()
    spacing = rec_grid._grid['spacing']
    test_point = origin + spacing * 2.5
    result = rec_grid._containing_cube(test_point)
    assert len(result) == 3
    eight_corners, nearest_ind, furthest_ind = result
    assert len(eight_corners) == 8
    assert all(isinstance(c, np.ndarray) for c in eight_corners)

def test_get_grid_func_names():
    names = rec_grid.get_grid_func_names()
    expected = ("occupancy", "electrostatic", "LJr", "LJa", "sasa", "water")
    assert names == expected
#
#
# # LigGrid class tests

def test_get_initial_com():
    com = lig_grid.get_initial_com()
    assert com is not None
    assert len(com) == 3
    assert all(isinstance(x, (int, float, np.floating)) for x in com)

def test_get_number_translations():
    n_trans = lig_grid.get_number_translations()
    expected = int(lig_grid._max_grid_indices.prod())
    assert n_trans == expected

def test_place_ligand_crd_in_grid():
    molecular_coord = lig_grid._crd.copy()
    original_crd = lig_grid._crd.copy()
    result = lig_grid._place_ligand_crd_in_grid(molecular_coord)
    assert result is None
    assert np.array_equal(lig_grid._crd, original_crd)

def test_cal_grids():
    result = lig_grid.cal_grids()
    assert result is None
    assert lig_grid._meaningful_energies is not None
    assert isinstance(lig_grid._meaningful_energies, np.ndarray)
    assert lig_grid.get_meaningful_corners() is not None
#
def test_get_box_volume():
    spacing = lig_grid._grid['spacing']
    expected_volume = float(((lig_grid._max_grid_indices - 1) * spacing).prod())
    assert lig_grid.get_box_volume() == expected_volume
#
# def test_get_meaningful_energies():
#     assert lig_grid.get_meaningful_energies() == 5

# def test_get_meaningful_corners():
#     assert lig_grid.get_meaningful_corners() == 5

#
def test_set_meaningful_energies_to_none():
    assert lig_grid.set_meaningful_energies_to_none() == None
    assert lig_grid._meaningful_energies == None

# # RecGrid class tests

def test_load_precomputed_grids():
    assert 'occupancy' in rec_grid._grid
    assert 'electrostatic' in rec_grid._grid
    assert 'LJr' in rec_grid._grid
    assert 'LJa' in rec_grid._grid
    assert 'sasa' in rec_grid._grid
    assert 'water' in rec_grid._grid

def test_get_FFTs():
    ffts = rec_grid.get_FFTs()
    assert ffts is not None
    assert 'occupancy' in ffts
    assert 'electrostatic' in ffts

def test_get_charges():
    charges = rec_grid.get_charges()
    assert charges is not None
    assert isinstance(charges, dict)
    assert "CHARGE_E_UNIT" in charges

def test_get_rho():
    rho = rec_grid.get_rho()
    assert rho == 9.0

def test_exact_values():
    origin = rec_grid._get_origin_crd()
    spacing = rec_grid._grid['spacing']
    test_coord = origin + spacing * np.array([10, 10, 10])
    exact_vals = rec_grid._exact_values(test_coord)
    assert exact_vals is not None
    assert isinstance(exact_vals, dict)
    assert 'occupancy' in exact_vals
    assert 'electrostatic' in exact_vals

@pytest.mark.skip(reason="Function raises 'Do not use, not tested yet'")
def test_trilinear_interpolation():
    origin = rec_grid._get_origin_crd()
    upper = rec_grid._get_uper_most_corner_crd()
    center = (origin + upper) / 2.0
    if rec_grid._is_in_grid(center):
        interp_val = rec_grid._trilinear_interpolation('occupancy', center)
        assert isinstance(interp_val, (float, np.floating))

def test_direct_energy():
    lig_charges = lig_grid.get_charges()
    lig_crd = lig_grid._crd
    energy = rec_grid.direct_energy(lig_crd, lig_charges)
    assert isinstance(energy, (float, np.floating))

@pytest.mark.skip(reason="Function raises 'Do not use, not tested yet'")
def test_interpolated_energy():
    lig_charges = lig_grid.get_charges()
    lig_crd = lig_grid._crd
    energy = rec_grid.interpolated_energy(lig_crd, lig_charges)
    assert isinstance(energy, (float, np.floating))
#
# def test_get_FFTs():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_write_box():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_write_pdb():
#     assert bpmfwfft.grids.RecGrid.

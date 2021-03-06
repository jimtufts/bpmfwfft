import pytest
import bpmfwfft.grids
import netCDF4
import os
import numpy as np
from pathlib import Path

cwd = Path.cwd()
mod_path = Path(__file__).parent

grid_nc_file = (mod_path / "../../examples/grid/t4_lysozyme/grid.nc").resolve()

rec_prmtop_file = (mod_path / "../../examples/amber/t4_lysozyme/receptor_579.prmtop").resolve()
rec_inpcrd_file = (mod_path / "../../examples/amber/t4_lysozyme/receptor_579.inpcrd").resolve()
grid_nc_file = (mod_path / "../../examples/grid/t4_lysozyme/grid.nc").resolve()
lj_sigma_scaling_factor = 0.8
#bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
bsite_file = None
spacing = 0.25

rec_grid = bpmfwfft.grids.RecGrid(rec_prmtop_file, lj_sigma_scaling_factor, rec_inpcrd_file,
                        bsite_file,
                        grid_nc_file,
                        new_calculation=False,
                        spacing=spacing)
lig_prmtop_file = (mod_path / "../../examples/amber/benzene/ligand.prmtop").resolve()
lig_inpcrd_file = (mod_path / "../../examples/amber/benzene/ligand.inpcrd").resolve()
lig_grid = bpmfwfft.grids.LigGrid(lig_prmtop_file, lj_sigma_scaling_factor, lig_inpcrd_file, rec_grid)

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
    good_key = 'SASAr'
    bad_key = 'this_key_is_bad'
    original_value = rec_grid._grid[good_key]
    bpmfwfft.grids.Grid._set_grid_key_value(rec_grid, good_key, 1)
    assert rec_grid._grid[good_key] == 1
    assert bad_key not in rec_grid._grid_allowed_keys
    rec_grid._grid[good_key] = original_value
#    print("test", original_value)
#
# def test_load_prmtop():
#     assert bpmfwfft.grids.Grid._load_prmtop()
#
# def test_load_inpcrd():
#     assert bpmfwfft.grids.Grid._load_inpcrd()
#
# def test_move_molecule_to():
#     assert bpmfwfft.grids.Grid._move_molecule_to()
#
def test_get_molecule_center_of_mass():
#    np.set_printoptions(precision=15)
#    print(bpmfwfft.grids.Grid._get_molecule_center_of_mass(rec_grid))
    assert np.ndarray.tolist(bpmfwfft.grids.Grid._get_molecule_center_of_mass(rec_grid)) == [15.143843333007636, 12.397102311613432, 12.961784483800248]
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
# def test_get_uper_most_corner():
#     assert bpmfwfft.grids.Grid._get_uper_most_corner()
#
# def test_get_uper_most_corner_crd():
#     assert bpmfwfft.grids.Grid._get_uper_most_corner_crd()
#
# def test_get_origin_crd():
#     assert bpmfwfft.grids.Grid._get_origin_crd()
#
# def test_initialize_convenient_para():
#     assert bpmfwfft.grids.Grid._initialize_convenient_para()
#
# def test_is_in_grid():
#     assert bpmfwfft.grids.Grid.
#
# def test_distance():
#     assert bpmfwfft.grids.Grid.
#
# def test_containing_cube():
#     assert bpmfwfft.grids.Grid.
#
# def test_get_grid_func_names():
#     assert bpmfwfft.grids.Grid.
#
#
# # LigGrid class tests
#
# def test_cal_corr_func():
#     assert bpmfwfft.grids.LigGrid.
#
# def test_do_forward_fft():
#     assert bpmfwfft.grids.LigGrid.
#
# def test_cal_energies():
#     assert bpmfwfft.grids.LigGrid.
#
#
# def test_cal_energies_NOT_USED():
#     assert bpmfwfft.grids.LigGrid.
#
#
# def test_cal_meaningful_corners():
#     assert bpmfwfft.grids.LigGrid.
#
# def test_place_ligand_crd_in_grid():
#     assert bpmfwfft.grids.LigGrid.
#
# def test_cal_grids():
#     assert bpmfwfft.grids.LigGrid.
#
# def test_get_bpmf():
#     assert bpmfwfft.grids.LigGrid.
#
#
# def test_get_number_translations():
#     assert bpmfwfft.grids.LigGrid.
#
def test_get_box_volume():
    assert lig_grid.get_box_volume() == 1374.140625
#
# def test_get_meaningful_energies():
#     assert lig_grid.get_meaningful_energies() == 5

# def test_get_meaningful_corners():
#     assert lig_grid.get_meaningful_corners() == 5

#
def test_set_meaningful_energies_to_none():
    assert lig_grid.set_meaningful_energies_to_none() == None
    assert lig_grid._meaningful_energies == None

# def test_get_initial_com():
#     assert lig_grid.get_initial_com()
#
# # RecGrid class tests
#
# def test_load_precomputed_grids():
#     assert rec_grid.
#
# def test_cal_FFT():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_write_to_nc():
#     assert bpmfwfft.grids.RecGrid.
#
#
# def test_cal_grid_parameters_with_bsite():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_cal_grid_parameters_without_bsite():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_move_receptor_to_grid_center():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_cal_grid_coordinates():
#     assert bpmfwfft.grids.RecGrid.
#
#
# def test_get_charges():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_cal_potential_grids():
#     assert bpmfwfft.grids.RecGrid.
#
#
# def test_exact_values():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_trilinear_interpolation():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_direct_energy():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_interpolated_energy():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_get_FFTs():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_write_box():
#     assert bpmfwfft.grids.RecGrid.
#
# def test_write_pdb():
#     assert bpmfwfft.grids.RecGrid.

import pytest
import grids
import netCDF4
import os

print(os.getcwd())

grid_nc_file = "../examples/grid/t4_lysozyme/grid.nc"

rec_prmtop_file = "../examples/amber/t4_lysozyme/receptor_579.prmtop"
rec_inpcrd_file = "../examples/amber/t4_lysozyme/receptor_579.inpcrd"
grid_nc_file = "../examples/grid/t4_lysozyme/grid.nc"
lj_sigma_scaling_factor = 0.8
#bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
bsite_file = None
spacing = 0.25

rec_grid = grids.RecGrid(rec_prmtop_file, lj_sigma_scaling_factor, rec_inpcrd_file,
                        bsite_file,
                        grid_nc_file,
                        new_calculation=False,
                        spacing=spacing)
lig_prmtop_file = "../examples/amber/benzene/ligand.prmtop"
lig_inpcrd_file = "../examples/amber/benzene/ligand.inpcrd"
lig_grid = grids.LigGrid(lig_prmtop_file, lj_sigma_scaling_factor, lig_inpcrd_file, rec_grid)

print(lig_grid.get_initial_com())

def test_is_nc_grid_good():
    print("test", lig_grid.get_initial_com())
    assert grids.is_nc_grid_good(grid_nc_file) == True
#
#
# #Grid class tests
#
# def test_get_six_corner_shifts():
#     assert grids.Grid._get_six_corner_shifts()
#
# def test_set_grid_key_value():
#     assert grids.Grid._set_grid_key_value()
#
# def test_load_prmtop():
#     assert grids.Grid._load_prmtop()
#
# def test_load_inpcrd():
#     assert grids.Grid._load_inpcrd()
#
# def test_move_molecule_to():
#     assert grids.Grid._move_molecule_to()
#
# def test_get_molecule_center_of_mass():
#     assert grids.Grid._get_molecule_center_of_mass()
#
# def test_get_corner_crd():
#     assert grids.Grid._get_corner_crd()
#
# def test_get_uper_most_corner():
#     assert grids.Grid._get_uper_most_corner()
#
# def test_get_uper_most_corner_crd():
#     assert grids.Grid._get_uper_most_corner_crd()
#
# def test_get_origin_crd():
#     assert grids.Grid._get_origin_crd()
#
# def test_initialize_convenient_para():
#     assert grids.Grid._initialize_convenient_para()
#
# def test_is_in_grid():
#     assert grids.Grid.
#
# def test_distance():
#     assert grids.Grid.
#
# def test_containing_cube():
#     assert grids.Grid.
#
# def test_get_grid_func_names():
#     assert grids.Grid.
#
#
# # LigGrid class tests
#
# def test_cal_corr_func():
#     assert grids.LigGrid.
#
# def test_do_forward_fft():
#     assert grids.LigGrid.
#
# def test_cal_energies():
#     assert grids.LigGrid.
#
#
# def test_cal_energies_NOT_USED():
#     assert grids.LigGrid.
#
#
# def test_cal_meaningful_corners():
#     assert grids.LigGrid.
#
# def test_place_ligand_crd_in_grid():
#     assert grids.LigGrid.
#
# def test_cal_grids():
#     assert grids.LigGrid.
#
# def test_get_bpmf():
#     assert grids.LigGrid.
#
#
# def test_get_number_translations():
#     assert grids.LigGrid.
#
def test_get_box_volume():
    assert lig_grid.get_box_volume() == 130472.484375
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
#     assert grids.RecGrid.
#
# def test_write_to_nc():
#     assert grids.RecGrid.
#
#
# def test_cal_grid_parameters_with_bsite():
#     assert grids.RecGrid.
#
# def test_cal_grid_parameters_without_bsite():
#     assert grids.RecGrid.
#
# def test_move_receptor_to_grid_center():
#     assert grids.RecGrid.
#
# def test_cal_grid_coordinates():
#     assert grids.RecGrid.
#
#
# def test_get_charges():
#     assert grids.RecGrid.
#
# def test_cal_potential_grids():
#     assert grids.RecGrid.
#
#
# def test_exact_values():
#     assert grids.RecGrid.
#
# def test_trilinear_interpolation():
#     assert grids.RecGrid.
#
# def test_direct_energy():
#     assert grids.RecGrid.
#
# def test_interpolated_energy():
#     assert grids.RecGrid.
#
# def test_get_FFTs():
#     assert grids.RecGrid.
#
# def test_write_box():
#     assert grids.RecGrid.
#
# def test_write_pdb():
#     assert grids.RecGrid.
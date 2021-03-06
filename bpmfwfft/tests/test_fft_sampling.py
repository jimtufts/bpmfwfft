import pytest
import bpmfwfft.fft_sampling
import netCDF4
import pickle

from pathlib import Path

cwd = Path.cwd()
mod_path = Path(__file__).parent

rec_prmtop = (mod_path / "../../examples/amber/t4_lysozyme/receptor_579.prmtop").resolve()
lj_sigma_scal_fact = 0.8
rec_inpcrd = (mod_path / "../../examples/amber/t4_lysozyme/receptor_579.inpcrd").resolve()

bsite_file = (mod_path / "../../examples/amber/t4_lysozyme/measured_binding_site.py").resolve()
grid_nc_file = (mod_path / "../../examples/grid/t4_lysozyme/grid.nc").resolve()

lig_prmtop = (mod_path / "../../examples/amber/benzene/ligand.prmtop").resolve()
lig_inpcrd = (mod_path / "../../examples/amber/benzene/ligand.inpcrd").resolve()

energy_sample_size_per_ligand = 500
output_nc = (mod_path / "../../examples/fft_sampling/t4_benzene/fft_sampling.nc").resolve()

ligand_md_trj_file = (mod_path / "../../examples/ligand_md/benzene/trajectory.nc").resolve()
lig_coord_ensemble = netCDF4.Dataset(ligand_md_trj_file, "r").variables["positions"][:]


Sampling_test = bpmfwfft.fft_sampling.Sampling(rec_prmtop, lj_sigma_scal_fact, rec_inpcrd,
                        bsite_file, grid_nc_file,
                        lig_prmtop, lig_inpcrd,
                        lig_coord_ensemble,
                        energy_sample_size_per_ligand,
                        output_nc,
                        temperature=300.)

#def test_create_rec_grid():
#    test_rec_grid = bpmfwfft.fft_sampling.Sampling._create_rec_grid(Sampling_test, rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file, grid_nc_file)
#    ref_rec_grid = pickle.load(open( "rec_grid.pickle", "rb" ))
#    assert test_rec_grid is ref_rec_grid
    
#    assert fft_sampling.Sampling._create_rec_grid(Sampling_test, rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file,
#                                                  grid_nc_file) == 5

# def test_create_lig_grid():
#     assert fft_sampling.Sampling._create_lig_grid(lig_prmtop, lj_sigma_scal_fact, lig_inpcrd, fft_sampling.Sampling._create_rec_grid(rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file, grid_nc_file))
#
# def test_load_ligand_coor_ensemble():
#     assert fft_sampling.Sampling._load_ligand_coor_ensemble()
#
# def test_initialize_nc():
#     assert fft_sampling.Sampling._initialize_nc()
#
# def test_write_grid_info():
#     assert fft_sampling.Sampling._write_grid_info()
#
# def test_save_data_to_nc():
#     assert fft_sampling.Sampling._save_data_to_nc()
#
# def test_do_fft():
#     assert fft_sampling.Sampling._test_do_fft()
#
# def test_run_sampling():
#     assert fft_sampling.Sampling.run_sampling()

# def test_
#     assert fft_sampling.Sampling

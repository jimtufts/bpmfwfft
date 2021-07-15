import pytest
import bpmfwfft.fft_sampling
import netCDF4

rec_prmtop = "../examples/amber/t4_lysozyme/receptor_579.prmtop"
lj_sigma_scal_fact = 0.8
rec_inpcrd = "../examples/amber/t4_lysozyme/receptor_579.inpcrd"

bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
grid_nc_file = "../examples/grid/t4_lysozyme/grid.nc"

lig_prmtop = "../examples/amber/benzene/ligand.prmtop"
lig_inpcrd = "../examples/amber/benzene/ligand.inpcrd"

energy_sample_size_per_ligand = 500
output_nc = "../examples/fft_sampling/t4_benzene/fft_sampling.nc"

ligand_md_trj_file = "../examples/ligand_md/benzene/trajectory.nc"
lig_coord_ensemble = netCDF4.Dataset(ligand_md_trj_file, "r").variables["positions"][:]


Sampling_test = fft_sampling.Sampling(rec_prmtop, lj_sigma_scal_fact, rec_inpcrd,
                        bsite_file, grid_nc_file,
                        lig_prmtop, lig_inpcrd,
                        lig_coord_ensemble,
                        energy_sample_size_per_ligand,
                        output_nc,
                        temperature=300.)

def test_create_rec_grid():
    test_quick = fft_sampling.Sampling._create_rec_grid(Sampling_test, rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file,
                                                  grid_nc_file)
    print('hello', test_quick.get_crd())
#    assert fft_sampling.Sampling._create_rec_grid(Sampling_test, rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file,
                                                  grid_nc_file) == 5

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

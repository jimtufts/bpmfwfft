"""
This is to generate interaction energies and corresponding translational vectors,
given a fixed receptor and an ensemble of ligand coordinates (including rotations and/or configurations)
"""
from __future__ import print_function

import numpy as np
import netCDF4
import os
import time
import multiprocessing

try:
    from bpmfwfft.grids import RecGrid
    from bpmfwfft.grids import LigGrid

except:
    from grids import RecGrid
    from grids import LigGrid

KB = 0.001987204134799235  # kcal/mol*K

omp_num_threads = int(os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count()))

class Sampling(object):
    def __init__(self, rec_prmtop, lj_sigma_scal_fact,
                 rc_scale, rs_scale, rm_scale,
                 lc_scale, ls_scale, lm_scale,
                 rho,
                 rec_inpcrd, bsite_file,
                 grid_nc_file,
                 lig_prmtop, lig_inpcrd,
                 lig_coord_ensemble,
                 energy_sample_size_per_ligand,
                 output_nc,
                 start_index,
                 temperature=300.):
        """
        :param rec_prmtop: str, name of receptor prmtop file
        :param lj_sigma_scal_fact: float, used to check consitency when loading receptor and ligand grids
        :param rec_inpcrd: str, name of receptor inpcrd file
        :param bsite_file: None or str, name of file defining the box, the same as
        from AlGDock pipeline. "measured_binding_site.py"
        :param grid_nc_file: str, name of receptor precomputed grid netCDF file
        :param lig_prmtop: str, name of ligand prmtop file
        :param lig_inpcrd: str, name of ligand inpcrd file
        :param lig_coord_ensemble: list of 2d array, each array is an ligand coordinate
        :param energy_sample_size_per_ligand: int, number of energies and translational vectors to store for each ligand crd
        :param output_nc: str, name of nc file
        :param temperature: float
        """
        self._energy_sample_size_per_ligand = energy_sample_size_per_ligand
        self._beta = 1. / temperature / KB

        rec_grid = self._create_rec_grid(rec_prmtop, lj_sigma_scal_fact, rc_scale,
                                         rs_scale, rm_scale, rho, rec_inpcrd,
                                         bsite_file, grid_nc_file)
        self._rec_grid_displacement = rec_grid._displacement
        self._rec_crd = rec_grid.get_crd()

        self._lig_grid = self._create_lig_grid(lig_prmtop, lj_sigma_scal_fact,
                                               lc_scale, ls_scale, lm_scale,
                                               lig_inpcrd, rec_grid)

        self._lig_coord_ensemble = self._load_ligand_coor_ensemble(lig_coord_ensemble)
        self._start_index = start_index
        self._nc_handle = self._initialize_nc(output_nc)

        self._resampled_energies_components = {}
        self._resampled_trans_vectors_components = {}

    def _create_rec_grid(self, rec_prmtop, lj_sigma_scal_fact,
                         rc_scale, rs_scale, rm_scale, rho,
                         rec_inpcrd, bsite_file, grid_nc_file):
        rec_grid = RecGrid(rec_prmtop, lj_sigma_scal_fact, rc_scale, rs_scale, rm_scale,
                           rho, rec_inpcrd, bsite_file, grid_nc_file, new_calculation=False)
        return rec_grid

    def _create_lig_grid(self, lig_prmtop, lj_sigma_scal_fact, lc_scale, ls_scale, lm_scale,
                         lig_inpcrd, rec_grid):
        lig_grid = LigGrid(lig_prmtop, lj_sigma_scal_fact, lc_scale, ls_scale, lm_scale, lig_inpcrd, rec_grid)
        return lig_grid

    def _load_ligand_coor_ensemble(self, lig_coord_ensemble):
        assert len(lig_coord_ensemble.shape) == 3, "lig_coord_ensemble must be 3-D array."
        ensemble = lig_coord_ensemble
        natoms = self._lig_grid.get_natoms()

        for i in range(len(ensemble)):
            if (ensemble[i].shape[0] != natoms) or (ensemble[i].shape[1] != 3):
                raise RuntimeError("Ligand crd %d does not have correct shape" % i)
        return ensemble

    def _initialize_nc(self, output_nc):
        if not os.path.exists(output_nc):
            nc_handle = netCDF4.Dataset(output_nc, mode="w", format="NETCDF4")

            nc_handle.createDimension("three", 3)
            nc_handle.createDimension("one", 1)
            rec_natoms = self._rec_crd.shape[0]
            nc_handle.createDimension("rec_natoms", rec_natoms)

            lig_natoms = self._lig_grid.get_natoms()
            nc_handle.createDimension("lig_natoms", lig_natoms)
            # nc_handle.createDimension("lig_sample_size", self._lig_coord_ensemble.shape[0])
            nc_handle.createDimension("lig_sample_size", None)

            nc_handle.createDimension("energy_sample_size_per_ligand", self._energy_sample_size_per_ligand)

            nc_handle.createVariable("rec_positions", "f8", ("rec_natoms", "three"))
            nc_handle.variables["rec_positions"][:, :] = self._rec_crd

            nc_handle.createVariable("lig_positions", "f8", ("lig_sample_size", "lig_natoms", "three"))
            nc_handle.createVariable("lig_com", "f8", ("lig_sample_size", "three"))
            nc_handle.createVariable("volume", "f8", ("lig_sample_size"))
            nc_handle.createVariable("nr_grid_points", "i8", ("lig_sample_size"))

            nc_handle.createVariable("exponential_sums", "f8", ("lig_sample_size"))
            nc_handle.createVariable("log_of_divisors", "f8", ("lig_sample_size"))

            nc_handle.createVariable("mean_energy", "f8", ("lig_sample_size"))
            nc_handle.createVariable("min_energy", "f8", ("lig_sample_size"))
            nc_handle.createVariable("energy_std", "f8", ("lig_sample_size"))

            nc_handle.createVariable("resampled_energies", "f8", ("lig_sample_size", "energy_sample_size_per_ligand"))
            nc_handle.createVariable("resampled_trans_vectors", "i8",
                                     ("lig_sample_size", "energy_sample_size_per_ligand", "three"))

            nc_handle.createVariable("native_pose_energy", "f8", ("one"))
            nc_handle.createVariable("native_crd", "f8", ("lig_natoms", "three"))
            nc_handle.createVariable("native_translation", "i8", ("three"))

            nc_handle.createVariable(f"LJ_resampled_energies", "f8", ("lig_sample_size", "energy_sample_size_per_ligand"))
            nc_handle.createVariable(f"LJ_resampled_trans_vectors", "i8",
                                     ("lig_sample_size", "energy_sample_size_per_ligand", "three"))
            nc_handle.createVariable(f"LJ_native_pose_energy", "f8", ("one"))

            nc_handle.createVariable(f"no_sasa_resampled_energies", "f8",
                                     ("lig_sample_size", "energy_sample_size_per_ligand"))
            nc_handle.createVariable(f"no_sasa_resampled_trans_vectors", "i8",
                                     ("lig_sample_size", "energy_sample_size_per_ligand", "three"))
            nc_handle.createVariable(f"no_sasa_native_pose_energy", "f8", ("one"))

            nc_handle.createVariable(f"sasa_resampled_energies", "f8", ("lig_sample_size", "energy_sample_size_per_ligand"))
            nc_handle.createVariable(f"sasa_resampled_trans_vectors", "i8",
                                     ("lig_sample_size", "energy_sample_size_per_ligand", "three"))
            nc_handle.createVariable(f"sasa_native_pose_energy", "f8", ("one"))

            nc_handle.createVariable(f"current_rotation_index", "i8", ("one"))

            nc_handle.set_auto_mask(False)

            nc_handle = self._write_grid_info(nc_handle)

        else:
            print(f"{output_nc} exists, opening in append mode.")
            nc_handle = netCDF4.Dataset(output_nc, mode="a", format="NETCDF4")

        return nc_handle
    def _write_grid_info(self, nc_handle):
        """
        write grid info, "x", "y", "z" ...
        """
        data = self._lig_grid.get_grids()
        grid_func_names = self._lig_grid.get_grid_func_names()
        keys = [key for key in data.keys() if key not in grid_func_names]

        for key in keys:
            for dim in data[key].shape:
                dim_name = "%d" % dim
                if dim_name not in nc_handle.dimensions.keys():
                    nc_handle.createDimension(dim_name, dim)

        for key in keys:
            if data[key].dtype == int:
                store_format = "i8"
            elif data[key].dtype == float:
                store_format = "f8"
            else:
                raise RuntimeError("Unsupported dtype %s" % data[key].dtype)
            dimensions = tuple(["%d" % dim for dim in data[key].shape])
            nc_handle.createVariable(key, store_format, dimensions)

        for key in keys:
            nc_handle.variables[key][:] = data[key]
        return nc_handle

    def _save_data_to_nc(self, step):
        step = step + self._start_index
        if step == 0:
            if self._nc_handle.variables["native_pose_energy"][:][0] >= np.iinfo(np.int64).max:
                self._nc_handle.variables["native_pose_energy"][:] = np.array(self._lig_grid._native_pose_energy)
                print("Native pose energy", self._lig_grid._native_pose_energy)
                self._nc_handle.variables["native_crd"][:, :] = self._lig_grid.get_crd()
                self._nc_handle.variables["native_translation"][:] = self._native_translation
                print("Native translation", self._native_translation)
        self._nc_handle.variables["lig_positions"][step, :, :] = self._lig_grid.get_crd()

        self._nc_handle.variables["lig_com"][step, :] = self._lig_grid.get_initial_com()

        self._nc_handle.variables["volume"][step] = self._lig_grid.get_box_volume()

        self._nc_handle.variables["nr_grid_points"][step] = self._lig_grid.get_number_translations()

        self._nc_handle.variables["exponential_sums"][step] = self._exponential_sum

        self._nc_handle.variables["log_of_divisors"][step] = self._log_of_divisor

        self._nc_handle.variables["mean_energy"][step] = self._mean_energy
        self._nc_handle.variables["min_energy"][step] = self._min_energy
        self._nc_handle.variables["energy_std"][step] = self._energy_std

        self._nc_handle.variables["resampled_energies"][step, :] = self._resampled_energies

        self._nc_handle.variables["resampled_trans_vectors"][step, :, :] = self._resampled_trans_vectors

        self._nc_handle.variables["current_rotation_index"][0] = step + 1

        return None

    def _save_sub_data_to_nc(self, name, step):
        # if step == 0:
        #     self._nc_handle.variables["native_pose_energy"][:] = np.array(self._lig_grid._native_pose_energy)
        #     print("Native pose energy", self._lig_grid._native_pose_energy)
        step = step + self._start_index
        if name == "sasa":
            self._nc_handle.variables[f"{name}_resampled_energies"][step, :] = self._resampled_energies_components[name]
            self._nc_handle.variables[f"{name}_resampled_trans_vectors"][step, :, :] = \
            self._resampled_trans_vectors_components[name]
        else:
            self._nc_handle.variables[f"{name}_resampled_energies"][step, :] = self._resampled_energies_components[name]
            self._nc_handle.variables[f"{name}_resampled_trans_vectors"][step, :, :] = \
            self._resampled_trans_vectors_components[name]
        return None

    def _cal_free_of_clash(self):
        self._lig_grid._max_i, self._lig_grid._max_j, self._lig_grid._max_k = self._lig_grid._max_grid_indices
        corr_func = self._lig_grid._cal_corr_func("occupancy")
        self._lig_grid._free_of_clash = (corr_func < 0.001)
        self._lig_grid._free_of_clash = self._lig_grid._free_of_clash[0:self._lig_grid._max_i, 0:self._lig_grid._max_j,
                                        0:self._lig_grid._max_k]  # exclude positions where ligand crosses border
        del corr_func
        print("Ligand positions excluding border crossers", self._lig_grid._free_of_clash.shape)

        return None

    def _remove_nonphysical_energies(self, grid):
        max_i, max_j, max_k = self._lig_grid._max_i, self._lig_grid._max_j, self._lig_grid._max_k  # self._lig_grid._max_grid_indices
        grid = grid[0:max_i, 0:max_j, 0:max_k]  # exclude positions where ligand crosses border
        grid = grid[self._lig_grid._free_of_clash[0:max_i, 0:max_j, 0:max_k]]  # only include positions with no clash
        return grid

    def _cal_energies(self, name, step):
        max_i, max_j, max_k = self._lig_grid._max_grid_indices
        if np.any(self._lig_grid._free_of_clash[0:max_i, 0:max_j, 0:max_k]):
            grid_energy = np.zeros((self._lig_grid._max_grid_indices))
            if name in ["electrostatic", "LJa", "LJr"]:
                grid_energy = self._lig_grid._cal_corr_func(name)
                # grid_energy = self._remove_nonphysical_energies(grid_energy)
                self._lig_grid._meaningful_energies += grid_energy
            elif name == "sasa":
                grid_energy = self._lig_grid._cal_delta_sasa_func(self._lig_grid._free_of_clash)
                grid_energy = grid_energy * -self._lig_grid.get_gamma()
                # grid_energy = self._remove_nonphysical_energies(grid_energy)
                self._lig_grid._meaningful_energies += grid_energy
            # save component energies for sasa, LJ, total without sasa
            if name == "sasa":
                grid_energy = self._remove_nonphysical_energies(grid_energy)
                sel_ind = np.argsort(grid_energy)[:self._energy_sample_size_per_ligand]
                self._resampled_energies_components[name] = [grid_energy[ind] for ind in sel_ind]
                trans_vectors = self._lig_grid.get_meaningful_corners_comp()
                self._resampled_trans_vectors_components[name] = [trans_vectors[ind] for ind in sel_ind]
                del grid_energy
                del trans_vectors
                self._save_sub_data_to_nc(name, step)
            elif name in ["LJa", "electrostatic"]:
                grid_energy = self._lig_grid._meaningful_energies.copy()
                grid_energy = self._remove_nonphysical_energies(grid_energy)
                sel_ind = np.argsort(grid_energy)[:self._energy_sample_size_per_ligand]
                if name == "LJa":
                    name = "LJ"
                elif name == "electrostatic":
                    name = "no_sasa"
                self._resampled_energies_components[name] = [grid_energy[ind] for ind in sel_ind]
                trans_vectors = self._lig_grid.get_meaningful_corners_comp()
                self._resampled_trans_vectors_components[name] = [trans_vectors[ind] for ind in sel_ind]
                del grid_energy
                del trans_vectors
                self._save_sub_data_to_nc(name, step)

    def _do_fft(self, step):
        start_time_fft = time.time()
        print(f"Doing FFT for step {self._start_index + step}, with {omp_num_threads} threads")
        lig_conf = self._lig_coord_ensemble[step]
        self._lig_grid._place_ligand_crd_in_grid(molecular_coord=lig_conf)
        self._cal_free_of_clash()
        self._lig_grid._meaningful_energies = np.zeros(self._lig_grid._grid["counts"], dtype=float)
        names = [name for name in self._lig_grid._grid_func_names if name not in ["occupancy", "water"]]
        for name in names:
            self._cal_energies(name, step)

        energies = self._lig_grid.get_meaningful_energies()
        # energies = self._remove_nonphysical_energies
        i_max, j_max, k_max = self._lig_grid._max_grid_indices
        energies = energies[0:i_max, 0:j_max, 0:k_max]
        energies = energies[self._lig_grid._free_of_clash]
        print("Energies shape:", energies.shape)

        self._mean_energy = energies.mean()
        self._min_energy = energies.min()
        self._min_energy_ind = np.argmax(self._min_energy)
        self._energy_std = energies.std()
        print("Number of finite energy samples", energies.shape[0])

        exp_energies = -self._beta * energies
        print(f"Max exp energy {exp_energies.max()}, Min exp energy {exp_energies.min()}")
        # print out bottom 5 lowest energies
        self._log_of_divisor = exp_energies.max()
        exp_energies = np.exp(exp_energies - self._log_of_divisor)
        self._exponential_sum = exp_energies.sum()
        exp_energies /= self._exponential_sum
        print("Number of exponential energy samples", exp_energies.sum())
        self._lig_grid._number_of_meaningful_energies = energies.flatten().shape[0]
        sel_ind = np.argsort(energies)[:self._energy_sample_size_per_ligand]
        del exp_energies
        self._resampled_energies = [energies[ind] for ind in sel_ind]
        del energies

        trans_vectors = self._lig_grid.get_meaningful_corners()
        self._resampled_trans_vectors = [trans_vectors[ind] for ind in sel_ind]
        del trans_vectors
        if step == 0:
            # get crystal pose here, use i,j,k of crystal pose
            self._native_translation = ((self._rec_grid_displacement - self._lig_grid._new_displacement) / self._lig_grid._spacing).astype(int)
            in_bounds = self._native_translation < (i_max, j_max, k_max)
            
            if np.all(in_bounds):
                self._lig_grid._native_pose_energy = self._lig_grid._meaningful_energies[0:i_max, 0:j_max, 0:k_max][
                    self._native_translation[0], self._native_translation[1],
                    self._native_translation[2]]
        self._lig_grid.set_meaningful_energies_to_none()
        self._resampled_energies = np.array(self._resampled_energies, dtype=float)
        self._resampled_trans_vectors = np.array(self._resampled_trans_vectors, dtype=int)

        self._save_data_to_nc(step)

        print(f"--- FFT step {step} calculated in {(time.time() - start_time_fft)} seconds ---", flush=True)

        return None

    def _do_fft_old(self, step):
        print("Doing FFT for step %d" % step, "test")
        lig_conf = self._lig_coord_ensemble[step]
        self._lig_grid.cal_grids(molecular_coord=lig_conf)

        energies = self._lig_grid.get_meaningful_energies()
        print("Energies shape:", energies.shape)

        self._mean_energy = energies.mean()
        self._min_energy = energies.min()
        self._min_energy_ind = np.argmax(self._min_energy)
        self._energy_std = energies.std()
        print("Number of finite energy samples", energies.shape[0])

        exp_energies = -self._beta * energies
        print(f"Max exp energy {exp_energies.max()}, Min exp energy {exp_energies.min()}")
        # print out bottom 5 lowest energies

        self._log_of_divisor = exp_energies.max()
        exp_energies = np.exp(exp_energies - self._log_of_divisor)
        self._exponential_sum = exp_energies.sum()
        exp_energies /= self._exponential_sum
        print("Number of exponential energy samples", exp_energies.sum())
        # sel_ind = np.random.choice(exp_energies.shape[0], size=self._energy_sample_size_per_ligand, p=exp_energies, replace=True)
        # try:
        #     sel_ind = np.random.choice(exp_energies.shape[0], size=self._energy_sample_size_per_ligand, p=exp_energies, replace=False)
        # except:
        # print(f"Only {np.count_nonzero(exp_energies)} non-zero entries in p, falling back to {self._energy_sample_size_per_ligand} lowest energies")
        sel_ind = np.argsort(energies)[:self._energy_sample_size_per_ligand]

        del exp_energies
        self._resampled_energies = [energies[ind] for ind in sel_ind]
        del energies
        self._lig_grid.set_meaningful_energies_to_none()

        trans_vectors = self._lig_grid.get_meaningful_corners()
        self._resampled_trans_vectors = [trans_vectors[ind] for ind in sel_ind]
        del trans_vectors

        self._resampled_energies = np.array(self._resampled_energies, dtype=float)
        self._resampled_trans_vectors = np.array(self._resampled_trans_vectors, dtype=int)

        self._save_data_to_nc(step)
        return None

    def run_sampling(self):
        """
        """
        for step in range(self._lig_coord_ensemble.shape[0]):
            self._do_fft(step)

            print("Min energy", self._min_energy, "Index", self._min_energy_ind)
            print("Mean energy", self._mean_energy)
            print("STD energy", self._energy_std)
            print("Initial center of mass", self._lig_grid.get_initial_com())
            print("Grid volume", self._lig_grid.get_box_volume())
            print("Number of translations", self._lig_grid.get_number_translations())
            print("-------------------------------\n\n")

        self._nc_handle.close()
        return None


#
# TODO   the class above assumes that the resample size is smaller than number of meaningful energies
#       in general, the number of meaningful energies can be very small or even zero (no energy)
#       when the number of meaningful energies is zero, that stratum contributes n_points zeros to the exponential mean
#
#       so when needs to consider separately 3 cases:
#           len(meaningful energies) == 0
#           0< len(meaningful energies) <= resample size
#           len(meaningful energies) > resample size
#


class Sampling_PL(Sampling):

    def _write_data_key_2_nc(self, data, key):
        if data.shape[0] == 0:
            return None

        for dim in data.shape:
            dim_name = "%d" % dim
            if dim_name not in self._nc_handle.dimensions.keys():
                self._nc_handle.createDimension(dim_name, dim)

        if data.dtype == int:
            store_format = "i8"
        elif data.dtype == float:
            store_format = "f8"
        else:
            raise RuntimeError("unsupported dtype %s" % data.dtype)
        dimensions = tuple(["%d" % dim for dim in data.shape])
        self._nc_handle.createVariable(key, store_format, dimensions)

        self._nc_handle.variables[key][:] = data
        return None

    def _initialize_nc(self, output_nc):
        """
        """
        nc_handle = netCDF4.Dataset(output_nc, mode="w", format="NETCDF4")

        nc_handle.createDimension("three", 3)
        rec_natoms = self._rec_crd.shape[0]
        nc_handle.createDimension("rec_natoms", rec_natoms)

        lig_natoms = self._lig_grid.get_natoms()
        nc_handle.createDimension("lig_natoms", lig_natoms)
        nc_handle.createDimension("lig_sample_size", self._lig_coord_ensemble.shape[0])

        # nc_handle.createDimension("energy_sample_size_per_ligand", self._energy_sample_size_per_ligand)

        nc_handle.createVariable("rec_positions", "f8", ("rec_natoms", "three"))
        nc_handle.variables["rec_positions"][:, :] = self._rec_crd

        nc_handle.createVariable("lig_positions", "f8", ("lig_sample_size", "lig_natoms", "three"))
        nc_handle.createVariable("lig_com", "f8", ("lig_sample_size", "three"))
        nc_handle.createVariable("volume", "f8", ("lig_sample_size"))
        nc_handle.createVariable("nr_grid_points", "i8", ("lig_sample_size"))
        nc_handle.createVariable("nr_finite_energy", "i8", ("lig_sample_size"))

        nc_handle.createVariable("exponential_sums", "f8", ("lig_sample_size"))
        nc_handle.createVariable("log_of_divisors", "f8", ("lig_sample_size"))

        nc_handle.createVariable("mean_energy", "f8", ("lig_sample_size"))
        nc_handle.createVariable("min_energy", "f8", ("lig_sample_size"))
        nc_handle.createVariable("energy_std", "f8", ("lig_sample_size"))

        # nc_handle.createVariable("resampled_energies", "f8", ("lig_sample_size", "energy_sample_size_per_ligand"))
        # nc_handle.createVariable("resampled_trans_vectors", "i8", ("lig_sample_size", "energy_sample_size_per_ligand", "three"))

        nc_handle = self._write_grid_info(nc_handle)
        return nc_handle

    def _save_data_to_nc(self, step):
        self._nc_handle.variables["lig_positions"][step, :, :] = self._lig_grid.get_crd()

        self._nc_handle.variables["lig_com"][step, :] = self._lig_grid.get_initial_com()

        self._nc_handle.variables["volume"][step] = self._lig_grid.get_box_volume()

        self._nc_handle.variables["nr_grid_points"][step] = self._lig_grid.get_number_translations()

        self._nc_handle.variables["nr_finite_energy"][step] = self._nr_finite_energy

        self._nc_handle.variables["exponential_sums"][step] = self._exponential_sum

        self._nc_handle.variables["log_of_divisors"][step] = self._log_of_divisor

        self._nc_handle.variables["mean_energy"][step] = self._mean_energy

        self._nc_handle.variables["min_energy"][step] = self._min_energy

        self._nc_handle.variables["energy_std"][step] = self._energy_std

        self._write_data_key_2_nc(self._resampled_energies, "resampled_energies_%d" % step)

        self._write_data_key_2_nc(self._resampled_trans_vectors, "resampled_trans_vectors_%d" % step)
        return None

    def _do_fft(self, step):
        print("Doing FFT for step %d" % step)
        lig_conf = self._lig_coord_ensemble[step]
        self._lig_grid.cal_grids(molecular_coord=lig_conf)

        energies = self._lig_grid.get_meaningful_energies()
        self._nr_finite_energy = energies.shape[0]
        print("Number of finite energy samples", self._nr_finite_energy)

        if energies.shape[0] > 0:

            self._mean_energy = energies.mean()
            self._min_energy = energies.min()
            self._energy_std = energies.std()

            exp_energies = -self._beta * energies
            self._log_of_divisor = exp_energies.max()
            exp_energies = np.exp(exp_energies - self._log_of_divisor)
            self._exponential_sum = exp_energies.sum()
            exp_energies /= self._exponential_sum

            sample_size = min(exp_energies.shape[0], self._energy_sample_size_per_ligand)
            sel_ind = np.random.choice(exp_energies.shape[0], size=sample_size, p=exp_energies, replace=True)

            del exp_energies

            self._resampled_energies = [energies[ind] for ind in sel_ind]
            del energies
            self._lig_grid.set_meaningful_energies_to_none()

            trans_vectors = self._lig_grid.get_meaningful_corners()
            self._resampled_trans_vectors = [trans_vectors[ind] for ind in sel_ind]
            del trans_vectors

            self._resampled_energies = np.array(self._resampled_energies, dtype=float)
            self._resampled_trans_vectors = np.array(self._resampled_trans_vectors, dtype=int)

        else:

            self._mean_energy = np.inf
            self._min_energy = np.inf
            self._energy_std = np.inf

            self._log_of_divisor = 1.
            self._exponential_sum = 0.

            self._resampled_energies = np.array([], dtype=float)
            del energies
            self._lig_grid.set_meaningful_energies_to_none()

            self._resampled_trans_vectors = np.array([], dtype=float)

        self._save_data_to_nc(step)
        return None

if __name__ == "__main__":
    # test
    test_dir = f"/mnt/fft"
    rec_prmtop = f"{test_dir}/FFT_PPI/2.redock/1.amber/2OOB_A:B/receptor.prmtop"
    lj_sigma_scal_fact = 1.0
    rec_inpcrd = f"{test_dir}/FFT_PPI/2.redock/2.minimize/2OOB_A:B/receptor.inpcrd"

    # bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
    bsite_file = None
    grid_nc_file = f"{test_dir}/FFT_PPI/2.redock/4.receptor_grid/2OOB_A:B/grid_2oob.nc"
    # grid_nc_file = "/home/jim/Desktop/test_results/grid_2oob.nc"

    lig_prmtop = f"{test_dir}/FFT_PPI/2.redock/1.amber/2OOB_A:B/ligand.prmtop"
    lig_inpcrd = f"{test_dir}/FFT_PPI/2.redock/2.minimize/2OOB_A:B/ligand.inpcrd"

    energy_sample_size_per_ligand = 300
    output_nc = f"{test_dir}/FFT_PPI/2.redock/5.fft_sampling/2OOB_A:B/fft_sampling_maintest.nc"
    # output_nc = "/home/jim/Desktop/test_results/fft_2oob.nc"

    ligand_md_trj_file = f"{test_dir}/FFT_PPI/2.redock/3.ligand_rand_rot/2OOB_A:B/rotation.nc"
    if os.path.exists(output_nc):
        rot_index = netCDF4.Dataset(output_nc, "r").variables["current_rotation_index"][0]
    else:
        rot_index = 0
    # uncomment this to start over
    # rot_index = 0
    lig_coord_ensemble = netCDF4.Dataset(ligand_md_trj_file, "r").variables["positions"][rot_index : rot_index + 1]

    rec_grid = RecGrid(rec_prmtop, lj_sigma_scal_fact,
                       0.76, 0.53, 0.55, 9.0, rec_inpcrd,
                       bsite_file, grid_nc_file,False,0.5,3.0,"VDW_RADII", True)

    sampler = Sampling(rec_prmtop, lj_sigma_scal_fact,
                       0.76, 0.53, 0.55, 0.81, 0.50, 0.54, 9.0,
                       rec_inpcrd,
                        bsite_file, grid_nc_file,
                        lig_prmtop, lig_inpcrd,
                        lig_coord_ensemble,
                        energy_sample_size_per_ligand,
                        output_nc,
                        start_index=rot_index,
                        temperature=300.)
    sampler.run_sampling()






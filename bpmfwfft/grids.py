
from __future__ import print_function

import os
import re
import concurrent.futures

import numpy as np
import netCDF4
from mdtraj.geometry import _geometry
from mdtraj.geometry.sasa import _ATOMIC_RADII

try:
    from bpmfwfft import IO
    try:        
        from bpmfwfft.util import c_is_in_grid, cdistance, c_containing_cube
        from bpmfwfft.util import c_cal_charge_grid_new
        from bpmfwfft.util import c_cal_potential_grid
        from bpmfwfft.util import c_cal_lig_sasa_grid
        from bpmfwfft.util import c_cal_lig_sasa_grids
    except:
        from util import c_is_in_grid, cdistance, c_containing_cube
        from util import c_cal_charge_grid_new
        from util import c_cal_potential_grid
        from util import c_cal_lig_sasa_grid
        from util import c_cal_lig_sasa_grids

except:
    import IO
    from util import c_is_in_grid, cdistance, c_containing_cube
    from util import c_cal_charge_grid_new
    from util import c_cal_potential_grid
    from util import c_cal_lig_sasa_grid
    from util import c_cal_lig_sasa_grids

SASA_SLOPE = 2.127036
SASA_INTERCEPT = 111.86583367165035
# Gamma taken from amber manual
GAMMA = 0.005

def process_potential_grid_function(
        name,
        crd,
        origin_crd,
        grid_spacing,
        grid_counts,
        charges,
        prmtop_ljsigma,
        atom_list,
        molecule_sasa,
        rec_res_names,
        rec_core_scaling,
        rec_surface_scaling,
        rec_metal_scaling,
        rho,
        sasa_grid
):
    """
    gets called by cal_potential_grid and assigned to a new python process
    use cython to calculate electrostatic, LJa, LJr, SASAr, and SASAi grids
    and save them to nc file
    """
    print("calculating Receptor %s grid" % name)
    grid_x = np.linspace(
        origin_crd[0],
        origin_crd[0] + ((grid_counts[0]-1) * grid_spacing[0]),
        num=grid_counts[0]
    )
    grid_y = np.linspace(
        origin_crd[1],
        origin_crd[1] + ((grid_counts[1] - 1) * grid_spacing[1]),
        num=grid_counts[1]
    )
    grid_z = np.linspace(
        origin_crd[2],
        origin_crd[2] + ((grid_counts[2] - 1) * grid_spacing[2]),
        num=grid_counts[2]
    )
    uper_most_corner_crd = origin_crd + (grid_counts - 1.) * grid_spacing
    uper_most_corner = (grid_counts - 1)

    grid = c_cal_potential_grid(name, crd,
                                grid_x, grid_y, grid_z,
                                origin_crd, uper_most_corner_crd, uper_most_corner,
                                grid_spacing, grid_counts,
                                charges, prmtop_ljsigma, atom_list, molecule_sasa, rec_res_names,
                                rec_core_scaling, rec_surface_scaling, rec_metal_scaling,
                                rho, sasa_grid)
    return grid

def process_charge_grid_function(
        name,
        crd,
        origin_crd,
        grid_spacing,
        eight_corner_shifts,
        six_corner_shifts,
        grid_counts,
        charges,
        prmtop_ljsigma,
        molecule_sasa,
):
    """
    gets called by cal_charge_grid and assigned to a new python process
    use cython to calculate electrostatic, LJa, LJr, SASAr, and SASAi grids
    and save them to nc file
    """
    print("calculating Ligand %s grid" % name)
    grid_x = np.linspace(
        origin_crd[0],
        origin_crd[0] + ((grid_counts[0]-1) * grid_spacing[0]),
        num=grid_counts[0]
    )
    grid_y = np.linspace(
        origin_crd[1],
        origin_crd[1] + ((grid_counts[1] - 1) * grid_spacing[1]),
        num=grid_counts[1]
    )
    grid_z = np.linspace(
        origin_crd[2],
        origin_crd[2] + ((grid_counts[2] - 1) * grid_spacing[2]),
        num=grid_counts[2]
    )
    uper_most_corner_crd = origin_crd + (grid_counts - 1.) * grid_spacing
    uper_most_corner = (grid_counts - 1)

    grid = c_cal_charge_grid_new(name, crd,
                                grid_x, grid_y, grid_z,
                                origin_crd, uper_most_corner_crd, uper_most_corner,
                                grid_spacing, eight_corner_shifts, six_corner_shifts,
                                grid_counts, charges, prmtop_ljsigma, molecule_sasa)

    return grid

def is_nc_grid_good(nc_grid_file):
    """
    :param nc_grid_file: name of nc file
    :return: bool
    """
    if not os.path.exists(nc_grid_file):
        return False

    if os.path.getsize(nc_grid_file) == 0:
        return False

    nc_handle = netCDF4.Dataset(nc_grid_file, "r")
    nc_keys = nc_handle.variables.keys()
    grid_keys = Grid().get_allowed_keys()
    for key in grid_keys:
        if key not in nc_keys:
            return False
    return True

# def get_min_dists_pool():


class Grid(object):
    """
    an abstract class that defines some common methods and data attributes
    working implementations are in LigGrid and RecGrid below
    """
    def __init__(self):
        self._grid = {}
        self._grid_func_names   = ("SASAi", "electrostatic", "LJr", "LJa", "SASAr")  # calculate all grids
        # self._grid_func_names = ("SASAi", "SASAr")  # uncomment to only calculate SASA grids
        # self._grid_func_names = ()  # don't calculate any grids, but make grid objects for testing
        cartesian_axes  = ("x", "y", "z")
        box_dim_names   = ("d0", "d1", "d2")
        others          = ("spacing", "counts", "origin", "lj_sigma_scaling_factor", "rec_core_scaling",
                           "rec_surface_scaling", "rec_metal_scaling")
        self._grid_allowed_keys = self._grid_func_names + cartesian_axes + box_dim_names + others

        self._eight_corner_shifts = [np.array([i,j,k], dtype=int) for i in range(2) for j in range(2) for k in range(2)]
        self._eight_corner_shifts = np.array(self._eight_corner_shifts, dtype=int)

        self._six_corner_shifts = self._get_six_corner_shifts()

        self._nearest_neighbor_shifts = self._get_nearest_neighbor_shifts()

    def _get_six_corner_shifts(self):
        six_corner_shifts = []
        for i in [-1, 1]:
            six_corner_shifts.append(np.array([i,0,0], dtype=int))
            six_corner_shifts.append(np.array([0,i,0], dtype=int))
            six_corner_shifts.append(np.array([0,0,i], dtype=int))
        return np.array(six_corner_shifts, dtype=int)

    def _get_nearest_neighbor_shifts(self):
        nearest_neighbor_shifts = []
        for i in [-1, 1]:
            nearest_neighbor_shifts.append(np.array([i, 0, 0], dtype=int))
            nearest_neighbor_shifts.append(np.array([0, i, 0], dtype=int))
            nearest_neighbor_shifts.append(np.array([0, 0, i], dtype=int))
            nearest_neighbor_shifts.append(np.array([0, i, i], dtype=int))
            nearest_neighbor_shifts.append(np.array([i, 0, i], dtype=int))
            nearest_neighbor_shifts.append(np.array([i, i, 0], dtype=int))
        return np.array(nearest_neighbor_shifts, dtype=int)
    
    def _set_grid_key_value(self, key, value):
        """
        key:    str
        value:  any object
        """
        assert key in self._grid_allowed_keys, key + " is not an allowed key"

        if key not in self._grid_func_names:
            print(value)
        self._grid[key] = value
        return None
    
    def _load_prmtop(self, prmtop_file_name, lj_sigma_scaling_factor):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param lj_sigma_scaling_factor: float, must have value in [0.5, 1.0].
        It is stored in self._grid["lj_sigma_scaling_factor"] as
        a array of shape (1,) for reason of saving to nc file.
         Experience says that 0.8 is good for protein-ligand calculations.
        :return: None
        """
        assert 0.5 <= lj_sigma_scaling_factor <= 1.0, "lj_sigma_scaling_factor is out of allowed range"
        self._prmtop = IO.PrmtopLoad(prmtop_file_name).get_parm_for_grid_calculation()
        self._prmtop["LJ_SIGMA"] *= lj_sigma_scaling_factor
        self._set_grid_key_value("lj_sigma_scaling_factor", np.array([lj_sigma_scaling_factor], dtype=float))
        return None
    
    def _load_inpcrd(self, inpcrd_file_name):
        self._crd = IO.InpcrdLoad(inpcrd_file_name).get_coordinates()
        natoms = self._prmtop["POINTERS"]["NATOM"]
        if (self._crd.shape[0] != natoms) or (self._crd.shape[1] != 3):
            raise RuntimeError("coordinates in %s has wrong shape"%inpcrd_file_name)
        return None
    
    def _move_molecule_to(self, location):
        """
        Move the center of mass of the molecule to location.
        location:   3-array.
        This method affects self._crd.
        """
        assert len(location) == 3, "location must have len 3"
        displacement = np.array(location, dtype=float) - self._get_molecule_center_of_mass()
        for atom_ind in range(len(self._crd)):
            self._crd[atom_ind] += displacement
        return None
    
    def _get_molecule_center_of_mass(self):
        """
        return the center of mass of self._crd
        """
        center_of_mass = np.zeros([3], dtype=float)
        masses = self._prmtop["MASS"]
        for atom_ind in range(len(self._crd)):
            center_of_mass += masses[atom_ind] * self._crd[atom_ind]
        total_mass = masses.sum()
        if total_mass == 0:
            raise RuntimeError("zero total mass")
        return center_of_mass / total_mass

    def _get_molecule_sasa_nanometer(self, probe_radius, n_sphere_points):
        """
        return the per atom SASA of the target molecule in nanometers
        """
        xyz = self._crd
        xyz = np.expand_dims(xyz, 0)
        # convert coordinates to nanometers for mdtraj
        xyz = xyz.astype(np.float32)/10.
        atom_radii = self._prmtop["VDW_RADII"]/10.

        radii = np.array(atom_radii, np.float32) + probe_radius
        dim1 = xyz.shape[1]
        atom_mapping = np.arange(dim1, dtype=np.int32)
        out = np.zeros((xyz.shape[0], dim1), dtype=np.float32)
        _geometry._sasa(xyz, radii, int(n_sphere_points), atom_mapping, out)
        # convert values from nm^2 to A^2
        out = out
        return out

    def _get_molecule_sasa(self, probe_radius, n_sphere_points):
        """
        return the per atom SASA of the target molecule in Angstroms
        """
        xyz = self._crd
        xyz = np.expand_dims(xyz, 0)
        # convert coordinates to nanometers for mdtraj
        xyz = xyz.astype(np.float32)/10.
        atom_radii = self._prmtop["VDW_RADII"]/10.

        radii = np.array(atom_radii, np.float32) + probe_radius
        dim1 = xyz.shape[1]
        atom_mapping = np.arange(dim1, dtype=np.int32)
        out = np.zeros((xyz.shape[0], dim1), dtype=np.float32)
        _geometry._sasa(xyz, radii, int(n_sphere_points), atom_mapping, out)
        # convert values from nm^2 to A^2
        out = out*100.
        return out

    def _get_corner_crd(self, corner):
        """
        corner: 3-array integers
        """
        i, j, k = corner
        return np.array([self._grid["x"][i], self._grid["y"][j], self._grid["z"][k]] , dtype=float)
    
    def _get_uper_most_corner(self):
        return np.array(self._grid["counts"] - 1, dtype=int)
    
    def _get_uper_most_corner_crd(self):
        uper_most_corner = self._get_uper_most_corner()
        return self._get_corner_crd(uper_most_corner)
    
    def _get_origin_crd(self):
        return self._get_corner_crd([0,0,0])

    def _initialize_convenient_para(self):
        self._origin_crd           = self._get_origin_crd()
        self._uper_most_corner_crd = self._get_uper_most_corner_crd()
        self._uper_most_corner     = self._get_uper_most_corner()
        self._spacing              = np.array([self._grid["d%d"%i][i] for i in range(3)], dtype=float)
        return None

    def _is_in_grid(self, atom_coordinate):
        """
        in grid means atom_coordinate >= origin_crd and atom_coordinate < uper_most_corner_crd
        :param atom_coordinate: 3-array of float
        :return: bool
        """
        return c_is_in_grid(atom_coordinate, self._origin_crd, self._uper_most_corner_crd)
    
    def _distance(self, corner, atom_coordinate):
        """
        corner: 3-array int
        atom_coordinate:    3-array of float
        return distance from corner to atom coordinate
        """
        corner_crd = self._get_corner_crd(corner)
        return cdistance(atom_coordinate, corner_crd)
    
    def _containing_cube(self, atom_coordinate):
        eight_corners, nearest_ind, furthest_ind = c_containing_cube(atom_coordinate, self._origin_crd,
                                                                     self._uper_most_corner_crd,
                                                                     self._spacing, self._eight_corner_shifts,
                                                                     self._grid["x"], self._grid["y"], self._grid["z"])
        return eight_corners, nearest_ind, furthest_ind
    
    def _is_row_in_matrix(self, row, matrix):
        for r in matrix:
            if (row == r).all():
                return True
        return False

    def get_grid_func_names(self):
        return self._grid_func_names
    
    def get_grids(self):
        return self._grid
    
    def get_crd(self):
        return self._crd
    
    def get_prmtop(self):
        return self._prmtop
    
    def get_charges(self):
        charges = dict()
        for key in ["CHARGE_E_UNIT", "R_LJ_CHARGE", "A_LJ_CHARGE"]:
            charges[key] = self._prmtop[key]
        return charges

    def get_natoms(self):
        return self._prmtop["POINTERS"]["NATOM"]

    def get_allowed_keys(self):
        return self._grid_allowed_keys


 

class LigGrid(Grid):
    """
    Calculate the "charge" part of the interaction energy.
    """
    def __init__(self, prmtop_file_name, lj_sigma_scaling_factor,
                       lig_core_scaling, lig_surface_scaling, lig_metal_scaling,
                       inpcrd_file_name, receptor_grid):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param lj_sigma_scaling_factor: float
        :param lig_core_scaling: float
        :param lig_surface_scaling: float
        :param lig_metal_scaling: float
        :param inpcrd_file_name: str, name of AMBER coordinate file
        :param receptor_grid: an instance of RecGrid class.
        """
        Grid.__init__(self)
        grid_data = receptor_grid.get_grids()
        if grid_data["lj_sigma_scaling_factor"][0] != lj_sigma_scaling_factor:
            raise RuntimeError("lj_sigma_scaling_factor is %f but in receptor_grid, it is %f" %(
                                lj_sigma_scaling_factor, grid_data["lj_sigma_scaling_factor"][0]))
        
        entries = [key for key in grid_data.keys() if key not in self._grid_func_names]
        print("Copy entries from receptor_grid", entries)
        for key in entries:
            self._set_grid_key_value(key, grid_data[key])
        self._initialize_convenient_para()

        self._rec_FFTs = receptor_grid.get_FFTs()

        self._load_prmtop(prmtop_file_name, lj_sigma_scaling_factor)
        self._load_inpcrd(inpcrd_file_name)
        self._move_ligand_to_lower_corner()
        # self._molecule_sasa = self._get_molecule_sasa(0.14, 960)
        self._molecule_sasa = self._get_molecule_sasa(0.001, 960)
        self._lig_core_scaling = lig_core_scaling
        self._lig_surface_scaling = lig_surface_scaling
        self._lig_metal_scaling = lig_metal_scaling
        self._rho = receptor_grid.get_rho()


    def _move_ligand_to_lower_corner(self):
        """
        move ligand to near the grid lower corner 
        store self._max_grid_indices and self._initial_com
        """
        spacing = self._grid["spacing"]
        lower_ligand_corner = np.array([self._crd[:,i].min() for i in range(3)], dtype=float) - 2.5*spacing
        lower_ligand_corner_grid_aligned = lower_ligand_corner - (spacing + lower_ligand_corner % spacing) #new grid aligned variable
        upper_ligand_corner = np.array([self._crd[:,i].max() for i in range(3)], dtype=float) + 2.5*spacing
        upper_ligand_corner_grid_aligned = upper_ligand_corner + (spacing - upper_ligand_corner % spacing) #new grid aligned variable
        ligand_box_lengths = upper_ligand_corner_grid_aligned - lower_ligand_corner_grid_aligned

        if np.any(ligand_box_lengths < 0):
            raise RuntimeError("One of the ligand box lengths are negative")

        max_grid_indices = np.ceil(ligand_box_lengths / spacing)
        self._max_grid_indices = self._grid["counts"] - np.array(max_grid_indices, dtype=int)
        if np.any(self._max_grid_indices <= 1):
            raise RuntimeError("At least one of the max grid indices is <= one")

        displacement = self._origin_crd - lower_ligand_corner_grid_aligned #formerly lower_ligand_corner
        for atom_ind in range(len(self._crd)):
            self._crd[atom_ind] += displacement
        print(f"Ligand translated by {displacement}")
        self._displacement = displacement
        lower_corner_origin = np.array([self._crd[:,i].min() for i in range(3)], dtype=float) - 1.5*spacing
        print(lower_corner_origin)
        self._initial_com = self._get_molecule_center_of_mass()
        return None
    
    def _get_charges(self, name):
        assert name in self._grid_func_names, "%s is not allowed"%name

        if name == "electrostatic":
            return np.array(self._prmtop["CHARGE_E_UNIT"], dtype=float)
        elif name == "LJa":
            return np.array(self._prmtop["A_LJ_CHARGE"], dtype=float)
        elif name == "LJr":
            return np.array(self._prmtop["R_LJ_CHARGE"], dtype=float)
        elif name == "SASAi":
            return np.array([0], dtype=float)
        elif name == "SASAr":
            return np.array([0], dtype=float)
        else:
            raise RuntimeError("%s is unknown"%name)

    def _cal_charge_grid(self, name):
        charges = self._get_charges(name)
        grid_counts = np.copy(self._grid["counts"])
        grid = c_cal_charge_grid_new(name, self._crd, self._grid["x"], self._grid["y"], self._grid["z"],
                                self._origin_crd, self._uper_most_corner_crd, self._uper_most_corner,
                                self._grid["spacing"], self._eight_corner_shifts, self._six_corner_shifts,
                                grid_counts, charges, self._prmtop["LJ_SIGMA"],
                                self._molecule_sasa
                                )
        # task_divisor = 1
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     futures = {}
        #     grid_names = [name for name in self._grid_func_names if name[:4] != "SASA"]
        #     for name in grid_names:
        #         futures_array = []
        #         for i in range(task_divisor):
        #             counts = np.copy(self._grid["counts"])
        #             counts_x = counts[0] // task_divisor
        #             if i == task_divisor - 1:
        #                 counts_x += counts[0] % task_divisor
        #             counts[0] = counts_x
        #
        #             grid_start_x = i * (self._grid["counts"][0] // task_divisor)
        #             origin = np.copy(self._origin_crd)
        #             origin[0] = grid_start_x * self._grid["spacing"][0]
        #
        #             futures_array.append(executor.submit(
        #                 process_charge_grid_function,
        #                 name,
        #                 self._crd,
        #                 origin,
        #                 self._grid["spacing"],
        #                 self._eight_corner_shifts,
        #                 self._six_corner_shifts,
        #                 counts,
        #                 self._get_charges(name),
        #                 self._prmtop["LJ_SIGMA"],
        #                 self._molecule_sasa,
        #             ))
        #         futures[name] = futures_array
        #     for name in futures:
        #         grid_array = []
        #         for i in range(task_divisor):
        #             partial_grid = futures[name][i].result()
        #             grid_array.append(partial_grid)
        #         grid = np.concatenate(tuple(grid_array), axis=0)
        #         # self._write_to_nc(nc_handle, name, grid)
        #         # self._set_grid_key_value(name, grid)
        #         # self._set_grid_key_value(name, None)     # to save memory

        return grid

    def _cal_corr_func(self, grid_name):
        """
        :param grid_name: str
        :return: fft correlation function
        """
        assert grid_name in self._grid_func_names, "%s is not an allowed grid name"%grid_name

        grid = self._cal_charge_grid(grid_name)

        self._set_grid_key_value(grid_name, grid)
        corr_func = np.fft.fftn(self._grid[grid_name])
        self._set_grid_key_value(grid_name, None)           # to save memory

        corr_func = corr_func.conjugate()
        corr_func = np.fft.ifftn(self._rec_FFTs[grid_name] * corr_func)
        corr_func = np.real(corr_func)
        return corr_func

    def _cal_shape_complementarity(self):
        """
        :param grid_name: str
        :return: fft correlation function
        """
        print("Calculating shape complementarity.")
        counts = self._grid["counts"]

        lig_sasai_grid, lig_sasar_grid = self.get_SASA_grids()
        lig_sasa_grid = np.add(lig_sasar_grid, lig_sasai_grid*1.j)

        # self._set_grid_key_value(grid_name, lig_sasa_grid)
        # crucially takes the conjugate of the complex ligand grid BEFORE FFT
        corr_func = np.fft.fftn(lig_sasa_grid.conjugate())
        # self._set_grid_key_value(grid_name, None)           # to save memory

        rec_sasa_grid = self._rec_FFTs["SASA"]

        rec_sasa_fft = np.fft.fftn(rec_sasa_grid)

        corr_func = np.fft.ifftn(rec_sasa_fft * corr_func.conjugate())  # * (1/(np.prod(counts)))
        corr_func = np.real(np.real(corr_func) - np.imag(corr_func))
        # self._shape_complementarity_func = corr_func
        return corr_func

    def _do_forward_fft(self, grid_name):
        assert grid_name in self._grid_func_names, "%s is not an allowed grid name"%grid_name
        grid = self._cal_charge_grid(grid_name)
        self._set_grid_key_value(grid_name, grid)
        forward_fft = np.fft.fftn(self._grid[grid_name])
        self._set_grid_key_value(grid_name, None)           # to save memory
        return forward_fft

    def _cal_corr_funcs(self, grid_names):
        """
        :param grid_names: list of str
        :return:
        """
        assert type(grid_names) == list, "grid_names must be a list"

        grid_name = grid_names[0]
        forward_fft = self._do_forward_fft(grid_name)
        corr_func = self._rec_FFTs[grid_name] * forward_fft.conjugate()

        for grid_name in grid_names[1:]:
            forward_fft = self._do_forward_fft(grid_name)
            corr_func += self._rec_FFTs[grid_name] * forward_fft.conjugate()

        corr_func = np.fft.ifftn(corr_func)
        corr_func = np.real(corr_func)
        return corr_func

    def _cal_energies(self):
        """
        calculate interaction energies
        store self._meaningful_energies (1-array) and self._meaningful_corners (2-array)
        meaningful means no border-crossing and no clashing
        TODO
        """
        max_i, max_j, max_k = self._max_grid_indices
        corr_func = self._cal_shape_complementarity()
        print(f"Max Shape Complementarity Score: {corr_func.max()} Min: {corr_func.min()}")
        self._free_of_clash = (corr_func > 1) #& (corr_func < 4000)
        print("number of poses free of clash:", self._free_of_clash.shape)
        self._free_of_clash = self._free_of_clash[0:max_i, 0:max_j, 0:max_k]  # exclude positions where ligand crosses border
        print("Ligand positions excluding border crossers", self._free_of_clash.shape)
        self._meaningful_energies = np.zeros(self._grid["counts"], dtype=float)
        if np.any(self._free_of_clash):
            grid_names = [name for name in self._grid_func_names if name[:4] != "SASA"]
            for name in grid_names:
                grid_func_energy = self._cal_corr_func(name)
                print(f"{name} energy: {grid_func_energy[71][43][54]}")
                # testing new LJr calculation method #DONTFORGETME
                if name == 'LJa':
                    grid_func_energy[grid_func_energy > 0] = 0
                    self._meaningful_energies += grid_func_energy
                if name == 'LJr':
                    grid_func_energy[grid_func_energy < 0] = 0
                    self._meaningful_energies += grid_func_energy
                else:
                    self._meaningful_energies += grid_func_energy
                del grid_func_energy
            # Add in energy for buried surface area E=SA*GAMMA, SA = SC*SLOPE + B
            bsa_energy = ((corr_func * SASA_SLOPE) + SASA_INTERCEPT) * GAMMA
            self._meaningful_energies += bsa_energy
            print(f"shape complementarity score: {corr_func[71][43][54]}")
            print(f"buried surface area energy: {bsa_energy[71][43][54]}")
            del bsa_energy
        # get crystal pose here, use i,j,k of crystal pose
        self._meaningful_energies = self._meaningful_energies[0:max_i, 0:max_j, 0:max_k] # exclude positions where ligand crosses border
        
        self._meaningful_energies = self._meaningful_energies[self._free_of_clash]         # exclude positions where ligand is in clash with receptor, become 1D array
        self._number_of_meaningful_energies = self._meaningful_energies.shape[0]
        
        return None

    def _cal_energies_NOT_USED(self):
        """
        calculate interaction energies
        store self._meaningful_energies (1-array) and self._meaningful_corners (2-array)
        meaningful means no boder-crossing and no clashing
        TODO
        """
        max_i, max_j, max_k = self._max_grid_indices

        corr_func = self._cal_corr_func("occupancy")
        self._free_of_clash = (corr_func  < 0.001)
        self._free_of_clash = self._free_of_clash[0:max_i, 0:max_j, 0:max_k]  # exclude positions where ligand crosses border

        if np.any(self._free_of_clash):
            grid_names = [name for name in self._grid_func_names if name != "occupancy"]
            self._meaningful_energies = self._cal_corr_funcs(grid_names)
        else:
            self._meaningful_energies = np.zeros(self._grid["counts"], dtype=float)

        self._meaningful_energies = self._meaningful_energies[0:max_i, 0:max_j, 0:max_k] # exclude positions where ligand crosses border
        self._meaningful_energies = self._meaningful_energies[self._free_of_clash]         # exclude positions where ligand is in clash with receptor, become 1D array
        self._number_of_meaningful_energies = self._meaningful_energies.shape[0]
        return None
    
    def _cal_meaningful_corners(self):
        """
        return grid corners corresponding to self._meaningful_energies
        """
        corners = np.where(self._free_of_clash)
        corners = np.array(corners, dtype=int)
        corners = corners.transpose()
        return corners

    def _place_ligand_crd_in_grid(self, molecular_coord):
        """
        molecular_coord:    2-array, new ligand coordinate
        """
        crd = np.array(molecular_coord, dtype=float)
        natoms = self._prmtop["POINTERS"]["NATOM"]
        if (crd.shape[0] != natoms) or (crd.shape[1] != 3):
            raise RuntimeError("Input coord does not have the correct shape.")
        self._crd = crd
        self._move_ligand_to_lower_corner()
        return None

    def cal_grids(self, molecular_coord=None):
        """
        molecular_coord:    2-array, new ligand coordinate
        compute charge grids, meaningful_energies, meaningful_corners for molecular_coord
        if molecular_coord==None, self._crd is used
        """
        if molecular_coord is not None:
            self._place_ligand_crd_in_grid(molecular_coord)
        else:
            self._move_ligand_to_lower_corner()         # this is just in case the self._crd is not at the right position
        
        self._cal_energies()
        return None
    
    def get_bpmf(self, kB=0.001987204134799235, temperature=300.0):
        """
        use self._meaningful_energies to calculate and return exponential mean
        """
        if len(self._meaningful_energies) == 0:
            return 0.

        beta = 1. / temperature / kB
        V_0 = 1661.

        nr_samples = self.get_number_translations()
        energies = -beta *  self._meaningful_energies
        e_max = energies.max()
        exp_mean = np.exp(energies - e_max).sum() / nr_samples

        bpmf = -temperature * kB * (np.log(exp_mean) + e_max)

        V_binding = self.get_box_volume()
        correction = -temperature * kB * np.log(V_binding / V_0 / 8 / np.pi**2)
        return bpmf + correction
    
    def get_number_translations(self):
        return self._max_grid_indices.prod()
    
    def get_box_volume(self):
        """
        in angstrom ** 3
        """
        spacing = self._grid["spacing"]
        volume = ((self._max_grid_indices - 1) * spacing).prod()
        return volume
    
    def get_meaningful_energies(self):
        return self._meaningful_energies
    
    def get_meaningful_corners(self):
        meaningful_corners = self._cal_meaningful_corners()
        if meaningful_corners.shape[0] != self._number_of_meaningful_energies:
            raise RuntimeError("meaningful_corners does not have the same len as self._number_of_meaningful_energies")
        return meaningful_corners

    def set_meaningful_energies_to_none(self):
        self._meaningful_energies = None
        return None

    def get_initial_com(self):
        return self._initial_com

    def get_SASA_grids(self, radii_type="VDW_RADII", exclude_H=True):
        """
        Return the SASAi and SASAr grids for the Ligand
        radii_type: string, either "LJ_SIGMA" or "VDW_RADII"
        """
        atoms = []
        core_atoms = []
        surface_atoms = []
        core_ions = []
        surface_ions = []
        metal_ions = ["ZN", "CA", "MG", "SR"]
        atom_names = self._prmtop["PDB_TEMPLATE"]["ATOM_NAME"]
        res_names = self._prmtop["PDB_TEMPLATE"]["RES_NAME"]

        radii_type = radii_type.upper()

        assert radii_type in ["LJ_SIGMA", "VDW_RADII"], "Radii_type %s not allowed, use LJ_SIGMA or VDW_RADII" % radii_type

        # Select radii type, LJ_SIGMA is a diameter in this code
        if radii_type == "LJ_SIGMA":
            radii = self._prmtop[radii_type] / 2.
        elif radii_type == "VDW_RADII":
            radii = self._prmtop[radii_type]

        # Exclude Hydrogen if requested
        for i in range(self._crd.shape[0]):
            if exclude_H:
                if atom_names[i][0] != 'H':
                    atoms.append(i)
            else:
                atoms.append(i)

        # Sort atoms/ions by core/surface based on SASA
        for i in range(len(atoms)):
            atom_ind = atoms[i]
            if res_names[atom_ind] not in metal_ions:
                if self._molecule_sasa[0][atom_ind] < 0.1:
                    core_atoms.append(atom_ind)
                else:
                    surface_atoms.append(atom_ind)
            else:
                if self._molecule_sasa[0][atom_ind] < 0.1:
                    core_ions.append(atom_ind)
                else:
                    surface_ions.append(atom_ind)

        sasai_grid, sasar_grid = c_cal_lig_sasa_grids(self._crd,
                                                      self._grid["x"], self._grid["y"], self._grid["z"],
                                                      self._origin_crd, self._uper_most_corner_crd,
                                                      self._uper_most_corner,
                                                      self._grid["spacing"], self._nearest_neighbor_shifts,
                                                      self._grid["counts"],
                                                      radii, core_atoms, surface_atoms, core_ions, surface_ions,
                                                      self._lig_core_scaling, self._lig_surface_scaling,
                                                      self._lig_metal_scaling,
                                                      self._rho)
        return sasai_grid, sasar_grid

    def translate_ligand(self, displacement):
        """
        translate the ligand by displacement in Angstroms
        """
        for atom_ind in range(len(self._crd)):
            self._crd[atom_ind] += displacement
        return None

    def write_pdb(self, file_name, mode):
        IO.write_pdb(self._prmtop, self._crd, file_name, mode)
        return None


 
class RecGrid(Grid):
    """
    calculate the potential part of the interaction energy.
    """
    def __init__(self,  prmtop_file_name, lj_sigma_scaling_factor,
                        rec_core_scaling, rec_surface_scaling, rec_metal_scaling,
                        rho,
                        inpcrd_file_name,
                        bsite_file,
                        grid_nc_file,
                        new_calculation=False,
                        spacing=0.25, extra_buffer=3.0,
                        radii_type="VDW_RADII", exclude_H=True):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param lj_sigma_scaling_factor: float
        :param inpcrd_file_name: str, name of AMBER coordinate file
        :param bsite_file: str or None, if not None, name of a file defining the box dimension.
        This file is the same as "measured_binding_site.py" from AlGDock pipeline.
        :param grid_nc_file: str, name of grid nc file
        :param new_calculation: bool, if True do the new grid calculation else load data in grid_nc_file.
        :param spacing: float and in angstrom.
        :param extra_buffer: float
        """
        Grid.__init__(self)
        self._load_prmtop(prmtop_file_name, lj_sigma_scaling_factor)
        self._FFTs = {}

        if new_calculation:
            self._load_inpcrd(inpcrd_file_name)
            self._molecule_sasa = self._get_molecule_sasa(0.14, 960)
            self._rho = rho
            self._rec_core_scaling = rec_core_scaling
            self._rec_surface_scaling = rec_surface_scaling
            self._rec_metal_scaling = rec_metal_scaling
            nc_handle = netCDF4.Dataset(grid_nc_file, "w", format="NETCDF4")
            self._write_to_nc(nc_handle, "lj_sigma_scaling_factor", 
                                np.array([lj_sigma_scaling_factor], dtype=float))
            self._write_to_nc(nc_handle, "rec_core_scaling",
                              np.array([rec_core_scaling], dtype=float))
            self._write_to_nc(nc_handle, "rec_surface_scaling",
                              np.array([rec_surface_scaling], dtype=float))
            self._write_to_nc(nc_handle, "rec_metal_scaling",
                              np.array([rec_metal_scaling], dtype=float))
            self._write_to_nc(nc_handle, "rho",
                              np.array([rho], dtype=float))
            self._write_to_nc(nc_handle, "molecule_sasa",
                              np.array(self._molecule_sasa, dtype=float))

            if bsite_file is not None:
                print("Receptor is assumed to be correctly translated such that box encloses binding pocket.")
                self._cal_grid_parameters_with_bsite(spacing, bsite_file, nc_handle)
                self._cal_grid_coordinates(nc_handle)
                self._initialize_convenient_para()
            else:
                print("No binding site specified, box encloses the whole receptor")
                self._cal_grid_parameters_without_bsite(spacing, extra_buffer, nc_handle)
                self._cal_grid_coordinates(nc_handle)
                self._initialize_convenient_para()
                self._move_receptor_to_grid_center()
                self._write_to_nc(nc_handle, "displacement", self._displacement)

            self._cal_potential_grids(nc_handle, radii_type, exclude_H)
            self._write_to_nc(nc_handle, "trans_crd", self._crd)
            nc_handle.close()
                
        self._load_precomputed_grids(grid_nc_file, lj_sigma_scaling_factor)

    def _load_precomputed_grids(self, grid_nc_file, lj_sigma_scaling_factor):
        """
        nc_file_name:   str
        lj_sigma_scaling_factor: float, used for consistency check
        load netCDF file, populate self._grid with all the data fields 
        """
        assert os.path.isfile(grid_nc_file), "%s does not exist" %grid_nc_file

        print(grid_nc_file)
        nc_handle = netCDF4.Dataset(grid_nc_file, "r")
        keys = [key for key in self._grid_allowed_keys if key not in self._grid_func_names]
        for key in keys:
            self._set_grid_key_value(key, nc_handle.variables[key][:])

        if self._grid["lj_sigma_scaling_factor"][0] != lj_sigma_scaling_factor:
            raise RuntimeError("lj_sigma_scaling_factor is %f but in %s, it is %f" %(
                lj_sigma_scaling_factor, grid_nc_file, self._grid["lj_sigma_scaling_factor"][0]))

        self._initialize_convenient_para()

        natoms = self._prmtop["POINTERS"]["NATOM"]
        if natoms != nc_handle.variables["trans_crd"].shape[0]:
            raise RuntimeError("Number of atoms is wrong in %s %nc_file_name")
        self._crd = nc_handle.variables["trans_crd"][:]
        self._rho = nc_handle.variables["rho"][:]

        for key in self._grid_func_names:
            if key[:4] != "SASA":
                self._set_grid_key_value(key, nc_handle.variables[key][:])
                self._FFTs[key] = self._cal_FFT(key)
                self._set_grid_key_value(key, None)     # to save memory
        self._set_grid_key_value("SASAi", nc_handle.variables["SASAi"][:])  #UNCOMMENT ME
        self._set_grid_key_value("SASAr", nc_handle.variables["SASAr"][:])  #UNCOMMENT ME
        self._FFTs["SASA"] = self._cal_SASA_FFT()  #UNCOMMENT ME
        self._set_grid_key_value("SASAi", None)  #UNCOMMENT ME
        self._set_grid_key_value("SASAr", None)  #UNCOMMENT ME
        nc_handle.close()
        return None

    def _cal_FFT(self, name):
        if name not in self._grid_func_names:
            raise RuntimeError("%s is not allowed.")
        print("Doing FFT for %s"%name)
        FFT = np.fft.fftn(self._grid[name])
        return FFT

    def _cal_SASA_FFT(self):
        print("Doing FFT for SASA")
        sasai_grid = self._grid["SASAi"]
        sasar_grid = self._grid["SASAr"]
        sasa_grid = np.add(sasar_grid, sasai_grid*1.j)
        FFT = np.fft.fftn(sasa_grid)
        return FFT

    def _write_to_nc(self, nc_handle, key, value):
        print("Writing %s into nc file"%key)
        # create dimensions
        for dim in value.shape:
            dim_name = "%d"%dim
            if dim_name not in nc_handle.dimensions.keys():
                nc_handle.createDimension(dim_name, dim)

        # create variable
        if value.dtype == int:
            store_format = "i8"
        elif value.dtype == float:
            store_format = "f8"
        else:
            raise RuntimeError("unsupported dtype %s"%value.dtype)
        dimensions = tuple(["%d"%dim for dim in value.shape])
        nc_handle.createVariable(key, store_format, dimensions)

        # save data
        nc_handle.variables[key][:] = value
        return None

    def _cal_grid_parameters_with_bsite(self, spacing, bsite_file, nc_handle):
        """
        :param spacing: float, unit in angstrom, the same in x, y, z directions
        :param bsite_file: str, the file name of "measured_binding_site.py" from AlGDock pipeline
        :param nc_handle: an instance of netCDF4.Dataset()
        :return: None
        """
        assert spacing > 0, "spacing must be positive"
        self._set_grid_key_value("origin", np.zeros([3], dtype=float))
        
        self._set_grid_key_value("d0", np.array([spacing, 0, 0], dtype=float))
        self._set_grid_key_value("d1", np.array([0, spacing, 0], dtype=float))
        self._set_grid_key_value("d2", np.array([0, 0, spacing], dtype=float))
        self._set_grid_key_value("spacing", np.array([spacing]*3, dtype=float))

        # function to easily grab a single float from a complex string
        def get_num(x):
            return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))

        # create a regular expression to parse the read lines
        parser = re.compile(r'\d+.\d+')

        for line in open(bsite_file, "r"):
            if line.startswith('com_min = '):
                com_min = [float(i) for i in parser.findall(line)]
            if line.startswith('com_max = '):
                com_max = [float(i) for i in parser.findall(line)]
            if line.startswith('site_R = '):
                site_R = [float(i) for i in parser.findall(line)][0]
            if line.startswith('half_edge_length = '):
                half_edge_length = [float(i) for i in parser.findall(line)][0]
        #half_edge_length = get_num(line)
        print("half_edge_length = ", half_edge_length)
        length = 2. * half_edge_length         # TODO: this is not good, half_edge_length is define in bsite_file
        count = np.ceil(length / spacing) + 1
        
        self._set_grid_key_value("counts", np.array([count]*3, dtype=int))

        for key in ["origin", "d0", "d1", "d2", "spacing", "counts"]:
            self._write_to_nc(nc_handle, key, self._grid[key])
        return None
    
    def _cal_grid_parameters_without_bsite(self, spacing, extra_buffer, nc_handle):
        """
        use this when making box encompassing the whole receptor
        spacing:    float, unit in angstrom, the same in x, y, z directions
        extra_buffer: float
        """
        assert spacing > 0 and extra_buffer > 0, "spacing and extra_buffer must be positive"
        self._set_grid_key_value("origin", np.zeros( [3], dtype=float))
        
        self._set_grid_key_value("d0", np.array([spacing, 0, 0], dtype=float))
        self._set_grid_key_value("d1", np.array([0, spacing, 0], dtype=float))
        self._set_grid_key_value("d2", np.array([0, 0, spacing], dtype=float))
        self._set_grid_key_value("spacing", np.array([spacing]*3, dtype=float))

        # TODO: Change LJ_radius to include scaling factors instead of shifting ligand position
        lj_radius = np.array(self._prmtop["LJ_SIGMA"]/2., dtype=float)
        dx = (self._crd[:,0] + lj_radius).max() - (self._crd[:,0] - lj_radius).min()
        dy = (self._crd[:,1] + lj_radius).max() - (self._crd[:,1] - lj_radius).min()
        dz = (self._crd[:,2] + lj_radius).max() - (self._crd[:,2] - lj_radius).min()

        print("Receptor enclosing box [%f, %f, %f]"%(dx, dy, dz))
        print("extra_buffer: %f"%extra_buffer)

        length = max([dx, dy, dz]) + 2.0*extra_buffer

        if np.ceil(length / spacing)%2 != 0:
            length = length + spacing
        count = np.ceil(length / spacing) + 1
        
        self._set_grid_key_value("counts", np.array([count]*3, dtype=int))
        print("counts ", self._grid["counts"])
        print("Total box size %f" %((count-1)*spacing))

        for key in ["origin", "d0", "d1", "d2", "spacing", "counts"]:
            self._write_to_nc(nc_handle, key, self._grid[key])
        return None
    
    def _move_receptor_to_grid_center(self):
        """
        use this when making box encompassing the whole receptor
        """
        spacing = self._grid["spacing"]        
        lower_receptor_corner = np.array([self._crd[:,i].min() for i in range(3)], dtype=float)
        upper_receptor_corner = np.array([self._crd[:,i].max() for i in range(3)], dtype=float)
        
        lower_receptor_corner_grid_aligned = lower_receptor_corner - (spacing + lower_receptor_corner % spacing)
        upper_receptor_corner_grid_aligned = upper_receptor_corner + (spacing - upper_receptor_corner % spacing)

        receptor_box_center_grid_aligned = (upper_receptor_corner_grid_aligned + lower_receptor_corner_grid_aligned) / 2.

        receptor_box_center = (upper_receptor_corner + lower_receptor_corner) / 2.
        
        total_grid_count = (self._uper_most_corner_crd+spacing)/spacing
        print(total_grid_count)             
        grid_center = (self._origin_crd + self._uper_most_corner_crd) / 2.       
        receptor_box_length = upper_receptor_corner - lower_receptor_corner
        receptor_box_length_grid_aligned = upper_receptor_corner_grid_aligned - lower_receptor_corner_grid_aligned

        #test redefs of variables
#        receptor_box_center = ([upper_receptor_corner_grid_aligned[0], 
#            upper_receptor_corner_grid_aligned[1]+0.5,
#            upper_receptor_corner_grid_aligned[2]+0.5] + lower_receptor_corner_grid_aligned) / 2.
        for index, coord in enumerate(upper_receptor_corner_grid_aligned):
            corner_to_corner_1D_distance = (coord - lower_receptor_corner_grid_aligned[index])/spacing[index]
            lower_corner_coord = lower_receptor_corner_grid_aligned[index]
            half_spacing = spacing[index]/2.
            print(corner_to_corner_1D_distance)            
            if corner_to_corner_1D_distance%2 == 0:
                shifted_upper_coord = coord + half_spacing
                shifted_lower_coord = lower_corner_coord - half_spacing
                upper_receptor_corner_grid_aligned[index] = shifted_upper_coord
                lower_receptor_corner_grid_aligned[index] = shifted_lower_coord

        receptor_box_center = (upper_receptor_corner_grid_aligned + lower_receptor_corner_grid_aligned) / 2.
        grid_snap = np.mod(receptor_box_center, spacing)
        if np.any(np.where(grid_snap != 0)):
            receptor_box_center = np.add(receptor_box_center, np.subtract(spacing, grid_snap))

        print('receptor_box_center', receptor_box_center)        
        displacement = grid_center - receptor_box_center
        
        print('lower_receptor_corner_grid_aligned: ', lower_receptor_corner_grid_aligned, 
            '\nupper_receptor_corner_grid_aligned: ', upper_receptor_corner_grid_aligned,
            '\nlower_receptor_corner: ', lower_receptor_corner, 
            '\nupper_receptor_corner: ', upper_receptor_corner,
            '\nreceptor_box_center: ', receptor_box_center,
            '\nreceptor_box_center_grid_aligned', receptor_box_center_grid_aligned,
            '\ngrid_center: ', grid_center,
            '\nreceptor_box_length: ', receptor_box_length,
            '\nreceptor_box_length_grid_aligned: ', receptor_box_length_grid_aligned,
            '\nspacing num', receptor_box_length_grid_aligned/spacing
            )
        print("Receptor is translated by ", displacement)
        self._displacement = displacement

        for atom_ind in range(len(self._crd)):
            self._crd[atom_ind] += displacement
        return None
    
    def _cal_grid_coordinates(self, nc_handle):
        """
        calculate grid coordinates (x,y,z) for each corner,
        save 'x', 'y', 'z' to self._grid
        """
        print("calculating grid coordinates")
        #
        x = np.zeros(self._grid["counts"][0], dtype=float)
        y = np.zeros(self._grid["counts"][1], dtype=float)
        z = np.zeros(self._grid["counts"][2], dtype=float)
        
        for i in range(self._grid["counts"][0]):
            x[i] = self._grid["origin"][0] + i*self._grid["d0"][0]

        for j in range(self._grid["counts"][1]):
            y[j] = self._grid["origin"][1] + j*self._grid["d1"][1]

        for k in range(self._grid["counts"][2]):
            z[k] = self._grid["origin"][2] + k*self._grid["d2"][2]

        self._set_grid_key_value("x", x)
        self._set_grid_key_value("y", y)
        self._set_grid_key_value("z", z)

        for key in ["x", "y", "z"]:
            self._write_to_nc(nc_handle, key, self._grid[key])
        return None

    def _get_charges(self, name):
        assert name in self._grid_func_names, "%s is not allowed"%name

        if name == "electrostatic":
            return 332.05221729 * np.array(self._prmtop["CHARGE_E_UNIT"], dtype=float)
        elif name == "LJa":
            return -2.0 * np.array(self._prmtop["A_LJ_CHARGE"], dtype=float)
        elif name == "LJr":
            return np.array(self._prmtop["R_LJ_CHARGE"], dtype=float)
        elif name == "SASAi":
            return np.array([0], dtype=float)
        elif name == "SASAr":
            return np.array([0], dtype=float)
        else:
            raise RuntimeError("%s is unknown"%name)

    def _cal_potential_grids(self, nc_handle, radii_type, exclude_H):
        """
        Divides each grid calculation into a separate process (electrostatic, LJr, LJa,
        SASAr, SASAi) and then divides the grid into slices along the x-axis determined by
        the "task divisor". Remainders are calculated in the last slice.  This adds
        multiprocessing functionality to the grid generation.
        """

        if radii_type == "LJ_SIGMA":
            sasa_radii = self._prmtop["LJ_SIGMA"]/2
        else:
            sasa_radii = self._prmtop["VDW_RADII"]

        atom_names = self._prmtop["PDB_TEMPLATE"]["ATOM_NAME"]
        atom_list = []
        for i in range(self._crd.shape[0]):
            if exclude_H:
                if atom_names[i][0] != 'H':
                    atom_list.append(i)
            else:
                atom_list.append(i)

        task_divisor = 6
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {}
            sasa_grid = np.empty((0,0,0))
            for name in self._grid_func_names:
                futures_array = []
                for i in range(task_divisor):
                    counts = np.copy(self._grid["counts"])
                    counts_x = counts[0] // task_divisor
                    if i == task_divisor-1:
                        counts_x += counts[0] % task_divisor
                    counts[0] = counts_x

                    grid_start_x = i * (self._grid["counts"][0] // task_divisor)
                    origin = np.copy(self._origin_crd)
                    origin[0] = grid_start_x * self._grid["spacing"][0]

                    if name != "SASAr":
                        dummy_grid = np.empty((1,1,1), dtype=np.float64)
                        if name[:4] != "SASA":
                            futures_array.append(executor.submit(
                                process_potential_grid_function,
                                name,
                                self._crd,
                                origin,
                                self._grid["spacing"],
                                counts,
                                self._get_charges(name),
                                sasa_radii,
                                atom_list,
                                self._molecule_sasa,
                                self._prmtop["PDB_TEMPLATE"]["RES_NAME"],
                                self._rec_core_scaling,
                                self._rec_surface_scaling,
                                self._rec_metal_scaling,
                                self._rho,
                                dummy_grid
                            ))
                        else:
                            futures_array.append(executor.submit(
                                process_potential_grid_function,
                                name,
                                self._crd,
                                origin,
                                self._grid["spacing"],
                                counts,
                                self._get_charges(name),
                                sasa_radii,
                                atom_list,
                                self._molecule_sasa,
                                self._prmtop["PDB_TEMPLATE"]["RES_NAME"],
                                self._rec_core_scaling,
                                self._rec_surface_scaling,
                                self._rec_metal_scaling,
                                self._rho,
                                dummy_grid
                            ))
                    else:
                        futures_array.append(executor.submit(
                            process_potential_grid_function,
                            name,
                            self._crd,
                            origin,
                            self._grid["spacing"],
                            counts,
                            self._get_charges(name),
                            sasa_radii,
                            atom_list,
                            self._molecule_sasa,
                            self._prmtop["PDB_TEMPLATE"]["RES_NAME"],
                            self._rec_core_scaling,
                            self._rec_surface_scaling,
                            self._rec_metal_scaling,
                            self._rho,
                            sasa_grid
                        ))
                futures[name] = futures_array
                if name == "SASAi":
                    sasa_array = []
                    for i in range(task_divisor):
                        partial_sasa_grid = futures[name][i].result()
                        sasa_array.append(partial_sasa_grid)
                    sasa_grid = np.concatenate(tuple(sasa_array))
            for name in futures:
                grid_array = []
                for i in range(task_divisor):
                    partial_grid = futures[name][i].result()
                    grid_array.append(partial_grid)
                grid = np.concatenate(tuple(grid_array), axis=0)
                if name == "SASAi":
                    sasa_grid = np.copy(grid)
                self._write_to_nc(nc_handle, name, grid)
                self._set_grid_key_value(name, grid)
                # self._set_grid_key_value(name, None)     # to save memory

        return None
    
    def _exact_values(self, coordinate):
        """
        coordinate: 3-array of float
        calculate the exact "potential" value at any coordinate
        """
        assert len(coordinate) == 3, "coordinate must have len 3"
        if not self._is_in_grid(coordinate):
            raise RuntimeError("atom is outside grid even after pbc translated")
        
        values = {}
        for name in self._grid_func_names:
            if name[:4] != "SASA":
                values[name] = 0.
        
        NATOM = self._prmtop["POINTERS"]["NATOM"]
        for atom_ind in range(NATOM):
            dif = coordinate - self._crd[atom_ind]
            R = np.sqrt((dif*dif).sum())
            lj_diameter = self._prmtop["LJ_SIGMA"][atom_ind]

            if R > lj_diameter:
                values["electrostatic"] +=  332.05221729 * self._prmtop["CHARGE_E_UNIT"][atom_ind] / R
                values["LJr"] +=  self._prmtop["R_LJ_CHARGE"][atom_ind] / R**12
                values["LJa"] += -2. * self._prmtop["A_LJ_CHARGE"][atom_ind] / R**6
        
        return values
    
    def _trilinear_interpolation( self, grid_name, coordinate ):
        """
        grid_name is a str one of "electrostatic", "LJr" and "LJa"
        coordinate is an array of three numbers
        trilinear interpolation
        https://en.wikipedia.org/wiki/Trilinear_interpolation
        """
        raise RuntimeError("Do not use, not tested yet")
        assert len(coordinate) == 3, "coordinate must have len 3"
        
        eight_corners, nearest_ind, furthest_ind = self._containing_cube( coordinate ) # throw exception if coordinate is outside
        lower_corner = eight_corners[0]
        
        (i0, j0, k0) = lower_corner
        (i1, j1, k1) = (i0 + 1, j0 + 1, k0 + 1)
        
        xd = (coordinate[0] - self._grid["x"][i0,j0,k0]) / (self._grid["x"][i1,j1,k1] - self._grid["x"][i0,j0,k0])
        yd = (coordinate[1] - self._grid["y"][i0,j0,k0]) / (self._grid["y"][i1,j1,k1] - self._grid["y"][i0,j0,k0])
        zd = (coordinate[2] - self._grid["z"][i0,j0,k0]) / (self._grid["z"][i1,j1,k1] - self._grid["z"][i0,j0,k0])
        
        c00 = self._grid[grid_name][i0,j0,k0]*(1. - xd) + self._grid[grid_name][i1,j0,k0]*xd
        c10 = self._grid[grid_name][i0,j1,k0]*(1. - xd) + self._grid[grid_name][i1,j1,k0]*xd
        c01 = self._grid[grid_name][i0,j0,k1]*(1. - xd) + self._grid[grid_name][i1,j0,k1]*xd
        c11 = self._grid[grid_name][i0,j1,k1]*(1. - xd) + self._grid[grid_name][i1,j1,k1]*xd
        
        c0 = c00*(1. - yd) + c10*yd
        c1 = c01*(1. - yd) + c11*yd
        
        c = c0*(1. - zd) + c1*zd
        return c
    
    def direct_energy(self, ligand_coordinate, ligand_charges):
        """
        :param ligand_coordinate: ndarray of shape (natoms, 3)
        :param ligand_charges: ndarray of shape (3,)
        :return: dic
        """
        assert len(ligand_coordinate) == len(ligand_charges["CHARGE_E_UNIT"]), "coord and charges must have the same len"
        energy = 0.
        for atom_ind in range(len(ligand_coordinate)):
            potentials = self._exact_values(ligand_coordinate[atom_ind])
            energy += potentials["electrostatic"]*ligand_charges["CHARGE_E_UNIT"][atom_ind]
            energy += potentials["LJr"]*ligand_charges["R_LJ_CHARGE"][atom_ind]
            energy += potentials["LJa"]*ligand_charges["A_LJ_CHARGE"][atom_ind]
        return energy
    
    def interpolated_energy(self, ligand_coordinate, ligand_charges):
        """
        ligand_coordinate:  array of shape (natoms, 3)
        ligand_charges: array of shape (3)
        assume that ligand_coordinate is inside grid
        """
        raise RuntimeError("Do not use, not tested yet")
        assert len(ligand_coordinate) == len(ligand_charges["CHARGE_E_UNIT"]), "coord and charges must have the same len"  
        grid_names = [name for name in self._grid_func_names if name[:4] != "SASA"]
        energy = 0.
        potentials = {}
        for atom_ind in range(len(ligand_coordinate)):
            for name in grid_names:
                potentials[name] = self._trilinear_interpolation(name, ligand_coordinate[atom_ind])
            
            energy += potentials["electrostatic"]*ligand_charges["CHARGE_E_UNIT"][atom_ind]
            energy += potentials["LJr"]*ligand_charges["R_LJ_CHARGE"][atom_ind]
            energy += potentials["LJa"]*ligand_charges["A_LJ_CHARGE"][atom_ind]
        
        return energy

    def get_FFTs(self):
        return self._FFTs

    def get_rho(self):
        return self._rho

    def write_box(self, file_name):
        IO.write_box(self, file_name)
        return None

    def write_pdb(self, file_name, mode):
        IO.write_pdb(self._prmtop, self._crd, file_name, mode)
        return None


if __name__ == "__main__":
    # do some test
    rec_prmtop_file = "../examples/amber/ubiquitin_ligase/receptor.prmtop"
    rec_inpcrd_file = "../examples/amber/ubiquitin_ligase/receptor.inpcrd"
    grid_nc_file = "../examples/grid/ubiquitin_ligase/grid.nc"
    lj_sigma_scaling_factor = 0.8
    # bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
    bsite_file = None
    spacing = 0.5

    rec_grid = RecGrid(rec_prmtop_file, lj_sigma_scaling_factor, rec_inpcrd_file, 
                        bsite_file,
                        grid_nc_file,
                        new_calculation=True,
                        spacing=spacing, radii_type="VDW_RADII")
    print("get_grid_func_names", rec_grid.get_grid_func_names())
    print("get_grids", rec_grid.get_grids())
    print("get_crd", rec_grid.get_crd())
    print("get_prmtop", rec_grid.get_prmtop())
    print("get_prmtop", rec_grid.get_charges())
    print("get_natoms", rec_grid.get_natoms())
    print("get_natoms", rec_grid.get_allowed_keys())

    rec_grid.write_box("../examples/grid/ubiquitin_ligase/box.pdb")
    rec_grid.write_pdb("../examples/grid/ubiquitin_ligase/test.pdb", "w")

    lig_prmtop_file = "../examples/amber/ubiquitin/ligand.prmtop"
    lig_inpcrd_file = "../examples/amber/ubiquitin/ligand.inpcrd"
    lig_grid = LigGrid(lig_prmtop_file, lj_sigma_scaling_factor, lig_inpcrd_file, rec_grid)
    lig_grid.cal_grids()
    print("get_bpmf", lig_grid.get_bpmf())
    print("get_number_translations", lig_grid.get_number_translations())
    print("get_box_volume", lig_grid.get_box_volume())
    print("get_meaningful_energies", lig_grid.get_meaningful_energies())
    print("get_meaningful_corners", lig_grid.get_meaningful_corners())
    print("set_meaningful_energies_to_none", lig_grid.set_meaningful_energies_to_none())
    print("get_initial_com", lig_grid.get_initial_com())
    print("Receptor SASA", rec_grid._get_molecule_sasa(0.14, 960))
    print("Ligand SASA", lig_grid._get_molecule_sasa(0.14, 960))



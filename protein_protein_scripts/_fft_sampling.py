"""
functions to run FFT sampling
"""
from __future__ import print_function

import sys
import os

import netCDF4
import numpy as np

sys.path.append("../bpmfwfft")
from bpmfwfft.fft_sampling import Sampling

BSITE_FILE = None


def sampling(rec_prmtop, lj_sigma_scal_fact,
                rec_inpcrd, grid_nc_file,
                lig_prmtop, lig_inpcrd,
                lig_coor_nc, nr_lig_conf,
                energy_sample_size_per_ligand,
                output_nc, output_dir,
                use_gpu_fft=False):
    lig_nc_handle = netCDF4.Dataset(lig_coor_nc, "r")
    if os.path.exists(output_nc):
        temp_nc = netCDF4.Dataset(output_nc, "r")
        start_index = temp_nc.variables["current_rotation_index"][:][0]
        temp_nc.close()
    else:
        start_index = 0

    print(f"Resuming job at rotation index {start_index}")
    print(f"debug info for start index: type:{type(start_index)}, value:{start_index}")
    if start_index < lig_nc_handle.variables["positions"][:].shape[0]:
        if start_index + nr_lig_conf < lig_nc_handle.variables["positions"][:].shape[0]:
            lig_coord_ensemble = lig_nc_handle.variables["positions"][:][start_index : start_index + nr_lig_conf]
        else:
            lig_coord_ensemble = lig_nc_handle.variables["positions"][:][start_index:]
        total_rotations = lig_nc_handle.variables["positions"].shape[0]
        lig_nc_handle.close()

        sampler = Sampling(rec_prmtop, lj_sigma_scal_fact,
                            rec_inpcrd,
                            BSITE_FILE, grid_nc_file, lig_prmtop, lig_inpcrd,
                            lig_coord_ensemble,
                            energy_sample_size_per_ligand,
                            output_nc,
                            start_index,
                            temperature=300.,
                            use_gpu_fft=use_gpu_fft)

        sampler.run_sampling()
        if start_index + nr_lig_conf >= total_rotations:
            print("Sampling Done")
    else:
        print("Sampling Done")
    return None


def is_sampling_nc_good(nc_file, rotation_nc_file):
    if not os.path.exists(nc_file):
        print(f"Error: {nc_file} doesn't exist")
        return False

    if not os.path.exists(rotation_nc_file):
        print(f"Error: {rotation_nc_file} doesn't exist")
        return False

    try:
        with netCDF4.Dataset(nc_file, "r") as nc_handle, \
                netCDF4.Dataset(rotation_nc_file, "r") as rotation_nc_handle:

            lig_positions = nc_handle.variables["lig_positions"][:]
            rotation_positions = rotation_nc_handle.variables["positions"][:]

            if lig_positions.shape[0] < rotation_positions.shape[0]:
                print(f"Current completed rotations: ({lig_positions.shape[0]}) "
                      f"is less than number of precalculated rotation positions: ({rotation_positions.shape[0]})")
                return False

            return True

    except Exception as e:
        print(f"Error opening or reading NC files: {str(e)}")
        return False

    except Exception as e:
        print(f"Error opening or reading {nc_file}: {str(e)}")
        return False


def parse_nr_ligand_confs(submit_file):
    if os.path.exists(submit_file):
        with open(submit_file, "r") as F:
            for line in F:
                if "--nr_lig_conf" in line:
                    nr_confs = line.split("--nr_lig_conf")[-1]
                    nr_confs = nr_confs.split()[0]
                    nr_confs = int(nr_confs)
                    return nr_confs
    return None



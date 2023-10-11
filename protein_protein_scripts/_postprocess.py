"""
define function to run postprocessing
"""
import sys
import numpy as np
import pickle

sys.path.append("/home/jim/src/p39/bpmfwfft/")
from bpmfwfft.postprocess import PostProcess


#SOLVENT_PHASES = ["OpenMM_GBn", "OpenMM_GBn2", "OpenMM_HCT", "OpenMM_OBC1", "OpenMM_OBC2"]
SOLVENT_PHASES = ["OpenMM_GBn", "OpenMM_GBn2", "OpenMM_OBC1", "OpenMM_OBC2"]
# SOLVENT_PHASES.extend(["sander_PBSA", "sander_OBC2"])
# SOLVENT_PHASES = ["OpenMM_OBC2"]

TEMPERATURE = 300.

def post_process(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file, 
                    nr_resampled_complexes, 
                    sander_tmp_dir, 
                    rec_pdb_out, lig_pdb_out, bpmf_pkl_out):
    post_pro = PostProcess(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                            SOLVENT_PHASES, nr_resampled_complexes, False, TEMPERATURE, sander_tmp_dir)
    post_pro.write_rececptor_pdb(rec_pdb_out)
    post_pro.write_resampled_ligand_pdb(lig_pdb_out)
    post_pro.pickle_bpmf(bpmf_pkl_out)
    return None

# def rotation_convergence(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
#                     nr_resampled_complexes,
#                     sander_tmp_dir,
#                     rec_pdb_out, lig_pdb_out, bpmf_pkl_out, n_rotations, M, N_step):
#     post_process_results = {}
#     for n in np.arange(1, n_rotations, step=N_step):
#         post_pro_list = []
#         for m in range(M):
#             rotation_indexes = np.random.choice(n_rotations, size=n, replace=True)
#             post_pro = PostProcess(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
#                                     SOLVENT_PHASES, nr_resampled_complexes, False, TEMPERATURE, sander_tmp_dir, rotation_indexes)
#             # post_pro.write_rececptor_pdb(rec_pdb_out)
#             # post_pro.write_resampled_ligand_pdb(lig_pdb_out)
#             post_pro.pickle_bpmf(bpmf_pkl_out[:-4] + f"_{n}_{m}" + bpmf_pkl_out[-4:])
#             post_pro_list.append(post_pro._bpmf)
#         post_process_results[n] = post_pro_list
#     pickle.dump(post_process_results, open(bpmf_pkl_out, "wb"))
#     return None

import multiprocessing as mp


def process_rotation(args):
    n, rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file, nr_resampled_complexes, sander_tmp_dir, M, bpmf_pkl_out, n_rotations = args
    post_pro_list = []
    for m in range(M):
        rotation_indexes = np.random.choice(n_rotations, size=n, replace=True)
        post_pro = PostProcess(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                               SOLVENT_PHASES, nr_resampled_complexes, False, TEMPERATURE, sander_tmp_dir,
                               rotation_indexes)
        # post_pro.write_rececptor_pdb(rec_pdb_out)
        # post_pro.write_resampled_ligand_pdb(lig_pdb_out)
        post_pro.pickle_bpmf(bpmf_pkl_out[:-4] + f"_{n}_{m}" + bpmf_pkl_out[-4:])
        post_pro_list.append(post_pro._bpmf)
    return n, post_pro_list


def rotation_convergence(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                         nr_resampled_complexes, sander_tmp_dir,
                         rec_pdb_out, lig_pdb_out, bpmf_pkl_out,
                         n_rotations, M, N_step):
    post_process_results = {}

    pool = mp.Pool()
    args_list = [(n, rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                  nr_resampled_complexes, sander_tmp_dir, M, bpmf_pkl_out, n_rotations) for n in
                 np.arange(1, n_rotations, step=N_step)]
    results = pool.map(process_rotation, args_list)
    pool.close()
    pool.join()

    for n, post_pro_list in results:
        post_process_results[n] = post_pro_list

    pickle.dump(post_process_results, open(bpmf_pkl_out, "wb"))
    return None



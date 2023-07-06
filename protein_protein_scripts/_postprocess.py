"""
define function to run postprocessing
"""
import sys

sys.path.append("/home/jim/src/p39/bpmfwfft/")
from bpmfwfft.postprocess import PostProcess


#SOLVENT_PHASES = ["OpenMM_GBn", "OpenMM_GBn2", "OpenMM_HCT", "OpenMM_OBC1", "OpenMM_OBC2"]
SOLVENT_PHASES = ["OpenMM_GBn", "OpenMM_GBn2", "OpenMM_OBC1", "OpenMM_OBC2"]
SOLVENT_PHASES.extend(["sander_PBSA", "sander_OBC2"])

TEMPERATURE = 300.

def post_process(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file, 
                    nr_resampled_complexes, 
                    sander_tmp_dir, 
                    rec_pdb_out, lig_pdb_out, bpmf_pkl_out, n_rotations):
    post_pro = PostProcess(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                            SOLVENT_PHASES, nr_resampled_complexes, False, TEMPERATURE, sander_tmp_dir, n_rotations)
    post_pro.write_rececptor_pdb(rec_pdb_out)
    post_pro.write_resampled_ligand_pdb(lig_pdb_out)
    post_pro.pickle_bpmf(bpmf_pkl_out)
    return None
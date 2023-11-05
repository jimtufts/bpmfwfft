"""
run post processing
"""

from __future__ import print_function

import sys
import os
import glob
import argparse
import netCDF4

from _fft_sampling import is_sampling_nc_good, parse_nr_ligand_confs 
from _postprocess import post_process, rotation_convergence


parser = argparse.ArgumentParser()
parser.add_argument("--max_jobs",  type=int, default=50)
parser.add_argument("--amber_dir", type=str, default="amber")
parser.add_argument("--sampling_dir", type=str, default="fft_sampling")
parser.add_argument("--sampling_nc", type=str, default="fft_sample.nc")
parser.add_argument("--sampling_out_name", type=str, default="ligand_resampled.pdb")
parser.add_argument("--pkl_name", type=str, default="bpmf.pkl")

parser.add_argument("--nr_resample", type=int, default=100)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)

parser.add_argument("--out_dir", type=str, default="out")
parser.add_argument("--check_convergence", action="store_true", default=False)
parser.add_argument("--pbs",   action="store_true", default=False)
parser.add_argument("--slurm",   action="store_true", default=False)
args = parser.parse_args()

RECEPTOR_PRMTOP = "receptor.prmtop"
LIGAND_PRMTOP = "ligand.prmtop"
COMPLEX_PRMTOP = "complex.prmtop"

FFT_SAMPLING_NC = args.sampling_nc

REC_PDB_OUT = "receptor_trans.pdb"
LIG_PDB_OUT = args.sampling_out_nameok
if args.check_convergence:
    BPMF_OUT = f"convergence_test.pkl"
else:
    BPMF_OUT = args.pkl_name


def is_sampling_good(sampling_dir):
    complex_name = sampling_dir.split("/")[-1]
    idx = complex_name[:4].lower()
    submit_file = idx + "_fft.job"
    nc_sampling_file = os.path.join(sampling_dir, FFT_SAMPLING_NC)
    print(nc_sampling_file)
    if os.path.exists(nc_sampling_file):
        temp_nc_handle = netCDF4.Dataset(nc_sampling_file, "r")
        nr_lig_confs = temp_nc_handle.variables["resampled_energies"][:].shape[0]
        temp_nc_handle.close()
        print(nr_lig_confs)
    else:
        nr_lig_confs = None
    if nr_lig_confs is None:
        return False

    return is_sampling_nc_good(nc_sampling_file, nr_lig_confs)

if args.pbs:
    this_script = os.path.abspath(sys.argv[0])
    amber_dir = os.path.abspath(args.amber_dir)
    sampling_dir = os.path.abspath(args.sampling_dir)

    complex_names = glob.glob(os.path.join(sampling_dir, "*"))
    complex_names = [os.path.basename(d) for d in complex_names if os.path.isdir(d)]
    complex_names = [c for c in complex_names if is_sampling_good(os.path.join(sampling_dir, c))]
    print(complex_names)

    for complex_name in complex_names:
        if not os.path.isdir(complex_name):
            os.makedirs(complex_name)

        idx = complex_name[:4].lower()
        amber_sub_dir = os.path.join(amber_dir, complex_name)
        sampling_sub_dir = os.path.join(sampling_dir, complex_name)
        out_dir = os.path.abspath(complex_name)

        qsub_file = os.path.join(out_dir, idx+"_post.job")
        log_file = os.path.join(out_dir, idx+"_post.log")
        qsub_script = f'''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s {log_file}
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=300:00:00
module load ambertools/14
source /home/jtufts/opt/module/anaconda.sh
date
python {this_script} \
        --amber_dir {amber_sub_dir} \
        --sampling_dir {sampling_sub_dir} \
        --out_dir {out_dir} \
        --nr_resample {args.nr_resample}
        --start {args.start}
        --end {args.end} \n'''

        bpmf_out = os.path.join(out_dir, BPMF_OUT)
        if not os.path.exists(bpmf_out):
            open(qsub_file, "w").write(qsub_script)
            print("Submiting " + qsub_file)
            os.system("qsub %s" % qsub_file)
elif args.slurm:
    this_script = os.path.abspath(sys.argv[0])
    amber_dir = os.path.abspath(args.amber_dir)
    sampling_dir = os.path.abspath(args.sampling_dir)
    out_dir = os.path.abspath(args.out_dir)

    complex_names = glob.glob(os.path.join(sampling_dir, "*"))
    complex_names = [os.path.basename(d) for d in complex_names if os.path.isdir(d)]
    complex_names = [c for c in complex_names if is_sampling_good(os.path.join(sampling_dir, c))]
    print(complex_names)

    if args.max_jobs > 0:
        max_jobs = args.max_jobs
    else:
        max_jobs = len(complex_names)
    print("max_jobs = %d" % max_jobs)

    job_count = 0

    for complex_name in complex_names:
        com_dir = os.path.join(out_dir, complex_name)
        if not os.path.isdir(com_dir):
            os.makedirs(com_dir)

        idx = complex_name[:4].lower()
        amber_sub_dir = os.path.join(amber_dir, complex_name)
        sampling_sub_dir = os.path.join(sampling_dir, complex_name)
        sbatch_file = os.path.join(com_dir, idx + "_post_slurm.job")
        log_file = os.path.join(com_dir, idx + "_post.log")
        sbatch_script = f'''#!/bin/bash
#SBATCH --job-name={idx}
#SBATCH --output={log_file}
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=iit103
#SBATCH --export=ALL
#SBATCH -t 2:00:00
#SBATCH --constraint="lustre"

module purge
module load gpu
module load slurm
module load openmpi			
module load amber
source /cm/shared/apps/spack/cpu/opt/spack/linux-centos8-zen/gcc-8.3.1/anaconda3-2020.11-da3i7hmt6bdqbmuzq6pyt7kbm47wyrjp/etc/profile.d/conda.sh
conda activate fft
date
#SET the number of openmp threads

#Run the job
python {this_script} \
--amber_dir {amber_sub_dir} \
--sampling_dir {sampling_sub_dir} \
--out_dir {com_dir} \
--nr_resample {args.nr_resample} 
--start {args.start}
--end {args.end} \n'''

        bpmf_out = os.path.join(com_dir, BPMF_OUT)
        if not os.path.exists(bpmf_out):
            open(sbatch_file, "w").write(sbatch_script)
            print("Submiting " + sbatch_file)
            os.system("sbatch %s" % sbatch_file)
            job_count += 1
            if job_count == max_jobs:
                print("Max number of jobs %d reached." % job_count)
                break

else:
    rec_prmtop = os.path.join(args.amber_dir, RECEPTOR_PRMTOP)
    lig_prmtop = os.path.join(args.amber_dir, LIGAND_PRMTOP)
    complex_prmtop = os.path.join(args.amber_dir, COMPLEX_PRMTOP)

    sampling_nc_file = os.path.join(args.sampling_dir, FFT_SAMPLING_NC)
    nr_resampled_complexes = args.nr_resample

    sander_tmp_dir = args.out_dir

    rec_pdb_out = os.path.join(args.out_dir, REC_PDB_OUT)
    lig_pdb_out = os.path.join(args.out_dir, LIG_PDB_OUT)
    bpmf_pkl_out = os.path.join(args.out_dir, BPMF_OUT )
    start = args.start
    end = args.end

    if args.check_convergence:
        n_rotations = netCDF4.Dataset(sampling_nc_file).variables["resampled_energies"].shape[0]
        M = 500 # bootstrap iterations
        N_step = 100 # resolution of convergence test
        rotation_convergence(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                     nr_resampled_complexes,
                     sander_tmp_dir,
                     rec_pdb_out, lig_pdb_out, bpmf_pkl_out, n_rotations, M, N_step)
    else:
        post_process(rec_prmtop, lig_prmtop, complex_prmtop, sampling_nc_file,
                nr_resampled_complexes, start, end,
                sander_tmp_dir,
                rec_pdb_out, lig_pdb_out, bpmf_pkl_out)


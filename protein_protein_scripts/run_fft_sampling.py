"""
run fft sampling
"""
from __future__ import print_function

import sys
import os
import glob
import argparse
import datetime

import numpy as np

from _receptor_grid_cal import is_nc_grid_good
from _fft_sampling import sampling, is_sampling_nc_good
from _receptor_grid_cal import get_grid_size_from_nc

parser = argparse.ArgumentParser()
parser.add_argument("--max_jobs",                      type=int, default=50)
parser.add_argument("--amber_dir",                     type=str, default="amber")
parser.add_argument("--coord_dir",                     type=str, default="min")
parser.add_argument("--grid_dir",                      type=str, default="grid")
parser.add_argument("--grid_name",                      type=str, default="grid.nc")
parser.add_argument("--result_name",                      type=str, default="fft_sample.nc")
parser.add_argument("--lig_ensemble_dir",              type=str, default="rotation")

parser.add_argument("--energy_sample_size_per_ligand", type=int, default=1000)
parser.add_argument("--nr_lig_conf",                   type=int, default=100)

parser.add_argument("--out_dir",                       type=str, default="out")

parser.add_argument("--lj_scale",                      type=float, default=1.0)
parser.add_argument("--rc_scale",                      type=float, default=0.76)
parser.add_argument("--rs_scale",                      type=float, default=0.53)
parser.add_argument("--rm_scale",                      type=float, default=0.55)
parser.add_argument("--lc_scale",                      type=float, default=0.81)
parser.add_argument("--ls_scale",                      type=float, default=0.50)
parser.add_argument("--lm_scale",                      type=float, default=0.54)
parser.add_argument("--rho",                           type=float, default=9.0)
parser.add_argument("--pbs",   action="store_true", default=False)
parser.add_argument("--slurm",   action="store_true", default=False)
args = parser.parse_args()

RECEPTOR_INPCRD = "receptor.inpcrd"
RECEPTOR_PRMTOP = "receptor.prmtop"

LIGAND_INPCRD = "ligand.inpcrd"
LIGAND_PRMTOP = "ligand.prmtop"

LIG_COOR_NC = "rotation.nc"

# GRID_NC = "all_grid_noH.nc"
GRID_NC = args.grid_name
FFT_SAMPLING_NC = args.result_name


def is_running_slurm(idx, out_dir):
    # if os.path.exists(qsub_file) and os.path.exists(nc_file) and (not os.path.exists(log_file)):
    #     return True
    # if os.path.exists(qsub_file) and (not os.path.exists(nc_file)) and (os.path.exists(log_file)):
    #     return True
    import subprocess
    command = f'squeue -u jtufts'
    output = subprocess.check_output(command, shell=True, text=True)
    if idx in output or os.path.exists(os.path.join(out_dir, "DONE")):
        return True
    return False

def is_running(qsub_file, log_file, nc_file):
    if os.path.exists(qsub_file) and os.path.exists(nc_file) and (not os.path.exists(log_file)):
        return True
    if os.path.exists(qsub_file) and (not os.path.exists(nc_file)) and (os.path.exists(log_file)):
        return True
    return False

if args.pbs:
    this_script = os.path.abspath(sys.argv[0])
    amber_dir = os.path.abspath(args.amber_dir)
    coord_dir = os.path.abspath(args.coord_dir)
    grid_dir = os.path.abspath(args.grid_dir)
    lig_ensemble_dir = os.path.abspath(args.lig_ensemble_dir)

    complex_names = glob.glob(os.path.join(grid_dir, "*"))
    complex_names = [os.path.basename(d) for d in complex_names if os.path.isdir(d)]

    complex_names = [c for c in complex_names if is_nc_grid_good(os.path.join(grid_dir, c, GRID_NC))]

    grid_sizes = {}

    for complex_name in complex_names:
        grid_sizes[complex_name] = get_grid_size_from_nc(os.path.join(grid_dir, complex_name, GRID_NC))
    complex_names.sort(key=lambda name: grid_sizes[name])
    print("Complex   grid size")

    for complex_name in complex_names:
        print(complex_name, grid_sizes[complex_name])

    pwd = os.getcwd()
    complex_names = [c for c in complex_names if not is_sampling_nc_good(
        os.path.join(pwd, c, FFT_SAMPLING_NC), args.nr_lig_conf)]

    if args.max_jobs > 0:
        max_jobs = args.max_jobs
    else:
        max_jobs = len(complex_names)
    print("max_jobs = %d" % max_jobs)

    job_count = 0
    for complex_name in complex_names:

        com_dir = os.path.join(out_dir, complex_name)
        if not os.path.isdir(complex_name):
            os.makedirs(complex_name)

        idx = complex_name[:4].lower()
        amber_sub_dir = os.path.join(amber_dir, complex_name)
        coor_sub_dir = os.path.join(coord_dir, complex_name)
        grid_sub_dir = os.path.join(grid_dir, complex_name)
        lig_ensemble_sub_dir = os.path.join(lig_ensemble_dir, complex_name)

        out_dir = os.path.abspath(complex_name)
        qsub_file = os.path.join(out_dir, idx + "_fft.job")
        log_file = os.path.join(out_dir, idx + "_fft.log")
        qsub_script = f'''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s {log_file}
#PBS -j oe
#PBS -l nodes=1:ppn=4,walltime=300:00:00

source /home/jtufts/opt/module/anaconda.sh
date
python {this_script}  \
        --amber_dir {amber_sub_dir} \
        --coord_dir {coor_sub_dir} \
        --grid_dir {grid_sub_dir} \
        --grid_name {args.grid_name} \
        --result_name {lig_ensemble_sub_dir} \
        --out_dir {out_dir} \
        --lj_scale {args.lj_scale:.6f} \
        --rc_scale {args.rc_scale:.6f} \
        --rs_scale {args.rs_scale:.6f} \
        --rm_scale {args.rm_scale:.6f} \
        --lc_scale {args.lc_scale:.6f} \
        --ls_scale {args.ls_scale:.6f} \
        --lm_scale {args.lm_scale:.6f} \
        --nr_lig_conf {args.nr_lig_conf} \
        --energy_sample_size_per_ligand {args.energy_sample_size_per_ligand} \n'''

        fft_sampling_nc_file = os.path.join(out_dir, FFT_SAMPLING_NC)
        if not is_running(qsub_file, log_file, fft_sampling_nc_file):

            if os.path.exists(log_file):
                print("remove file " + log_file)
                os.system("rm " + log_file)

            print("Submitting %s" % complex_name)
            open(qsub_file, "w").write(qsub_script)
            os.system("qsub %s" % qsub_file)
            job_count += 1
            if job_count == max_jobs:
                print("Max number of jobs %d reached." % job_count)
                break
elif args.slurm:
    this_script = os.path.abspath(sys.argv[0])
    amber_dir = os.path.abspath(args.amber_dir)
    coord_dir = os.path.abspath(args.coord_dir)
    grid_dir = os.path.abspath(args.grid_dir)
    lig_ensemble_dir = os.path.abspath(args.lig_ensemble_dir)
    out_dir = os.path.abspath(args.out_dir)

    complex_names = glob.glob(os.path.join(grid_dir, "*"))
    complex_names = [os.path.basename(d) for d in complex_names if os.path.isdir(d)]

    complex_names = [c for c in complex_names if is_nc_grid_good(os.path.join(grid_dir, c, GRID_NC))]

    grid_sizes = {}

    for complex_name in complex_names:
        grid_sizes[complex_name] = get_grid_size_from_nc(os.path.join(grid_dir, complex_name, GRID_NC))
    complex_names.sort(key=lambda name: grid_sizes[name])
    print("Complex   grid size   n_cpu   memory")

    for complex_name in complex_names:
        # cpu_count = np.ceil(((0.00045279032 * grid_sizes[complex_name] ** 3) / 128000) * 128)
        cpu_count = 16
        memory_amt = np.ceil((0.00045279032 * grid_sizes[complex_name] ** 3)+4000)
        if memory_amt < 32000.:
            memory_amt = 32000.
        print(complex_name, grid_sizes[complex_name], cpu_count, memory_amt)

    pwd = os.getcwd()
    complex_names = [c for c in complex_names if not is_sampling_nc_good(
        os.path.join(pwd, c, FFT_SAMPLING_NC), args.nr_lig_conf)]

    if args.max_jobs > 0:
        max_jobs = args.max_jobs
    else:
        max_jobs = len(complex_names)
    print("max_jobs = %d" % max_jobs)

    job_count = 0
    for complex_name in complex_names:

        cpu_count = 16
        memory_amt = np.ceil((0.00045279032 * grid_sizes[complex_name] ** 3) + 4000)
        if memory_amt < 32000.:
            memory_amt = 32000.

        com_dir = os.path.join(out_dir, complex_name)
        if not os.path.isdir(com_dir):
            os.makedirs(com_dir)

        idx = complex_name[:4].lower()
        amber_sub_dir = os.path.join(amber_dir, complex_name)
        coor_sub_dir = os.path.join(coord_dir, complex_name)
        grid_sub_dir = os.path.join(grid_dir, complex_name)
        lig_ensemble_sub_dir = os.path.join(lig_ensemble_dir, complex_name)

        # out_dir = os.path.abspath(complex_name)
        qsub_file = os.path.join(com_dir, idx+"_fft_slurm.job")
        current_datetime = datetime.datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(com_dir, idx+f"_fft_{date_time_string}.log")
        qsub_script = f'''#!/bin/bash
#SBATCH --job-name={idx}
#SBATCH --output={log_file}
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={int(cpu_count)}
#SBATCH --mem={int(memory_amt)}M
#SBATCH --account=iit103
#SBATCH --export=ALL
#SBATCH -t 48:00:00
#SBATCH --constraint="lustre"
module purge 
module load cpu
module load slurm
module load gcc
module load openmpi
source /cm/shared/apps/spack/cpu/opt/spack/linux-centos8-zen/gcc-8.3.1/anaconda3-2020.11-da3i7hmt6bdqbmuzq6pyt7kbm47wyrjp/etc/profile.d/conda.sh
conda activate fft
date

#SET the number of openmp threads
export OMP_NUM_THREADS={int(cpu_count)}

#Run the job
python {this_script}  \
        --amber_dir {amber_sub_dir} \
        --coord_dir {coor_sub_dir} \
        --grid_dir {grid_sub_dir} \
        --grid_name {args.grid_name} \
        --result_name {args.result_name} \
        --lig_ensemble_dir {lig_ensemble_sub_dir} \
        --out_dir {com_dir} \
        --lj_scale {args.lj_scale:.6f} \
        --rc_scale {args.rc_scale:.6f} \
        --rs_scale {args.rs_scale:.6f} \
        --rm_scale {args.rm_scale:.6f} \
        --lc_scale {args.lc_scale:.6f} \
        --ls_scale {args.ls_scale:.6f} \
        --lm_scale {args.lm_scale:.6f} \
        --nr_lig_conf {args.nr_lig_conf} \
        --energy_sample_size_per_ligand {args.energy_sample_size_per_ligand} \n'''

        fft_sampling_nc_file = os.path.join(com_dir, FFT_SAMPLING_NC)
        if not is_running_slurm(idx, out_dir):

            if os.path.exists(log_file):
                print("remove file " + log_file)
                os.system("rm "+log_file)

            print(f"Submitting {complex_name} log: {log_file}")
            open(qsub_file, "w").write(qsub_script)
            os.system("sbatch %s" % qsub_file)
            job_count += 1
            if job_count == max_jobs:
                print("Max number of jobs %d reached." % job_count)
                break

else:
    rec_prmtop = os.path.join(args.amber_dir, RECEPTOR_PRMTOP)
    lj_sigma_scal_fact = args.lj_scale
    rc_scale = args.rc_scale
    rs_scale = args.rs_scale
    rm_scale = args.rm_scale
    lc_scale = args.lc_scale
    ls_scale = args.ls_scale
    lm_scale = args.lm_scale
    rho = args.rho
    rec_inpcrd = os.path.join(args.amber_dir, RECEPTOR_INPCRD)

    grid_nc_file = os.path.join(args.grid_dir, GRID_NC)

    lig_prmtop = os.path.join(args.amber_dir, LIGAND_PRMTOP)
    lig_inpcrd = os.path.join(args.amber_dir, LIGAND_INPCRD)

    lig_coor_nc = os.path.join(args.lig_ensemble_dir, LIG_COOR_NC)
    nr_lig_conf = args.nr_lig_conf

    energy_sample_size_per_ligand = args.energy_sample_size_per_ligand
    output_nc = os.path.join(args.out_dir, FFT_SAMPLING_NC)
    output_dir = args.out_dir

    sampling(rec_prmtop, lj_sigma_scal_fact,
             rc_scale, rs_scale, rm_scale,
             lc_scale, ls_scale, lm_scale,
             rho,
             rec_inpcrd, grid_nc_file,
             lig_prmtop, lig_inpcrd,
             lig_coor_nc, nr_lig_conf,
             energy_sample_size_per_ligand,
             output_nc, output_dir)

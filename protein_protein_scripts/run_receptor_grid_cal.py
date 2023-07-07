"""
run receptor grid calculation
"""

from __future__ import print_function

import os
import sys
import glob
import argparse

from math import sqrt

from _receptor_grid_cal import rec_grid_cal, is_nc_grid_good, get_grid_size_from_lig_rec_crd

parser = argparse.ArgumentParser()
parser.add_argument("--max_jobs",           type=int, default=100)
parser.add_argument("--amber_dir",          type=str, default="amber")
parser.add_argument("--coord_dir",          type=str, default="min")
parser.add_argument("--out_dir",            type=str, default="out")
parser.add_argument("--grid_file_name",     type=str, default="grid.nc")
parser.add_argument("--radii_type",         type=str, default="VDW_RADII")

parser.add_argument("--lj_scale",    type=float, default=1.0)
parser.add_argument("--rc_scale",    type=float, default=0.76)
parser.add_argument("--rs_scale",    type=float, default=0.53)
parser.add_argument("--rm_scale",    type=float, default=0.55)
parser.add_argument("--rho",         type=float, default=9.0)
parser.add_argument("--spacing",     type=float, default=0.5)
parser.add_argument("--buffer",      type=float, default=1.0)

parser.add_argument("--exclude_H",    type=bool, default=True)

parser.add_argument("--pbs",   action="store_true", default=False)
parser.add_argument("--slurm",   action="store_true", default=False)

args = parser.parse_args()

LIGAND_INPCRD = "ligand.inpcrd"
RECEPTOR_INPCRD = "receptor.inpcrd"
RECEPTOR_PRMTOP = "receptor.prmtop"

GRID_NC = args.grid_file_name
PDB_OUT = "receptor_trans.pdb"
BOX_OUT = "box.pdb"


def is_running(qsub_file, log_file, nc_file):
    if os.path.exists(qsub_file) and os.path.exists(nc_file) and (not os.path.exists(log_file)):
        return True

    if os.path.exists(qsub_file) and (not os.path.exists(nc_file)) and (not os.path.exists(log_file)):
        return True
    return False

if args.pbs:
    this_script = os.path.abspath(sys.argv[0])
    amber_dir = os.path.abspath(args.amber_dir)
    coord_dir = os.path.abspath(args.coord_dir)
    out_dir = os.path.abspath(args.out_dir)

    amber_sub_dirs = glob.glob(os.path.join(amber_dir, "*"))
    amber_sub_dirs = [dir for dir in amber_sub_dirs if os.path.isdir(dir)]
    complex_names = [os.path.basename(dir) for dir in amber_sub_dirs]

    box_sizes = {}
    for complex in complex_names:
        rec_inpcrd = os.path.join(coord_dir, complex, RECEPTOR_INPCRD)
        lig_inpcrd = os.path.join(coord_dir, complex, LIGAND_INPCRD)
        box_sizes[complex] = get_grid_size_from_lig_rec_crd(rec_inpcrd, lig_inpcrd, args.buffer)
    complex_names.sort(key=lambda name: box_sizes[name])
    print("Complex    box size")
    for c in complex_names:
        print(c, box_sizes[c])

    if args.max_jobs > 0:
        max_jobs = args.max_jobs
    else:
        max_jobs = len(complex_names)
    print("max_jobs = %d"%max_jobs)

    job_count = 0
    for complex in complex_names:

        com_dir = os.path.join(out_dir, complex)
        if not os.path.isdir(com_dir):
            os.makedirs(com_dir)

        id = complex[:4].lower()
        amber_sub_dir = os.path.join(amber_dir, complex)
        coor_sub_dir = os.path.join(coord_dir, complex)

        qsub_file = os.path.join(com_dir, id + "_grid_pbs.job")
        log_file = os.path.join(com_dir, id+"_grid.log")
        qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s '''%log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=2,walltime=300:00:00

source /home/jtufts/opt/module/anaconda.sh
date
python ''' + this_script + \
        ''' --amber_dir ''' + amber_sub_dir + \
        ''' --coord_dir ''' + coor_sub_dir + \
        ''' --out_dir '''   + com_dir + \
        ''' --grid_file_name ''' + args.grid_file_name + \
        ''' --lj_scale %f'''%args.lj_scale + \
        ''' --rc_scale %f''' % args.rc_scale + \
        ''' --rs_scale %f''' % args.rs_scale + \
        ''' --rm_scale %f''' % args.rm_scale + \
        ''' --spacing %f'''%args.spacing + \
        ''' --buffer %f'''%args.buffer + \
        ''' --exclude_H %f''' % args.buffer + '''\n'''

        if not is_nc_grid_good(os.path.join(com_dir, GRID_NC)) and not is_running(qsub_file, log_file,
                                                                os.path.join(com_dir, GRID_NC)):
            print("Submitting %s"%complex)
            open(qsub_file, "w").write(qsub_script)
            os.system("qsub %s" %qsub_file)
            job_count += 1
            if job_count == max_jobs:
                print("Max number of jobs %d reached."%job_count)
                break
        else:
            print("Calculation for %s is done"%complex)
elif args.slurm:
    this_script = os.path.abspath(sys.argv[0])
    amber_dir = os.path.abspath(args.amber_dir)
    coord_dir = os.path.abspath(args.coord_dir)
    out_dir = os.path.abspath(args.out_dir)

    amber_sub_dirs = glob.glob(os.path.join(amber_dir, "*"))
    amber_sub_dirs = [dir for dir in amber_sub_dirs if os.path.isdir(dir)]
    complex_names = [os.path.basename(dir) for dir in amber_sub_dirs]

    box_sizes = {}
    for complex in complex_names:
        rec_inpcrd = os.path.join(coord_dir, complex, RECEPTOR_INPCRD)
        lig_inpcrd = os.path.join(coord_dir, complex, LIGAND_INPCRD)
        box_sizes[complex] = get_grid_size_from_lig_rec_crd(rec_inpcrd, lig_inpcrd, args.buffer)
    complex_names.sort(key=lambda name: box_sizes[name])
    print("Complex    box size")
    for c in complex_names:
        print(c, box_sizes[c])

    if args.max_jobs > 0:
        max_jobs = args.max_jobs
    else:
        max_jobs = len(complex_names)
    print("max_jobs = %d"%max_jobs)

    job_count = 0
    for complex in complex_names:

        com_dir = os.path.join(out_dir, complex)
        if not os.path.isdir(com_dir):
            os.makedirs(com_dir)

        id = complex[:4].lower()
        amber_sub_dir = os.path.join(amber_dir, complex)
        coor_sub_dir = os.path.join(coord_dir, complex)

        # out_dir = os.path.abspath(complex)
        sbatch_file = os.path.join(com_dir, id+"_grid_slurm.job")
        log_file = os.path.join(com_dir, id+"_grid.log")
        sbatch_script = f'''#!/bin/bash
#SBATCH --job-name={id}
#SBATCH --output={log_file}
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=31125M
#SBATCH --account=iit103
#SBATCH --export=ALL
#SBATCH -t 10:00:00
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
export OMP_NUM_THREADS=16

#Run the job
python {this_script} \
        --amber_dir {amber_sub_dir} \
        --coord_dir {coor_sub_dir} \
        --out_dir {com_dir} \
        --grid_file_name {args.grid_file_name} \
        --lj_scale {args.lj_scale:.6f} \
        --rc_scale {args.rc_scale:.6f} \
        --rs_scale {args.rs_scale:.6f} \
        --rm_scale {args.rm_scale:.6f} \
        --spacing {args.spacing:.6f} \
        --buffer {args.buffer:.6f} \
        --radii_type {args.radii_type} \
        --exclude_H \
        \n'''
        if not is_nc_grid_good(os.path.join(com_dir, GRID_NC)) and not is_running(sbatch_file, log_file,
                                                                os.path.join(com_dir, GRID_NC)):
            print("Submitting %s"%complex)
            open(sbatch_file, "w").write(sbatch_script)
            os.system("sbatch %s" %sbatch_file)
            job_count += 1
            if job_count == max_jobs:
                print("Max number of jobs %d reached."%job_count)
                break
        else:
            print("Calculation for %s is done"%complex)
else:
    prmtop = os.path.join(args.amber_dir, RECEPTOR_PRMTOP)
    lj_scale = args.lj_scale
    rc_scale = args.rc_scale
    rs_scale = args.rs_scale
    rm_scale = args.rm_scale
    rec_inpcrd = os.path.join(args.coord_dir, RECEPTOR_INPCRD)
    lig_inpcrd = os.path.join(args.coord_dir, LIGAND_INPCRD)
    rho = args.rho
    exclude_H = args.exclude_H

    spacing = args.spacing
    buffer = args.buffer
    radii_type = args.radii_type

    grid_out = os.path.join(args.out_dir, GRID_NC)
    pdb_out = os.path.join(args.out_dir, PDB_OUT)
    box_out = os.path.join(args.out_dir, BOX_OUT)
    print()
    rec_grid_cal(prmtop, lj_scale, rc_scale, rs_scale, rm_scale, rho,
                 rec_inpcrd, lig_inpcrd, spacing, buffer, grid_out, pdb_out, box_out, radii_type, exclude_H)


"""
run modeller to add missing residues
"""
import os
import sys
import shutil
import glob
import argparse
import subprocess

from _fix_pdb import AddMissing

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_dir", type=str, default="pdbs")
parser.add_argument("--pdb", type=str, default="xxx.pdb")
parser.add_argument("--submit", action="store_true", default=False)
parser.add_argument("--local", action="store_true", default=False)

args = parser.parse_args()

if args.submit:
    
    this_script = os.path.abspath(sys.argv[0])
    pwd = os.getcwd()

    working_dirs = {}
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))
    for pdb in pdb_files:
        id = os.path.basename(pdb)[:-4]

        if not os.path.isdir(id):
            os.makedirs(id)
        working_dirs[id] = os.path.join(pwd, id)
        shutil.copy(pdb, working_dirs[id])

    for id, dir in working_dirs.items():

        pdb_file_name = id+".pdb"

        qsub_file = os.path.join(dir, id+"_modeller.job")
        log_file  = os.path.join(dir,id+"_modeller.log" )
        qsub_script = '''
#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s '''%log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=300:00:00

source /home/jtufts/opt/module/anaconda.sh
cd ''' + dir + '''
date
python ''' + this_script + \
        ''' --pdb ''' + pdb_file_name
        open( qsub_file, "w" ).write( qsub_script )
        os.system( "qsub %s" %qsub_file )

if args.local:

    this_script = os.path.abspath(sys.argv[0])
    pwd = os.getcwd()

    working_dirs = {}
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))

    for pdb in pdb_files:
        basename = os.path.basename(pdb)

        if len(basename) == 8 and basename.endswith('.pdb'):
            id = basename[:4]
            dirname = f"{id}"

        elif len(basename) == 12 and basename.endswith('_u.pdb'):
            id = basename[:4]
            dirname = f"{id}_u"

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if dirname not in working_dirs:
            working_dirs[dirname] = {'dir': os.path.join(pwd, dirname), 'files': []}

        working_dirs[dirname]['files'].append(basename)
        shutil.copy(pdb, working_dirs[dirname]['dir'])

    for id, values in working_dirs.items():
        dir = values['dir']
        for pdb_file_name in values['files']:
            script_file = os.path.join(dir, pdb_file_name[:-4]  + "_modeller.sh")
            log_file = os.path.join(dir, pdb_file_name[:-4] + "_modeller.log")
            shell_script = f'''#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fft
cd {dir}
date
python {this_script} --pdb {pdb_file_name}'''
            with open(script_file, "w") as script:
                script.write(shell_script)
            os.chmod(script_file, 0o755)
            with open(log_file, "a") as log:
                result = subprocess.run(f"bash {script_file}", shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                log.write(result.stdout)
                log.write(result.stderr)


else:
    print(args.pdb)
    fix = AddMissing(args.pdb)
    fix.write_dummy_alignments()
    fix.do_auto_align()
    fix.do_auto_modeller()
    fix.write_vmd_script()

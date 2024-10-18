"""
to generate AMBER topology and coordinate files for bound and unbound complexes and their binding partners
"""
from __future__ import print_function

import os
import argparse
import logging
import re
import sys
import traceback

from _affinity_data import AffinityData
from _chains_combine import write_b_receptor_ligand_pdbs, write_u_receptor_ligand_pdbs
from _amber_tleap import generate_prmtop

parser = argparse.ArgumentParser()
parser.add_argument("--affinity_dir", type=str, default="affinity")
parser.add_argument("--ions_cofactors_dir", type=str, default="ions_cofactors")
parser.add_argument("--cofactors_frcmod_dir", type=str, default="cofactors_frcmod")
parser.add_argument("--modeller_dir", type=str, default="modeller")
parser.add_argument("--benchmark", type=str, choices=["affinity", "docking"], default="affinity")

args = parser.parse_args()

if args.benchmark == "affinity":
    AFFINITY_DATA_FILES = ["affinity_v1.tsv", "affinity_v2.tsv"]
    AFFINITY_DATA_FILES = [os.path.join(args.affinity_dir, file) for file in AFFINITY_DATA_FILES]
else:
    AFFINITY_DATA_FILES = ["docking_v5.tsv"]
    AFFINITY_DATA_FILES = [os.path.join(args.affinity_dir, file) for file in AFFINITY_DATA_FILES]

TER_CUTOFF = 5
LOOP_CUTOFF = 15

if not os.path.isdir(args.modeller_dir):
    raise RuntimeError("%s does not exist" % args.modeller_dir)

if not os.path.isdir(args.ions_cofactors_dir):
    raise RuntimeError("%s does not exist" % args.ions_cofactors_dir)

aff = AffinityData(AFFINITY_DATA_FILES)
b_complexes = aff.get_bound_complexes()
u_complexes = aff.get_unbound_complexes()

def clean_name(name):
    """Remove trailing spaces and special characters from a name."""
    return re.sub(r'[^a-zA-Z0-9_:]', '', name.strip())

def create_directory_name(name, is_bound, complex_id):
    """Create a consistent directory name for both bound and unbound complexes."""
    clean = clean_name(name)
    parts = clean.split('_')

    if len(parts) != 2:
        # If the name doesn't follow the expected format, use it as is
        return f"{clean}_U" if not is_bound else clean

    pdb_id, chains = parts
    chain_parts = chains.split(':')

    if is_bound:
        # For bound complexes, use the original format
        return clean
    else:
        # For unbound complexes, use a consistent format
        return f"{pdb_id}_{chain_parts[0]}:{chain_parts[1]}_U"

def process_complexes(complexes, is_bound):
    for name, complex_id in complexes.items():
        dir_name = create_directory_name(name, is_bound, complex_id)
        print(f"Processing {'bound' if is_bound else 'unbound'} complex: {dir_name}")
        
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        
        try:
            if is_bound:
                write_b_receptor_ligand_pdbs({name: complex_id}, args.modeller_dir, args.ions_cofactors_dir,
                                             ter_cutoff=TER_CUTOFF, loop_cutoff=LOOP_CUTOFF)
            else:
                empty_structures = write_u_receptor_ligand_pdbs(complexes, modeller_dir, ions_cofactors_dir, ter_cutoff, loop_cutoff)

                if empty_structures:
                    print(f"Found {len(empty_structures)} complexes with empty structures.")
                    print("Check the log for details.")
                
                    # Optionally, write the empty structures to a file for further analysis
                    with open("empty_structures.txt", "w") as f:
                        for struct in empty_structures:
                            f.write(f"Complex: {struct['name']}\n")
                            f.write(f"  Receptor empty: {struct['receptor_empty']}\n")
                            f.write(f"  Ligand empty: {struct['ligand_empty']}\n")
                            f.write(f"  Receptor PDBs: {', '.join(struct['receptor_pdbs'])}\n")
                            f.write(f"  Ligand PDBs: {', '.join(struct['ligand_pdbs'])}\n\n")
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            error_message = ''.join(lines)
            logging.error(f"Error processing complex {dir_name}: {str(e)}, \n{error_message}")
            continue

logging.basicConfig(level=logging.DEBUG)

print("Processing bound complexes...")
process_complexes(b_complexes, is_bound=True)

print("\nProcessing unbound complexes...")
# Add debugging information
print(f"Number of unbound complexes: {len(u_complexes)}")
print("Sample of unbound complexes:")
for i, (name, complex_id) in enumerate(u_complexes.items()):
    print(f"{name}: {complex_id}")
    if i >= 4:  # Print only the first 5 samples
        break
process_complexes(u_complexes, is_bound=False)

print("\nGenerating AMBER topologies...")
generate_prmtop(args.cofactors_frcmod_dir)

print("Done")

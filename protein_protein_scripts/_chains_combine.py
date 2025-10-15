"""
to combines chains into protein.
Include only term_cutoff modelled residues at the termini.
A modelled loop will not be included if it is longer than loop_cutoff.
If so, "TER" will be inserted to split the chain
"""

from __future__ import print_function

import os
import copy
import glob
import traceback
import logging
import re
import sys

MODELLED_PDB_SUBFIX = "_modelled.pdb"
MODELLED_RESIDUES = "REMARK  MODELLED RESIDUES:"
ATOM = "ATOM"
HETATM = "HETATM"
TER = "TER"

LIGAND_OUT = "ligand_modelled.pdb"
RECEPTOR_OUT = "receptor_modelled.pdb"
LIGAND_RES_MINIMIZE = "ligand_minimize_list.dat"
RECEPTOR_RES_MINIMIZE = "receptor_minimize_list.dat"

class ChainCombine(object):
    def __init__(self, pdb_id, chains, modelling_dir, ions_cofactors_files, unbound_pdbs=None, is_ligand=False):
        self._pdb_id = pdb_id
        self._chains = chains
        self._ions_cofactors_files = ions_cofactors_files
        self._unbound_pdbs = unbound_pdbs
        self._is_ligand = is_ligand

        if unbound_pdbs:
            logging.info(f"Loading unbound PDBs for {pdb_id}")
            self._original_pdb_data = self._load_unbound_pdbs(unbound_pdbs, chains, chains if is_ligand else [])
        else:
            logging.info(f"Loading bound chains for {pdb_id}")
            self._original_pdb_data = self._load_chains(pdb_id, chains, modelling_dir)
        for chain, data in self._original_pdb_data.items():
            logging.debug(f"Loaded data for chain {chain}: {data.keys()}")
            logging.debug(f"modelled_residues for chain {chain}: {data['modelled_residues']}")

        # Initialize _trimed_pdb_data
        self._trimed_pdb_data = {}

    def trim_residues(self, ter_cutoff, loop_cutoff):
        for chain in self._chains:
            logging.debug(f"Processing chain: {chain}")
            original_modelled_residues = self._original_pdb_data[chain]["modelled_residues"]
            logging.debug(f"Original modelled_residues: {original_modelled_residues}")
            modelled_residues = self._trim_residues(original_modelled_residues, ter_cutoff, loop_cutoff) 
            logging.debug(f"Trimmed modelled_residues: {modelled_residues}")
            self._trimed_pdb_data[chain] = {"modelled_residues": modelled_residues}
    
            self._trimed_pdb_data[chain]["residues_to_minimize"] = self._residues_to_minimize(modelled_residues)
    
            logging.debug(f"Atoms before trimming: {len(self._original_pdb_data[chain]['atoms'])}")
            self._trimed_pdb_data[chain]["atoms"] = self._trim_atoms(modelled_residues, 
                    self._original_pdb_data[chain]["atoms"], self._original_pdb_data[chain]["residues"])
            logging.debug(f"Atoms after trimming: {len(self._trimed_pdb_data[chain]['atoms'])}")
            
            self._trimed_pdb_data[chain]["residues"] = self._count_residues(self._trimed_pdb_data[chain]["atoms"])
            logging.debug(f"Residues after trimming: {len(self._trimed_pdb_data[chain]['residues'])}")
        return None

    def combine(self):
        atoms = []
        residues_to_minimize = []
        shift = 0
        for chain in self._chains:
            atoms.extend(self._trimed_pdb_data[chain]["atoms"])
            atoms.append("TER")
            residues_to_minimize.extend([ r+shift for r in self._trimed_pdb_data[chain]["residues_to_minimize"] ])
            shift += len(self._trimed_pdb_data[chain]["residues"])

        self._combined_pdb_data = {"atoms":copy.deepcopy(atoms),
                                   "residues_to_minimize":residues_to_minimize, "nresidues":shift}
        self._combined_pdb_data["atoms"] = self._change_resid_sequentially()
        self._combined_pdb_data["natoms"] = self._count_atoms(self._combined_pdb_data["atoms"])

        ions_cofactors = self._load_ions_cofactors(self._ions_cofactors_files)
        if ions_cofactors is not None:
            ic_atoms = []
            ic_natoms = 0
            ic_nresidues = 0
            for chain in ions_cofactors.keys():
                ic_atoms.extend(ions_cofactors[chain]["atoms"])
                ic_natoms += ions_cofactors[chain]["natoms"]
                ic_nresidues += ions_cofactors[chain]["nresidues"]
            
            self._combined_pdb_data["atoms"].extend(ic_atoms)
            self._combined_pdb_data["natoms"] += ic_natoms
            self._combined_pdb_data["nresidues"] += ic_nresidues
        return None

    def write_pdb(self, out=None):
        if out==None:
            out = self._pdb_id + "".join(self._chains) + "_modelled.pdb"

        with open(out, "w") as F:
            header = "REMARK NRESIDUES %d\n"%self._combined_pdb_data["nresidues"]
            header += "REMARK NATOMS %d\n"%self._combined_pdb_data["natoms"]
            header += "REMARK MINIMIZE THESE "
            nres_to_minimize = len(self._combined_pdb_data["residues_to_minimize"])
            for i in range(nres_to_minimize):
                header += "%5d"%self._combined_pdb_data["residues_to_minimize"][i]
                if (i+1)%10 == 0 and i < nres_to_minimize-1:
                    header += "\nREMARK MINIMIZE THESE"
            F.write(header + "\n")
            for line in self._combined_pdb_data["atoms"]:
                F.write(line + "\n")
        return None

    def get_nresidues(self):
        return self._combined_pdb_data["nresidues"]

    def get_natoms(self):
        return self._combined_pdb_data["natoms"]

    def write_residues_to_minimize(self, file):
        nresidues = self._combined_pdb_data["nresidues"]
        out_str = "# nresidues %d\n"%nresidues

        residues = self._combined_pdb_data["residues_to_minimize"]
        residues = ["%d"%r for r in residues]
        out_str += "\n".join(residues)
        open(file, "w").write(out_str)
        return None

    def _load_chains(self, pdb_id, chains, modelling_dir):
        chains_pdb_data = {}
        for chain in chains:
            chains_pdb_data[chain] = self._load_chain(pdb_id, chain, modelling_dir)
        return chains_pdb_data

    def is_empty(self):
        return self._combined_pdb_data["natoms"] == 0 or self._combined_pdb_data["nresidues"] == 0

    def _load_chain(self, pdb_id, chain, modelling_dir, is_unbound=False, is_ligand=False):
        assert len(chain) == 1, "chain must be a single letter"
        if is_unbound:
            prefix = 'l' if is_ligand else 'r'
            infile = os.path.join(modelling_dir, f"{pdb_id}_{prefix}_u{chain}_modelled.pdb")
        else:
            infile = os.path.join(modelling_dir, f"{pdb_id}{chain}_modelled.pdb")

        logging.debug(f"Loading {'unbound' if is_unbound else 'bound'} {'ligand' if is_ligand else 'receptor'} chain {chain} for PDB {pdb_id} from file: {infile}")

        if not os.path.exists(infile):
            logging.error(f"File not found: {infile}")
            return None

        pdb_data = {"modelled_residues": {"nter": [], "loops": [], "cter": []}}
        
        with open(infile, "r") as F:
            lines = F.readlines()  # Read all lines at once
        
        for line in lines:
            if MODELLED_RESIDUES in line:
                modelled_res = eval(line.strip(MODELLED_RESIDUES))
                logging.debug(f"Parsed modelled_residues: {modelled_res}")
                if isinstance(modelled_res, list):
                    pdb_data["modelled_residues"]["loops"] = modelled_res
                elif isinstance(modelled_res, dict):
                    pdb_data["modelled_residues"] = modelled_res
                else:
                    logging.error(f"Unexpected modelled_residues format: {type(modelled_res)}")
                break
        
        pdb_data["atoms"] = [line.strip() for line in lines if line.startswith(ATOM)]
        
        res_list = self._count_residues(pdb_data["atoms"])
        
        # Adjust N-terminal and C-terminal residues if necessary
        if pdb_data["modelled_residues"]["loops"]:
            if pdb_data["modelled_residues"]["loops"][0][0] == res_list[0]:
                pdb_data["modelled_residues"]["nter"].append(pdb_data["modelled_residues"]["loops"].pop(0))
            if pdb_data["modelled_residues"]["loops"] and pdb_data["modelled_residues"]["loops"][-1][-1] == res_list[-1]:
                pdb_data["modelled_residues"]["cter"].append(pdb_data["modelled_residues"]["loops"].pop())
        
        logging.debug(f"Final modelled_residues: {pdb_data['modelled_residues']}")
        
        pdb_data["residues"] = res_list
        return pdb_data

    def _load_unbound_pdbs(self, unbound_pdbs, receptor_chains, ligand_chains):
        """
        Load data from multiple unbound PDB files
        """
        pdb_data = {}
        for pdb_file, chain in zip(unbound_pdbs, self._chains):
            if os.path.exists(pdb_file):
                pdb_id = os.path.basename(pdb_file)[:4]
                is_ligand = chain in ligand_chains
                pdb_data[chain] = self._load_chain(pdb_id, chain, os.path.dirname(pdb_file), is_unbound=True, is_ligand=is_ligand)
                if pdb_data[chain] is None:
                    logging.error(f"Failed to load data for unbound PDB {pdb_file}")
                    return None
            else:
                logging.error(f"PDB file not found: {pdb_file}")
                return None
        return pdb_data

    def _process_pdb_file(self, pdb_file):
        """
        Process a single PDB file and extract relevant information
        """
        pdb_data = {"modelled_residues": {"nter": [], "loops": [], "cter": []}, "atoms": [], "residues": []}
        
        with open(pdb_file, "r") as F:
            for line in F:
                if line.startswith(MODELLED_RESIDUES):
                    pdb_data["modelled_residues"] = eval(line.strip(MODELLED_RESIDUES))
                elif line.startswith(ATOM):
                    pdb_data["atoms"].append(line.strip())

        pdb_data["residues"] = self._count_residues(pdb_data["atoms"])

        return pdb_data

    def _count_residues(self, atom_list):
        res_list = set([ int(atom[22:30]) for atom in atom_list if atom.startswith(ATOM) ])
        res_list = sorted(res_list)
        return res_list

    def _count_atoms(self, atom_list):
        count = 0
        for atom in atom_list:
            if ATOM in atom:
                count += 1
        return count

    def _trim_residues(self, original_modelled_residues, ter_cutoff, loop_cutoff):
        logging.debug(f"Entering _trim_residues with: {original_modelled_residues}")
        
        if not isinstance(original_modelled_residues, dict):
            logging.error(f"modelled_residues is not a dictionary: {type(original_modelled_residues)}")
            # Convert list to expected dictionary structure
            modelled_residues = {"nter": [], "loops": original_modelled_residues, "cter": []}
        else:
            modelled_residues = copy.deepcopy(original_modelled_residues)
    
        # Process N-terminal residues
        if "nter" in modelled_residues and modelled_residues["nter"]:
            begin, end = modelled_residues["nter"][0]
            if end - begin + 1 > ter_cutoff:
                modelled_residues["nter"] = [(end - ter_cutoff + 1, end)]
    
        # Process C-terminal residues
        if "cter" in modelled_residues and modelled_residues["cter"]:
            begin, end = modelled_residues["cter"][0]
            if end - begin + 1 > ter_cutoff:
                modelled_residues["cter"] = [(begin, begin + ter_cutoff - 1)]
    
        # Process loops
        missing_loops = []
        modelled_loops = []
        if "loops" in modelled_residues:
            for begin, end in modelled_residues["loops"]:
                if end - begin + 1 > loop_cutoff:
                    missing_loops.append((begin, end))
                else:
                    modelled_loops.append((begin, end))
        
        modelled_residues["loops"] = modelled_loops
        modelled_residues["missing_loops"] = missing_loops
    
        logging.debug(f"Exiting _trim_residues with: {modelled_residues}")
        return modelled_residues

    def _trim_atoms(self, modelled_residues, atoms, residue_list):
        """
        :param modelled_residues: dic with keys "nter", "cter", "loops" and "missing_loops"
        :param atoms: list of ATOM lines in pdb
        :param residue_list: list of str
        :return:
        """
        missing_res = []
        if "missing_loops" in modelled_residues:
            for missing in modelled_residues["missing_loops"]:
                missing_res.extend(range(missing[0], missing[1]+1))

        if len(modelled_residues.get("nter", [])) == 1:
            first_res_id = modelled_residues["nter"][0][0]
        else:
            first_res_id = 1

        if len(modelled_residues.get("cter", [])) == 1:
            last_res_id  = modelled_residues["cter"][0][1]
        else:
            last_res_id  = len(residue_list)

        trimed_atoms = []
        for line in atoms:
            resid = int(line[22:30])
            if (first_res_id <= resid <= last_res_id) and (resid not in missing_res):
                trimed_atoms.append(line)

        trimed_atoms = self._insert_ter(trimed_atoms)
        return trimed_atoms

    def _insert_ter(self, atoms):
        """
        insert a "TER" if resid not continuous
        """
        new_atoms = []
        for i in range( len(atoms) - 1):
            new_atoms.append(atoms[i])

            current_resid = int(atoms[i][22:30])
            next_resid    = int(atoms[i+1][22:30])
            if (current_resid != next_resid) and (current_resid+1 < next_resid):
                new_atoms.append("TER")

        new_atoms.append(atoms[-1])
        return new_atoms

    def _residues_to_minimize(self, modelled_residues):
        keys = [key for key in modelled_residues.keys() if key != "missing_loops"]
        residues = []
        for key in keys:
            for begin, end in modelled_residues[key]:
                residues.extend( range(begin, end+1) )
        residues = sorted(residues)
        if len(modelled_residues["nter"]) == 1:
            if modelled_residues["nter"][0][0] != 1:
                shift = 1 - modelled_residues["nter"][0][0]
                residues = [r + shift for r in residues]
        return residues

    def _change_resid(self, atom, new_resid):
        if ATOM not in atom:
            return atom
        entries = atom.split(atom[22:30])
        new_atom = entries[0] + "%4d"%new_resid + " "*4 + entries[1]
        return new_atom

    def _change_resid_sequentially(self):
        atoms = []
        resid = 1
        nlines = len(self._combined_pdb_data["atoms"])
        for i in range( nlines ):
            this_line = self._combined_pdb_data["atoms"][i]
            if ATOM not in this_line:
                atoms.append(this_line)
            else:
                atoms.append( self._change_resid(this_line, resid) )
                this_resid = int(this_line[22:30])
                if i < nlines-1:           # not the last
                    next_line = self._combined_pdb_data["atoms"][i+1]
                    if ATOM in next_line:
                        next_resid = int(next_line[22:30])
                        if next_resid > this_resid:
                            resid += 1
                    else:
                        resid += 1
                else:
                    atoms.append(self._change_resid(this_line, resid))
        return atoms
    
    def _load_ions_cofactors(self, ions_cofactors_files):
        """
        :param ions_cofactors_files: list of str
        :return: None if ions_cofactors_files is empty
        """
        if len(ions_cofactors_files) == 0:
            return None

        for file in ions_cofactors_files:
            pdb_id = os.path.basename(file)[:4]
            if pdb_id != self._pdb_id:
                raise RuntimeError("%s is not from the same pdb with id %s"%(file, self._pdb_id))

        chains = [os.path.basename(file)[4] for file in ions_cofactors_files]
        if len(set(chains).intersection(self._chains)) == 0:
            return None

        pdb_data = {}
        for file in ions_cofactors_files:
            chain = os.path.basename(file)[4]
            if chain in self._chains:
                pdb_data[chain] = self._load_ions_cofactors_file(file)
        return pdb_data

    def _load_ions_cofactors_file(self, file):
        pdb_data = {}
        with open(file, "r") as F:
            pdb_data["atoms"] = [line.strip() for line in F if (line.startswith(HETATM) or line.startswith(TER))]
        pdb_data["natoms"]   = len([line for line in pdb_data["atoms"] if line.startswith(HETATM)])
        pdb_data["nresidues"] = len([line for line in pdb_data["atoms"] if line.startswith(TER)])
        return pdb_data


def parse_modelling_dir(complex_id, modeller_dir):
    """
    :param complex_id: tuple of (pdb_id, chains1, chains2)
    :param modeller_dir: str
    :return:
    """
    pdb_id, chains1, chains2 = complex_id
    modelling_dir = os.path.join(modeller_dir, pdb_id)
    if not os.path.isdir(modelling_dir):
        raise RuntimeError("%s does not exist"%modelling_dir)
    modelled_pdbs = glob.glob( os.path.join(modelling_dir, "*_modelled.pdb") )
    modelled_pdbs = [os.path.basename(file) for file in modelled_pdbs]

    if pdb_id != modelled_pdbs[0][:4]:
        raise RuntimeError("%s does not exist"%pdb_id)
    all_chains = set([pdb[4] for pdb in modelled_pdbs])

    mod_chains1 = False
    for c in chains1:
        if c not in all_chains:
            print("chain %s does not exist in %s"%(c, pdb_id))
            mod_chains1 = True
    if mod_chains1:
        chains1 = [c for c in chains1 if c in all_chains]

    mod_chains2 = False
    for c in chains2:
        if c not in all_chains:
            print("chain %s does not exist in %s"%(c, pdb_id))
            mod_chains2 = True
    if mod_chains2:
        chains2 = [c for c in chains2 if c in all_chains]

    return pdb_id, chains1, chains2, modelling_dir

def clean_chain_id(chain):
    """Remove NMR structure indicators from chain IDs"""
    return re.sub(r'\([0-9]+\)', '', chain)

def write_b_receptor_ligand_pdbs(complexes, modeller_dir, ions_cofactors_dir, ter_cutoff=10, loop_cutoff=20):
    """
    complexes:  dict returned by _affinity_data.AffinityData.get_bound_complexes
    """
    print("Combining chains to form ligands and receptors for ...")
    for name, complex_id in complexes.items():
        # Do bound structures
        print(name)

        if not os.path.isdir(name):
            os.makedirs(name)

        pdb_id, chains1, chains2, modelling_dir = parse_modelling_dir(complex_id, modeller_dir)

        ic_dir = os.path.join(ions_cofactors_dir, name)
        if not os.path.isdir(ic_dir):
            ions_cofactors_files = []
        else:
            ions_cofactors_files = glob.glob(os.path.join(ic_dir, pdb_id + "*" + ".pdb"))

        partners = [ ChainCombine(pdb_id, chains, modelling_dir, ions_cofactors_files) for chains in (chains1, chains2) ]
        for p in partners:
            p.trim_residues(ter_cutoff, loop_cutoff)
            p.combine()
        partners.sort(key=lambda c: c.get_nresidues())
        partners[0].write_pdb(out = os.path.join(name, LIGAND_OUT))
        partners[0].write_residues_to_minimize(os.path.join(name, LIGAND_RES_MINIMIZE))

        partners[1].write_pdb(out = os.path.join(name, RECEPTOR_OUT))
        partners[1].write_residues_to_minimize(os.path.join(name, RECEPTOR_RES_MINIMIZE))

    print("Done combining chains")
    print("")
    return None

def write_u_receptor_ligand_pdbs(complexes, modeller_dir, ions_cofactors_dir, ter_cutoff=10, loop_cutoff=20):
    logging.info("Combining chains to form ligands and receptors for unbound structures...")
    empty_structures = []

    for name, complex_data in complexes.items():
        logging.info(f"Processing unbound complex: {name}")
        try:
            bound_name, receptor_pdb, receptor_chains, ligand_pdb, ligand_chains = complex_data
            bound_name = bound_name[:4].lower()

            receptor_chains = [clean_chain_id(chain) for chain in receptor_chains]
            ligand_chains = [clean_chain_id(chain) for chain in ligand_chains]

            unbound_name = f"{name}_U"
            if not os.path.isdir(unbound_name):
                os.makedirs(unbound_name)

            receptor_pdbs = [os.path.join(modeller_dir, f"{bound_name}_u", f"{bound_name}_r_u{chain}_modelled.pdb") for chain in receptor_chains]
            ligand_pdbs = [os.path.join(modeller_dir, f"{bound_name}_u", f"{bound_name}_l_u{chain}_modelled.pdb") for chain in ligand_chains]

            logging.debug(f"Receptor PDBs: {receptor_pdbs}")
            logging.debug(f"Ligand PDBs: {ligand_pdbs}")

            if not all(os.path.exists(pdb) for pdb in receptor_pdbs + ligand_pdbs):
                missing_pdbs = [pdb for pdb in receptor_pdbs + ligand_pdbs if not os.path.exists(pdb)]
                logging.warning(f"Warning: Not all unbound PDB files found for {name}. Missing: {missing_pdbs}")
                continue

            ic_dir = os.path.join(ions_cofactors_dir, name)
            ions_cofactors_files = glob.glob(os.path.join(ic_dir, f"{bound_name}*.pdb")) if os.path.isdir(ic_dir) else []

            receptor = ChainCombine(receptor_pdb, receptor_chains, modeller_dir, ions_cofactors_files, unbound_pdbs=receptor_pdbs, is_ligand=False)
            receptor.trim_residues(ter_cutoff, loop_cutoff)
            receptor.combine()

            ligand = ChainCombine(ligand_pdb, ligand_chains, modeller_dir, ions_cofactors_files, unbound_pdbs=ligand_pdbs, is_ligand=True)
            ligand.trim_residues(ter_cutoff, loop_cutoff)
            ligand.combine()

            if receptor.is_empty() or ligand.is_empty():
                empty_structures.append({
                    "name": name,
                    "receptor_empty": receptor.is_empty(),
                    "ligand_empty": ligand.is_empty(),
                    "receptor_pdbs": receptor_pdbs,
                    "ligand_pdbs": ligand_pdbs
                })
                logging.warning(f"Empty structure detected for {name}. Skipping...")
                continue

            receptor.write_pdb(out=os.path.join(unbound_name, RECEPTOR_OUT))
            receptor.write_residues_to_minimize(os.path.join(unbound_name, RECEPTOR_RES_MINIMIZE))

            ligand.write_pdb(out=os.path.join(unbound_name, LIGAND_OUT))
            ligand.write_residues_to_minimize(os.path.join(unbound_name, LIGAND_RES_MINIMIZE))

            logging.info(f"Successfully processed {name}")

        except Exception as e:
            logging.error(f"Error processing complex {name}: {str(e)}")
            logging.debug(traceback.format_exc())

    logging.info("Done combining chains for unbound structures")

    if empty_structures:
        logging.warning(f"Found {len(empty_structures)} complexes with empty structures:")
        for struct in empty_structures:
            logging.warning(f"Complex: {struct['name']}")
            logging.warning(f"  Receptor empty: {struct['receptor_empty']}")
            logging.warning(f"  Ligand empty: {struct['ligand_empty']}")
            logging.warning(f"  Receptor PDBs: {struct['receptor_pdbs']}")
            logging.warning(f"  Ligand PDBs: {struct['ligand_pdbs']}")

    return empty_structures

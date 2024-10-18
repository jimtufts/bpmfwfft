
"""
define class to handle the affinity data pushlished at
http://bmm.crick.ac.uk/~bmmadmin/Affinity/
"""

import csv
import numpy as np
import pandas as pd
import re
import logging

class AffinityData(object):
    """
    load the affinity data files
    give interface to access the data
    """
    def __init__(self, affinity_data_files):
        self._data_frame = self._load_tsvfiles(affinity_data_files)
        self._column_tags = ["Complex PDB", "Unbound PDB Protein A", "Unbound PDB Protein B"]
        self._clean_complex_names()

    def _clean_complex_names(self):
        """Clean up complex names by removing trailing spaces and special characters."""
        self._data_frame[self._column_tags[0]] = self._data_frame[self._column_tags[0]].apply(self._clean_name)

    @staticmethod
    def _clean_name(name):
        """Remove trailing spaces and special characters from a name."""
        return re.sub(r'[^a-zA-Z0-9_:]', '', name.strip())

    def get_complex_names(self):
        return list(self._data_frame[self._column_tags[0]])
        """
        return a set of unique pdb ids to be downloaded
        """
        ids = []
        for col in self._column_tags:
            ids.extend( list(self._data_frame[col].values) )
        ids = [id.split("_")[0].lower() for id in ids]
        return set(ids)

    def get_bound_complexes(self):
        """
        Each name is unique, and can be used to identify the complex.
        The structure of names is "XXXX_AB:FGH", where "XXXX" is 4-letter pdb id,
        "AB" is chains of protein A, "FGH" is chains of protein B.

        return a dic { name:(pdb_id (str), chains1, chains2 ) }
                    name: str; chains1:  list of str; chains2:  list of str
        """
        names = list(self._data_frame[self._column_tags[0]])
        complex_chains = {}
        for name in names:
            pdb_id, chains = name.split("_")
            pdb_id = pdb_id.lower()
            chains1, chains2 = chains.split(":")
            chains1 = [c for c in chains1]
            chains2 = [c for c in chains2]
            if len(chains1) == 0 or len(chains2) == 0:
                raise RuntimeError("%s does not have one or both binding partners"%name)
            complex_chains[name] = (pdb_id, chains1, chains2)
        return complex_chains

    def get_unbound_complexes(self):
        names = self.get_complex_names()
        unbound_complexes = {}

        def parse_pdb_info(pdb_string):
            parts = pdb_string.split('_')
            pdb_id = parts[0].lower()
            chains = list(parts[1]) if len(parts) > 1 else ["A"]
            return pdb_id, chains

        for name in names:
            row = self._data_frame[self._column_tags[0]] == name
            if row.sum() == 0:
                logging.warning(f"No data found for complex {name}")
                continue

            try:
                matching_rows = self._data_frame[row]
                if matching_rows.empty:
                    logging.warning(f"No matching rows found for complex {name}")
                    continue
                
                bound_name = matching_rows[self._column_tags[0]].values[0]
                pdb_a = matching_rows[self._column_tags[1]].values[0]
                pdb_b = matching_rows[self._column_tags[2]].values[0]

                pdb_id_a, chains_a = parse_pdb_info(pdb_a)
                pdb_id_b, chains_b = parse_pdb_info(pdb_b)

                unbound_complexes[name] = (bound_name, pdb_id_a, chains_a, pdb_id_b, chains_b)
                logging.debug(f"Processed unbound complex {name}: {pdb_id_a}, {chains_a}, {pdb_id_b}, {chains_b}")

            except Exception as e:
                logging.error(f"Error processing unbound complex {name}: {str(e)}")
                continue

        return unbound_complexes

    def get_col_names(self):
        return list(self._data_frame.columns)

    def get_data_from_col(self, col_name):
        complex_names = list(self._data_frame[self._column_tags[0]])
        col = {}
        for name in complex_names:
            row = self._data_frame[self._column_tags[0]] == name
            value = self._data_frame[col_name][row].values
            if value.shape != (1,):
                raise RuntimeError("There are more than one %s at %s"%(col_name, name))
            col[name] = value[0]
        return col

    def get_complex_names(self):
        complex_names = list(self._data_frame[self._column_tags[0]])
        complex_names = [name.rstrip('*').strip() for name in complex_names]
        return complex_names

    def get_dG(self):
        """
        return a dic, dG[complex_name] -> dG
        """
        dG = self.get_data_from_col("dG")
        for name in dG.keys():
            dG[name] = np.float(dG[name])
        return dG

    def _load_tsvfile(self, file):
        """
        load tsv file
        file is a str
        return a pandas.DataFrame object
        TODO: use pd.read_table
        """
        with open(file) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            # next(tsvreader)                            # ignore the first line
            data_fields = next(tsvreader)
            # in the the file, there are two columns with the same name "Unbound PDB"
            # make each field in data_fields unique
            for i in range(len(data_fields)):
                if data_fields[i] == "Unbound PDB":
                    data_fields[i] += " " + data_fields[i+1]

            # records is a list of dict
            records = []
            for line in tsvreader:
                if len(line) != len(data_fields):
                    raise RuntimeError("line %s does not have the same len as %s" %("\t".join(line), "\t".join(data_fields)))
                # put each line into a dict
                tmp = {}
                for i in range( len(data_fields) ):
                    tmp[data_fields[i]] = line[i]
                records.append(tmp)
        return pd.DataFrame(records)
    
    def _load_tsvfiles(self, files):
        """
        load multiple tsv files
        return a concatenated pd.DataFrame
        files is a list of str
        """
        assert len(files) == len(set(files)), "some files have the same name"
        frames = [self._load_tsvfile(file) for file in files]
        return pd.concat(frames)
    



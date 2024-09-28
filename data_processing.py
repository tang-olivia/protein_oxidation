import requests
import pandas as pd
from biopandas.pdb import PandasPdb
from Bio.PDB import *
from Bio.Align import PairwiseAligner
import torch
import numpy as np

from IPython import embed

organism_dict = {
    "HUMAN": "Homo sapiens",
    "RAT": "Rattus norvegicus",
    "BACSU": "Bacillus subtilis",
    "ARATH": "Arabidopsis thaliana",
    "MOUSE": "Mus musculus",
    "YEASB": "Saccaromyces cerevisiae",
    "RABIT": "Oryctolagus cuniculus",
    "SCHPO": "Schizosaccharomyces pombe",
    "ECO27": "Escherichia coli K-12",
    "BOVIN": "Bos taurus",
    "ECOLI": "Escherichia coli",
    "YEAST": "Saccharomyces cerevisiae",
    "HORVU": "Hordeum vulare"
}

aa_mapping = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def get_pdb_id(gene, species):
    """
    Use RSCB API to return one PDB ID corresponding to the gene and species of interest
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.rcsb_gene_name.value",
                        "operator": "exact_match",
                        "value": gene
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.scientific_name",
                        "operator": "exact_match",
                        "value": organism_dict[species]
                    }
                }
            ]
        },
        "return_type": "entry"
    }
    response = requests.post(url, json=query)
    if response.status_code == 200:
        data = response.json()
        pdb_ids = [result['identifier'] for result in data['result_set']]
        return pdb_ids
    
def global_seqalign(chain_seq_str, target_sequence, chain_residues):
    valid_chars = "-GAVLITSMCPFYWHKRDENQX"
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -2
    alignments = aligner.align(chain_seq_str, target_sequence)
    alignment = None
    best_score = float('-inf')
    for agmt in alignments:
        if agmt.score > best_score:
            alignment = agmt
            best_score = agmt.score
    temp = alignment.format().split("\n")
    aligned_seq = ""
    for part in temp:
        if part.startswith("query"):
            for char in part:
                if char in valid_chars:
                    aligned_seq += char
    aligned_list = [len(elt) for elt in aligned_seq.split("-") if elt != ""]
    if (len(chain_seq_str) == len(aligned_seq)) and (max(aligned_list) >= 8): 
        count = 0
        for i, residue in enumerate(aligned_seq):
            if residue != "-" and residue == chain_seq_str[i]:
                count += 1
                if count == 11 and residue == "C":
                    return (True, i)

def search_sequence(pdb_file, target_sequence):
    parser = PDBParser(QUIET=True)
    protein_structure = parser.get_structure("structure", pdb_file)
    ppb = PPBuilder()

    for polypeptide in ppb.build_peptides(protein_structure):
        chain_seq = polypeptide.get_sequence()
        chain_residues = [res.id[1] for res in polypeptide]

        chain_seq_str = str(chain_seq)
        if target_sequence in chain_seq_str:
            start_index = chain_seq_str.find(target_sequence)
            res_num = chain_residues[start_index] + 10 - chain_residues[0]
            return (True, res_num) 
        alignment_res = global_seqalign(chain_seq_str, target_sequence, chain_residues)
        if alignment_res is not None:
            return alignment_res

def search_balosequence(protein_df, position, target_sequence):
    sequence = ""
    residues = []
    for _, row in protein_df.iterrows():
        if row["residue_name"] in aa_mapping:
            sequence += aa_mapping[row["residue_name"]]
        else:
            sequence += "X"
        residues.append(row["residue_number"])
    if residues[0] > position:
        position += residues[0]
    if residues[0] < position:
        position -= residues[0]
    if (position < len(sequence) and position >= 0 and sequence[position] == "C" and sequence[position-5:position+6] == target_sequence[5:16]):
        return (True, position)
    alignment_res = global_seqalign(sequence, target_sequence, residues)
    if alignment_res is not None:
        return alignment_res


def create_chain_map(chain_ids):
    chain_map = {}
    current = 1.0
    for chain_id in chain_ids:
        chain_map[chain_id] = current
        current += 1
    return chain_map

def create_proteinmpnn_parameters(protein_df):
    # sort df by chain_id and residue_number
    protein_df = protein_df.sort_values(by=["chain_id", "residue_number"])

    # create chain to float mapping
    chain_ids = protein_df["chain_id"].unique().tolist()
    chain_map = create_chain_map(chain_ids)

    # initialize tensors
    X, mask, residue_idx, chain_encoding_all = [], [], [], []
    mask.append([1.0] * len(protein_df))

    current_residx = 0
    current_chain = None
    first = True
    for index, row in protein_df.iterrows():
        X.append([row["x_coord"], row["y_coord"], row["z_coord"]])
        if first:
            residue_idx.append(current_residx)
            current_chain = row["chain_id"]
            first = False
        elif row["chain_id"] != current_chain:
            current_residx += 101
            current_chain = row["chain_id"]
            residue_idx.append(current_residx)
        else:
            prev_index = protein_df.index[protein_df.index.get_loc(index)-1]
            difference = row["residue_number"] - protein_df.loc[prev_index, "residue_number"]
            current_residx += difference
            residue_idx.append(current_residx)
        chain_encoding_all.append(chain_map[row["chain_id"]])

    return {
        "X": torch.tensor([X]),
        "mask": torch.tensor(mask),
        "residue_idx": torch.tensor([residue_idx]),
        "chain_encoding_all": torch.tensor([chain_encoding_all])
    }

def create_esm_params(protein_df):
    sequence = ""
    for _, row in protein_df.iterrows():
        if row["residue_name"] in aa_mapping:
            sequence += aa_mapping[row["residue_name"]]
        else:
            sequence += "X"
    return sequence


def prepare_rcparams(path):
    df = pd.read_csv(path, sep="\t")
    pdb_bank, parameters, sequences, cys_positions, labels = {}, [], [], [], []
    for _, row in df.iterrows():
        if row["Seq_id"] not in pdb_bank:
            gene, species = row["Seq_id"].split("_")
            pdb_bank[row["Seq_id"]] = get_pdb_id(gene, species)
        if pdb_bank[row["Seq_id"]] is None:
            continue
        for pdb_id in pdb_bank[row["Seq_id"]]:
            pdbl = PDBList()
            pdb_path = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
            try:
                fulldf = PandasPdb().read_pdb(pdb_path)
                fulldf = fulldf.df["ATOM"]
                protein_df = fulldf[fulldf["atom_name"] == "CA"]
                protein_df = protein_df[["residue_name", "chain_id", "residue_number", "x_coord", "y_coord", "z_coord"]]
            except FileNotFoundError:
                continue
            sequence_tuple = search_sequence(pdb_path, row["Flanking sequence"])
            if sequence_tuple is not None:
                parameters.append(create_proteinmpnn_parameters(protein_df))
                sequences.append(create_esm_params(protein_df))
                cys_positions.append(sequence_tuple[1])
                if row["Label"] == 1:
                    labels.append(1)
                else:
                    labels.append(0)
                break
    return parameters, sequences, cys_positions, labels

def prepare_baloparams(path):
    df = pd.read_csv(path, sep="\t")
    pdb_bank, parameters, sequences, cys_positions, labels = {}, [], [], [], []
    for _, row in df.iterrows():
        if row["Seq_id"] not in pdb_bank:
            pdb_id, select_chain = tuple(row["Seq_id"].split("_"))
            try:
                pdbl = PDBList()
                pdb_path = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
                fulldf = PandasPdb().read_pdb(pdb_path)
                fulldf = fulldf.df["ATOM"]
                fulldf = fulldf[(fulldf["atom_name"] == "CA") & (fulldf["chain_id"] == select_chain)]
                fulldf = fulldf[["residue_name", "chain_id", "residue_number", "x_coord", "y_coord", "z_coord"]]
                pdb_bank[row["Seq_id"]] = (pdb_path, fulldf)
            except FileNotFoundError:
                pdb_bank[row["Seq_id"]] = None
                continue
        if pdb_bank[row["Seq_id"]] is None:
            continue
        pdb_file, protein_df = pdb_bank[row["Seq_id"]]
        sequence_tuple = search_balosequence(protein_df, row["Position"], row["Flanking_sequence"])
        if sequence_tuple is not None:
            parameters.append(create_proteinmpnn_parameters(protein_df))
            sequences.append(create_esm_params(protein_df))
            cys_positions.append(sequence_tuple[1])
            if row["Label"] == 1:
                labels.append(1)
            else:
                labels.append(0)
    return parameters, sequences, cys_positions, labels


def prepare_parameters(rc_path, balo_path):
    rc_params, rc_seqs, rc_positions, rc_labels = prepare_rcparams(rc_path)
    balo_params, balo_seqs, balo_positions, balo_labels = prepare_baloparams(balo_path)
    return rc_params + balo_params, rc_seqs + balo_seqs, rc_positions + balo_positions, rc_labels + balo_labels

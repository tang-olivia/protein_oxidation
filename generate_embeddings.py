import json, time, os, sys, glob
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils import *
#from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
from data_processing import *
from transformers import EsmTokenizer, EsmModel

# clone Github repository 
if not os.path.isdir("ProteinMPNN"):
  os.system("git clone -q https://github.com/dauparas/ProteinMPNN.git")
sys.path.append('/ProteinMPNN/')

def load_pretrained_mpnn():
  device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
  #v_48_010=version with 48 edges 0.10A noise
  model_name = "v_48_002" #@param ["v_48_002", "v_48_020"]

  backbone_noise=0.00

  path_to_model_weights="/om/user/oliviat/ProteinMPNN/ca_model_weights"      
  hidden_dim = 128
  num_layers = 3 
  if path_to_model_weights[-1] != '/':
      path_to_model_weights = path_to_model_weights + '/'
  checkpoint_path = path_to_model_weights + f'{model_name}.pt'

  checkpoint = torch.load(checkpoint_path, map_location=device) 
  print('Number of edges:', checkpoint['num_edges'])
  noise_level_print = checkpoint['noise_level']
  print(f'Training noise level: {noise_level_print}A')
  model = ProteinMPNN(ca_only=True, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
  model.to(device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  print("Model loaded")
  return model

def get_mpnn_embeddings(parameters, cysteine_positions):
  model = load_pretrained_mpnn()
  mpnn_embeddings = []
  for (param, position) in zip(parameters, cysteine_positions):
    extracted = model.extract_node_embeddings(param["X"].to("cuda"), param["mask"].to("cuda"), param["residue_idx"].to("cuda"), param["chain_encoding_all"].to("cuda")).detach().cpu().numpy()
    mpnn_embeddings.append(extracted[0, position, :])
  return np.array(mpnn_embeddings)

def get_esm_embeddings(sequences, cysteine_positions):
  tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
  model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
  model.eval()
  context_embeddings = []
  for (seq, position) in zip(sequences, cysteine_positions):
    inputs = tokenizer(seq, return_tensors="pt", padding=True)
    with torch.no_grad():
      outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    context_embeddings.append(last_hidden_state.detach().cpu()[0, position, :])
  return np.array(context_embeddings)

def get_general_embeddings(rc_path, balo_path):
  mpnn_params, sequences, cysteine_positions, labels = prepare_parameters(rc_path, balo_path)
  upd_mpnn_params, upd_cysteine_positions, upd_sequences, upd_labels = [], [], [], []
  for i, seq in enumerate(sequences):
    if seq[cysteine_positions[i]] == "C":
      upd_mpnn_params.append(mpnn_params[i])
      upd_cysteine_positions.append(cysteine_positions[i])
      upd_labels.append(labels[i])
      upd_sequences.append(seq)
  #mpnn_embeddings = get_mpnn_embeddings(upd_mpnn_params, upd_cysteine_positions)
  esm_embeddings = get_esm_embeddings(upd_sequences, upd_cysteine_positions)
  np.save("embeddings2560.npy", esm_embeddings)
  #np.save("labels640.npy", np.array(upd_labels))

get_general_embeddings("/om/user/oliviat/bioML/RSC758.txt", "/om/user/oliviat/bioML/BALOSCTdb.txt")

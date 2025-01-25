#!/usr/bin/env python
# coding: utf-8

# # Imports, constants

# In[1]:
# standard libs
import argparse
import itertools
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pprint import pprint

# third-party libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# torch related libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# constants
DATA = "../../../data/created_data/seqs100000.tsv"
SEED = 566
# Results and models file paths
ACC_FILE = (
    "../../../data/out_metrics/accuracies_{timestamp}_seq_{config_index}.pkl"
)
CAP_FILE = (
    "../../../data/out_metrics/capacities_{timestamp}_seq_{config_index}.pkl"
)
LOSSES_FILE = (
    "../../../data/out_metrics/losses_{timestamp}_seq_{config_index}.pkl"
)
MODELS_FILE = "../../../data/out_models/models_{timestamp}_seq_{config_index}.pkl"

# fix random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[37]:


##########################################
# 0) tokenizers
##########################################

def read_and_tokenize(file_path):
    EOS_TOKEN = "<EOS>"
    node_edge_vocab = {EOS_TOKEN: 0}  # Start with EOS token
    node_edge_counter = 1  # Start token IDs from 2 to reserve 1 for EOS
    # Step 1: Read the TSV file with variable columns
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split('\t')
            data.append(row)

    # Determine the maximum number of columns dynamically
    max_cols = max(len(row) for row in data)
    column_names = [f"col_{i}" for i in range(max_cols)]
    df = pd.DataFrame(data, columns=column_names).fillna(value=EOS_TOKEN)

    # Step 2: Replace '.' and ',' with spaces, and handle missing columns by filling with "0"
    df = df.applymap(lambda x: x.replace('.', ' ').replace(',', ' '))

    # Step 3: Tokenize nodes/edges uniquely
    # (In addition to this node/edge tokenization, the data was further processed at the word level.
    #  Words were treated as individual tokens, with spaces, underscores, and dots acting as separators
    #  to split compound terms. Tokens were arranged in separate columns, maintaining an unflattened
    #  structure that preserved the sequence hierarchy. An <EOS> token was appended to each sequence
    #  to clearly indicate its termination. To ensure consistent sequence length, zeros were appended
    #  at the end of each sequence, complementing the earlier padding approach .)
    def tokenize_node_edge(value):
        nonlocal node_edge_counter
        if value not in node_edge_vocab:
            node_edge_vocab[value] = node_edge_counter
            node_edge_counter += 1
        return node_edge_vocab[value]

    df_tokenized_node_edge = df.applymap(lambda x: tokenize_node_edge(x))
#     df_tokenized_node_edge[f"col_{len(column_names)}"]=0

    # Step 4: Tokenize word-by-word (using "_" and spaces as separators)
    word_vocab = {EOS_TOKEN: 1}  # Start with EOS token
    word_counter = 2  # Start token IDs from 2 to reserve 1 for EOS

    def tokenize_word_by_word(value):
        nonlocal word_counter
        tokens = []
        if value != "0":
            for word in value.replace("_", " ").replace(".", " ").split():  # Split on spaces and "_"
                if word not in word_vocab:
                    word_vocab[word] = word_counter
                    word_counter += 1
                tokens.append(word_vocab[word])
            tokens.append(word_vocab[EOS_TOKEN])  # Append EOS token
        tokens.append(0)  # Append EOS token
        return tokens

    df_tokenized_word_by_word = df.applymap(lambda x: tokenize_word_by_word(x) if x != "0" else [0])

    # Step 5: Unflatten word-by-word into a new DataFrame
    word_by_word_expanded = []
    for index, row in df_tokenized_word_by_word.iterrows():
        expanded_row = []
        for cell in row:
            if isinstance(cell, list):
                expanded_row.extend(cell)
            else:
                expanded_row.append(cell)
        word_by_word_expanded.append(expanded_row)

    max_words = max(len(row) for row in word_by_word_expanded)
    df_word_by_word_unflattened = pd.DataFrame(word_by_word_expanded, columns=[f"word_{i}" for i in range(max_words)]).fillna(0).astype(int)

    return df, df_tokenized_node_edge, df_word_by_word_unflattened, node_edge_vocab, word_vocab


# In[56]:


##########################################
# 1) Dataset to Differ Node and Edges
##########################################
class NodeEdgeSequenceDataset(Dataset):
    """
    Each item in 'sequences' is a list of integer tokens [n1, e1, n2, e2, ...],
    where nX are node tokens, eX are edge tokens, and 0 is pad if needed.
    'node_mask' is a list of the same length indicating which positions are nodes (1) or not (0).
    """
    def __init__(self, sequences, node_masks, pad_token=0):
        assert len(sequences) == len(node_masks), "Sequences and node_masks must match in length"
        self.sequences = sequences
        self.node_masks = node_masks
        self.pad_token = pad_token
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        mask = torch.tensor(self.node_masks[idx], dtype=torch.long)
        return seq, mask

def collate_fn(batch, pad_token=0):
    """
    Dynamically pads sequences and their node_masks to the longest item in the batch.
    Returns [batch_size, seq_len], [batch_size, seq_len].
    """
    seqs, masks = zip(*batch)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    padded_seqs = []
    padded_masks = []
    for seq, msk in zip(seqs, masks):
        pad_len = max_len - len(seq)
        padded_seqs.append(torch.cat([seq, torch.full((pad_len,), pad_token)]))
        padded_masks.append(torch.cat([msk, torch.full((pad_len,), 0)]))
    return torch.stack(padded_seqs), torch.stack(padded_masks)


##############################################################################
# 2) GPT-like Transformer Model
##############################################################################

# Define the custom activation functions
def get_activation_function(name):
    if name == "default":
        return nn.ReLU()
    elif name == "GELU":
        return nn.GELU()
    elif name == "RReLU":
        return nn.RReLU()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown activation function: {name}")


class GPTLikeModelPos(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=50,
        pad_token=0,
        activation_fn='gelu',
        seed=42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        self.pad_token = pad_token
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        print(vocab_size, d_model, pad_token)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        # Simple trainable positional embedding:
#         self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer "decoder" layers, each with self-attention
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                activation=activation_fn,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        """
        x: [batch_size, seq_len]
        causal_mask: [seq_len, seq_len] for auto-regression
        key_padding_mask: [batch_size, seq_len], True where we want to ignore positions
        """
        bsz, seq_len = x.shape
#         positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
#         # embedding: shape => [batch_size, seq_len, d_model]
#         x = self.embedding(x) + self.pos_emb(positions)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]

        out = x
        for layer in self.layers:
            out = layer(
                tgt=out,
                memory=out,
                tgt_mask=causal_mask,
                memory_mask=None,
                tgt_key_padding_mask=key_padding_mask,
                memory_key_padding_mask=key_padding_mask
            )
        # project to vocab
        logits = self.fc_out(out)  # [batch_size, seq_len, vocab_size]
        return logits
    
    

##########################################
# 3) Utilities: Generate Causal Mask and node mask
##########################################
def generate_causal_mask(seq_len, device):
    """
    Generates an upper-triangular mask [seq_len, seq_len] 
    with True in positions that should NOT be attended (i.e., future tokens).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

def get_node_masks(sequences):
    """
    Generate a mask where nodes are represented by 1s, 
    and edges/padding are represented by 0s.
    
    Parameters:
        sequences (np.ndarray): 2D array of tokenized sequences.
    
    Returns:
        np.ndarray: 2D array of the same shape as sequences with 
                    1s for nodes and 0s for edges/padding.
    """
    # Determine mask for nodes
    node_mask = (sequences > 0) & (np.arange(sequences.shape[1]) % 2 == 0)
    
    # Convert boolean mask to integer (1s and 0s)
    return node_mask.astype(int)

##########################################
# 4) Training: we only compute loss on node positions
##########################################
def train_one_epoch_iterative(
    model,
    data_loader,
    optimizer,
    device,
    pad_token=0
):
    """
    Iterative approach: for each sequence in the batch:
      - For i in [1..(seq_len-1)]:
          feed tokens [0..(i-1)]
          get the distribution for token i
          compute loss vs. the i-th token
          update
    This matches the style "model(inputs) -> next token at out[:, 0, :]" if you
    only keep the last time-step.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    total_loss = 0.0
    correct = 0
    total = 0
    
    train_loader = data_loader
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    for (batch_seqs, batch_node_masks) in tqdm(data_loader):  # Testing on the same dataset
        batch_seqs = batch_seqs.to(device)
        batch_node_masks = batch_node_masks.to(device)
        bsz, seq_len = batch_seqs.shape
        for i in range(2, batch_seqs.shape[1], 2):
            inputs = batch_seqs[:,:i]
            targets = batch_seqs[:,i]
            masks = batch_node_masks[:,i].bool()
            # Forward pass
            optimizer.zero_grad()  # can be placed anywhere before loss.backward
            outputs = model(inputs)
#                     print(outputs.shape)
            outputs = outputs[:, i-1, :]  # Only take the first token prediction
            predicted = torch.argmax(outputs, dim=1)
        
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if not torch.isnan(loss).item():
                total_loss += loss.item()
    return total_loss / len(data_loader)


def train_model(model, 
                data_loader, 
                epochs, 
                lr=1e-3, 
                pad_token=0, 
                device='cpu',  
                max_to_memorize=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses_for_n = []
    accuracies_for_n = []
    capacities_for_n = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_iterative(model, data_loader, optimizer, device, pad_token)
        # Testing on the same data (memorization check)
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for (batch_seqs, batch_node_masks) in tqdm(data_loader):  # Testing on the same dataset
                batch_seqs = batch_seqs.to(device)
                batch_node_masks = batch_node_masks.to(device)
                for i in range(2,batch_seqs.shape[1], 2):
                    inputs = batch_seqs[:,:i]
                    targets = batch_seqs[:,i]
                    masks = batch_node_masks[:,i]
                    # Forward pass
                    outputs = model(inputs)
#                     print(outputs.shape)
                    outputs = outputs[:, i-1, :]  # Only take the first token prediction
                    predicted = torch.argmax(outputs, dim=1)

                    correct += ((targets.view(-1)==predicted) & masks).sum().item()
                    total += masks.sum().item()
                
        losses_for_n.append(loss)
        accuracies_for_n.append(correct/total*100)
        capacities_for_n.append(correct)
        print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {correct/total*100:.4f}% | Capacity: {correct:.0f}/{total:.0f}")

        
    return losses_for_n, accuracies_for_n, capacities_for_n

# Example function to save results
def save_results(results, filename):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")



# In[42]:


##############################################################################
# 4) Launching
##############################################################################


# Load TSV data
df_original, df_node_edge, df_word_by_word, node_edge_vocab, word_vocab = read_and_tokenize(DATA)

# Print a sample
print("Original DataFrame:")
print(df_original.head())
print("\nNode/Edge Tokenized DataFrame:")
print(df_node_edge.head())
print("\nWord-by-Word Tokenized DataFrame:")
print(df_word_by_word.head())
print("\nNode/Edge Vocabulary len:")
print(len(node_edge_vocab))
print("\nWord Vocabulary len:")
print(len(word_vocab))
    
# ==========================
# Model hyperparameters
# ==========================

data, vocab = df_node_edge, node_edge_vocab

vocab_size = len(vocab)+1
pad_token = 0

batch_size=128
lr=1e-3
epochs=400

d_model=64
n_heads=4
max_seq_len=data.shape[1]

num_iterations = 3

# Experiment parameters
n_values = [20000, 50000, 100000]
n_layers_values = [1, 2, 4]
activation_functions=["RReLU", "softmax"]
emb_values = ['Pos', ]

# Calculate the number of parameters to keep the total number constant
base_num_layers = 1
base_d_model = d_model
base_num_params = base_d_model * base_num_layers

# Prepare the list of all possible configurations using Cartesian product
configurations = list(
    itertools.product(n_values, activation_functions, n_layers_values, emb_values)
)


# Save configurations to a file
config_filename = "../../../data/configs/experiment_seq_configs.pkl"
os.makedirs(os.path.dirname(config_filename), exist_ok=True)
with open(config_filename, "wb") as f:
    pickle.dump(configurations, f)
    print(f"Configurations saved to {config_filename}")
    
print(len(configurations))


# Load configurations and determine starting point
parser = argparse.ArgumentParser(description='Experiment Configuration')
parser.add_argument('--start_index', type=int, default=0, help='Start index for configurations')
parser.add_argument('--end_index', type=int, default=len(configurations), help='End index for configurations')
parser.add_argument('--cuda_index', type=int, default=-1, help='Cuda index (negative value means without index).')
args = parser.parse_args()

start_index, end_index = args.start_index, args.end_index
cuda_index = args.cuda_index


accuracies = defaultdict(list)
capacities = defaultdict(list)
losses = defaultdict(list)
final_models = {}

# Iterate through configurations starting from the specified index
for config_index in range(start_index, min(end_index, len(configurations))):
    n, activation_fn_name, n_layers, emb = configurations[config_index]
    adjusted_d_model = int(base_num_params / n_layers)
    activation_fn = get_activation_function(activation_fn_name)

    for iteration in range(num_iterations):
        print(
            f"Training with n={n}, activation={activation_fn_name}, layers={n_layers}, iteration={iteration + 1}"
        )
        iteration_seed = SEED + iteration
        
        # Move the model to GPU if available
        if cuda_index>=0:
            device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        # Let's create sequences from df_node_edge, etc.
        sequences = data.sample(n=n, random_state=iteration_seed).to_numpy()
        node_masks = get_node_masks(sequences)
        
        # Prepare dataset and data loader
        dataset = NodeEdgeSequenceDataset(sequences, node_masks, pad_token=pad_token)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token=pad_token)
        )
        if emb=='Pos':
            # Build model
            model = GPTLikeModelPos(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                max_seq_len=max_seq_len,   # can exceed real max length
                pad_token=pad_token,
                activation_fn=activation_fn,
                seed=iteration_seed
            ).to(device)
        else:
            raise ValueError
        # Train
        losses_for_n, accuracies_for_n, capacities_for_n = train_model(model, 
                                                                       data_loader, 
                                                                       epochs=epochs, 
                                                                       lr=lr, 
                                                                       pad_token=pad_token, 
                                                                       device=device, 
                                                                       max_to_memorize = node_masks.sum()-node_masks.shape[0])
        # Save the final model for this iteration
        final_models[(n, activation_fn_name, n_layers, iteration)] = model
        # Save all accuracies for the current n value
        accuracies[(n, activation_fn_name, n_layers)].append(accuracies_for_n)
        capacities[(n, activation_fn_name, n_layers)].append(capacities_for_n)
        losses[(n, activation_fn_name, n_layers)].append(losses_for_n)
        
        # Intermediate save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        indexes = f"{start_index}_{end_index}_{iteration}"
        # Pickle the results and final models dictionaries
        save_results(
            accuracies,
            ACC_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
        )
        save_results(
            capacities,
            CAP_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
        )
        save_results(
            losses,
            LOSSES_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
        )

    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    indexes = f"{start_index}_{end_index}"
    # Pickle the results and final models dictionaries
    save_results(
        accuracies,
        ACC_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )
    save_results(
        capacities,
        CAP_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )
    save_results(
        losses,
        LOSSES_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )

    save_results(
        final_models,
        MODELS_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )

print('Finished! (c) TohaRhymes et al. 2025')
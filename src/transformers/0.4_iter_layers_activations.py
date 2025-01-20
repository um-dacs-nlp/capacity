#!/usr/bin/env python
# coding: utf-8

# # Imports, constants

# In[1]:
# imports
import pickle
import random
import argparse
from copy import deepcopy
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# constants
DATA = "../../../data/triplets.tsv"
SEED = 566
# Results and models file paths
RESULTS_FILE = (
    "../../../data/out_metrics/results_{timestamp}_lay_act_{config_index}.pkl"
)
LOSSES_FILE = (
    "../../../data/out_metrics/losses_{timestamp}_lay_act_{config_index}.pkl"
)
MODELS_FILE = "../../../data/out_models/models_{timestamp}_lay_act_{config_index}.pkl"


# fix random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# # All Functions

# In[2]:


# ===============================
# For data
# ===============================


def tokenize_columns(df_in, columns):
    df = deepcopy(df_in)
    # Create an empty vocabulary
    vocab = {}
    token_counter = (
        1  # Start token IDs from 1 (you can reserve 0 for padding if needed)
    )

    # Function to add unique column values to the vocab
    def add_to_vocab(value):
        nonlocal token_counter
        if value not in vocab:
            vocab[value] = token_counter
            token_counter += 1

    # Add all unique values from the specified columns to the vocabulary
    for column in columns:
        df[column].apply(add_to_vocab)

    # Function to tokenize a column value based on the vocab
    def tokenize(value):
        return [
            vocab[value]
        ]  # Return token ID as a list to keep compatibility with batch processing

    # Tokenize the specified columns
    for column in columns:
        df[f'tokenized_{column.lower().replace(" ", "_")}'] = df[column].apply(tokenize)

    # Combine tokenized concept and property into a single input sequence
    df["input_sequence"] = df.apply(
        lambda row: row["tokenized_concept"] + row["tokenized_property"], axis=1
    )

    return df, vocab


class TripletDataset(Dataset):
    def __init__(self, df):
        self.inputs = df["input_sequence"].tolist()
        self.targets = df["tokenized_related_concept"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sequence = torch.tensor(self.inputs[idx], dtype=torch.long)
        target_sequence = torch.tensor(self.targets[idx], dtype=torch.long)
        return input_sequence, target_sequence


# ===============================
# Model
# ===============================


# Define the custom activation functions
def get_activation_function(name):
    if name == "default":
        return nn.ReLU()
    elif name == "GELU":
        return nn.GELU()
    elif name == "RAF":
        return nn.RReLU()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown activation function: {name}")


# Modified model class to include dynamic activation function and adaptable hidden size
class GPTLikeModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        num_layers,
        max_seq_len,
        activation_fn,
        seed=42,
    ):
        super(GPTLikeModel, self).__init__()
        # Fix random seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=n_heads, activation=activation_fn
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Embedding and positional encoding
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]

        # Pass through each transformer decoder layer
        for layer in self.transformer_layers:
            x = layer(x, x)  # Decoder takes input twice in GPT-like models

        # Output layer
        logits = self.fc_out(x)
        return logits


# other logic


# Define a constant for the filename format
# Example function to save results
def save_results(results, filename):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")


# ## Read and tokenize dataset
# 

# In[3]:


# Load TSV data
data = pd.read_csv(DATA, sep="\t")
df = data.sample(n=200000, random_state=SEED)
columns_to_tokenize = ["Concept", "Property", "Related Concept"]
df_tokenized, vocab = tokenize_columns(df, columns_to_tokenize)


# ## Train and save

# In[4]:


# Experiment parameters
n_values = [50000, 70000, 100000]
activation_functions = ["default", "GELU", "RAF", "softmax"]
n_layers_values = [1, 2, 4]

# Model hyperparameters
vocab_size = len(vocab) + 1  # Include 1 for padding (if needed)
d_model = 128  # Embedding size
n_heads = 4  # Number of attention heads
max_seq_len = 2  # Maximum sequence length (concept + property)
batch_size = 128
lr = 0.001
epochs = 1000

# Calculate the number of parameters to keep the total number constant
base_num_layers = 1
base_d_model = d_model
base_num_params = base_d_model * base_num_layers

# Prepare the list of all possible configurations using Cartesian product
configurations = list(
    itertools.product(n_values, activation_functions, n_layers_values)
)

# Save configurations to a file
config_filename = "../../../data/configs/experiment_configs.pkl"
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


results = defaultdict(list)
losses = defaultdict(list)
final_models = {}
num_iterations = 10

# Iterate through configurations starting from the specified index
for config_index in range(start_index, min(end_index, len(configurations))):
    n, activation_fn_name, n_layers = configurations[config_index]
    adjusted_d_model = int(base_num_params / n_layers)
    activation_fn = get_activation_function(activation_fn_name)

    for iteration in range(num_iterations):
        results_for_n = []  # Create a separate list for each n value
        losses_for_n = []
        print(
            f"Training with n={n}, activation={activation_fn_name}, layers={n_layers}, iteration={iteration + 1}"
        )
        iteration_seed = SEED + iteration

        # Prepare dataset and data loader
        dataset = TripletDataset(df_tokenized.sample(n=n, random_state=iteration_seed))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the model with adjusted d_model and activation function
        model = GPTLikeModel(
            vocab_size,
            adjusted_d_model,
            n_heads,
            n_layers,
            max_seq_len,
            activation_fn,
            seed=iteration_seed,
        )

        # Move the model to GPU if available
        if cuda_index>=0:
            device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Device using: {device}")
        # Define the optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # RUN LAUNCHING
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            model.train()  # Set model to training mode

            # Training on the same data
            for batch in tqdm(train_loader):
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                optimizer.zero_grad()  # can be placed anywhere before loss.backward
                outputs = model(inputs)

                # We only care about the first token in the output sequence
                outputs = outputs[:, 0, :]  # Shape becomes: (batch_size, vocab_size)

                targets = targets.view(-1)  # Flatten the targets

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} (iteration {iteration + 1}/{num_iterations}), Training Loss: {total_loss / len(train_loader)}")
            losses_for_n.append(total_loss / len(train_loader))
            if epoch % 2 == 0:
                continue  # skip 0,2,4... (e.g. if n_iters=100, so we plot 1,3,..99 (99th is the last))
            # Testing on the same data (memorization check)
            model.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(train_loader):  # Testing on the same dataset
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    outputs = outputs[:, 0, :]  # Only take the first token prediction
                    predicted = torch.argmax(outputs, dim=1)

                    total += targets.size(0)
                    correct += (predicted == targets.view(-1)).sum().item()
            #             print(total, correct)

            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}, Memorization Accuracy: {accuracy:.5f}%")
            results_for_n.append(accuracy)
        # Save the final model for this iteration
        final_models[(n, activation_fn_name, n_layers, iteration)] = model
        # Save all accuracies for the current n value
        results[(n, activation_fn_name, n_layers)].append(results_for_n)
        losses[(n, activation_fn_name, n_layers)].append(losses_for_n)

        # Intermediate save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        indexes = f"{start_index}_{end_index}_{iteration}"
        # Pickle the results and final models dictionaries
        save_results(
            results,
            RESULTS_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
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
        results,
        RESULTS_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )
    save_results(
        losses,
        LOSSES_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )
    save_results(
        final_models,
        MODELS_FILE.format(**{"timestamp": timestamp, "config_index": indexes}),
    )


# In[ ]:

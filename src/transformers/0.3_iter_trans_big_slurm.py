#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import random

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

DATA = "/home/achangalidi/project/data/triplets.tsv"
SEED = 566


# In[2]:


# Fix random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ## Read and tokenize dataset
# 

# In[3]:


def tokenize_columns(df, columns):
    # Create an empty vocabulary
    vocab = {}
    token_counter = 1  # Start token IDs from 1 (you can reserve 0 for padding if needed)

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
        return [vocab[value]]  # Return token ID as a list to keep compatibility with batch processing

    # Tokenize the specified columns
    for column in columns:
        df[f'tokenized_{column.lower().replace(" ", "_")}'] = df[column].apply(tokenize)

    # Combine tokenized concept and property into a single input sequence
    df['input_sequence'] = df.apply(lambda row: row['tokenized_concept'] + row['tokenized_property'], axis=1)

    return df, vocab


class TripletDataset(Dataset):
    def __init__(self, df):
        self.inputs = df['input_sequence'].tolist()
        self.targets = df['tokenized_related_concept'].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sequence = torch.tensor(self.inputs[idx], dtype=torch.long)
        target_sequence = torch.tensor(self.targets[idx], dtype=torch.long)
        return input_sequence, target_sequence


# In[4]:


# Load TSV data
data = pd.read_csv(DATA, sep="\t")
df = data.sample(n=100000, random_state=SEED)
# df = data[:100000]
df.head()


# In[5]:


columns_to_tokenize = ["Concept", "Property", "Related Concept"]

df_tokenized, vocab = tokenize_columns(df, columns_to_tokenize)
print("First 10 rows of Vocabulary:")
pprint(dict(list(vocab.items())[:10]))


# ## The Model

# In[6]:


class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, max_seq_len, seed=42):
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
                nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads)
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


# ## Train and save

# In[7]:


# model hyperparameters
vocab_size = len(vocab) + 1  # Include 1 for padding (if needed)
d_model = 128  # Embedding size
n_heads = 4  # Number of attention heads
num_layers = 1  # Number of transformer layers
max_seq_len = 2  # Maximum sequence length (concept + property)
batch_size = 64
lr = 0.001
# data params
n_values = [50000, 60000, 70000, 80000, 90000, 100000]
# n_values = [10, 100, 1000]
# epochs
epochs = 500

results = defaultdict(list)
final_models = {}
# Set the number of iterations for each n value
num_iterations = 10

for n in n_values:
    # Repeat the training process for each n value num_iterations times
    for iteration in range(num_iterations):
        results_for_n = []  # Create a separate list for each n value
        print(f"Training with n={n}, iteration={iteration + 1}")
        # Set a different seed for each iteration to get different accuracies
        iteration_seed = SEED + iteration
        
        # PREPARE EVTH
        # Create dataset and data loader
        dataset = TripletDataset(df.sample(n=n, random_state=iteration_seed))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        # Initialize the model
        model = GPTLikeModel(
            vocab_size, d_model, n_heads, num_layers, max_seq_len, seed=iteration_seed
        )
        
        torch.cuda.empty_cache()
        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    
        # Define the optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
    
        # RUN LAUNCHING
    
        # Training and testing loop (memorization task)
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
    
            print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")
            if epoch % 2 != 0:
                continue
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
        final_models[(n, iteration)] = model    
        # Save all accuracies for the current n value
        results[n].append(results_for_n)
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Pickle the results and final models dictionaries
    with open(f"../../../data/out_metrics/results_{timestamp}_50_100k_trans.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"../../../data/out_models/models_{timestamp}_50_100k_trans.pkl", "wb") as f:
        pickle.dump(final_models, f)
    print(f'Finished for {n}, timestamp: {timestamp}')


# In[ ]:


# Add timestamp to filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Pickle the results and final models dictionaries
with open(f"../../../data/out_metrics/results_{timestamp}_init_trans.pkl", "wb") as f:
    pickle.dump(results, f)
with open(f"../../../data/out_models/models_{timestamp}_init_trans.pkl", "wb") as f:
    pickle.dump(final_models, f)




#!/usr/bin/env python
# coding: utf-8

# In[114]:


from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
# from owlready2.pymedtermino2.icd10_french import *

from tqdm import tqdm
import pandas as pd

from collections import defaultdict
import random
import networkx as nx
from tqdm import tqdm

import re
import random


# In[60]:


DB_NAME = "../../../data/pym.sqlite3"
ZIP_NAME = "../../../data/umls-2024AA-full.zip"
SAVE_TO = "../../../data/created_data/seqs{}.tsv"


# previously deleted:     'term_type', 'terminology', 
# added: 'ctv3id', originals'
BANNED_PROPS = [
    'icd-o-3_code',  #ok almost empty
    'ctv3id', #ok unique for all, which is bad, full memorization
    'subset_member',  #ok unique for all, which is bad, full memorization
    'label', #ok almost the same
    'synonyms',  #ok almost the same to the parent node
    'unifieds', #ok almost the same to the parent node
    'originals'  #ok almost the same to the parent node
               ]
RANDOM_SEED=30_239_566

BANNED_GROUP = ['case_significance_id',
 'groups',
 'type_id',
 'ctv3id',
 'effective_time',
 'unifieds',
 'active',
 'synonyms',
 'terminology',
 'subset_member',
 'definition_status_id',
 'term_type']

# mapped_to: example -- 'ICD10["I51.8"] # Other ill-defined heart diseases\n'


# # Read KG  and build graph

# In[41]:


default_world.set_backend(filename = DB_NAME)
PYM = get_ontology("http://PYM/").load()


# In[61]:


# Graph initialization from SNOMED data
def build_graph(ontology):
    G = nx.DiGraph()
    for concept in tqdm(ontology.classes()):
        for prop in concept.get_class_properties():
            if prop.name in BANNED_PROPS or prop.name in BANNED_GROUP:
                continue
            related_concepts = getattr(concept, prop.name, [])
            if not isinstance(related_concepts, (list, set)):
                related_concepts = [related_concepts, ]
            for rc in related_concepts:
                G.add_edge(concept, rc, relationship=prop.name)
                # Add edge with attribute for reversed relationship
                G.add_edge(rc, concept, relationship=f"reversed_{prop.name}")
    return G


# In[63]:


ontology = PYM
G = build_graph(ontology)


# # Algorithms fopr traversal

# In[119]:


# Function to beautify nodes
def beautify_node(node_str):
    patterns = [r'SNOMEDCT_US\[".*?"\] #\s*', r'ICD10\[".*?"\] #\s*']
    cleaned_node = node_str
    for pattern in patterns:
        cleaned_node = re.sub(pattern, '', cleaned_node)
    return cleaned_node.strip()


def beautify_graph_nodes(graph):
    """Pre-beautify all nodes in the graph."""
    mapping = {node: beautify_node(str(node)) for node in graph.nodes}
    return nx.relabel_nodes(graph, mapping)

# Create a subgraph around a random starting node
def create_subgraph(G, start_node, hops=3):
    bfs_nodes = nx.single_source_shortest_path_length(G, start_node, cutoff=hops)
    subgraph_nodes = list(bfs_nodes.keys())
    return G.subgraph(subgraph_nodes)

# Generate a sequence from the subgraph with global ambiguity avoidance
def generate_sequence_from_subgraph(subgraph, edge_count_range=(3, 5)):
    global visited_pairs  # Access global visited_pairs
    local_triplets = set()  # Track node-edge-node triplets for the current sequence
    sequence = []
    
    # Randomly select the target number of edges within the range
    target_edge_count = random.randint(edge_count_range[0], edge_count_range[1])
    current_edge_count = 0  # Counter for edges in the sequence
    
    # Random starting node
    start_node = random.choice(list(subgraph.nodes))
    current_node = start_node

    while current_edge_count < target_edge_count:
        neighbors = list(subgraph.neighbors(current_node))
        
        # Filter neighbors to exclude already visited (node, edge) pairs
        valid_neighbors = []
        for next_node in neighbors:
            edge_data = subgraph.get_edge_data(current_node, next_node)
            edge_name = edge_data.get('relationship', 'No relationship') if edge_data else 'No relationship'
            
            # Create the triplet for the current context
            triplet = (current_node, edge_name, next_node)
            
            # Check for both global and local ambiguity
            if (current_node, edge_name) not in visited_pairs and triplet not in local_triplets:
                valid_neighbors.append((next_node, edge_name, triplet))

        if not valid_neighbors:  # If no valid neighbors, terminate or restart
            break  # Terminate sequence generation
        
        # Randomly select a valid neighbor
        next_node, edge_name, triplet = random.choice(valid_neighbors)
        
        # Add the triplet to the local visited set
        local_triplets.add(triplet)
        
        # Add the (node, edge) pair to the global visited set
        visited_pairs.add((current_node, edge_name))
        
        # Beautify and add nodes/edges to the sequence
        sequence.append(str(current_node))
        sequence.append(edge_name)
        current_node = next_node
        current_edge_count += 1

    # Add the final node to the sequence
    sequence.append(str(current_node))
    
    return sequence


# Function to format and save a sequence
def save_sequence(sequence, file):
    file.write("\t".join(sequence) + "\n")


# In[117]:

ROWS=100000



# Global set to track visited node+edge pairs across all sequences
visited_pairs = set()

graph = beautify_graph_nodes(G.copy())

# Open the file once, clear its content initially, and write all sequences
with open(SAVE_TO.format(ROWS), "w") as f:  # Open in write mode to clear and write
    for i in tqdm(range(ROWS)):
        start_node = random.choice(list(graph.nodes))  # Random starting node
        subgraph = create_subgraph(graph, start_node, hops=5)  # Create a subgraph around the node
        sequence = generate_sequence_from_subgraph(subgraph, edge_count_range=(3, 5))  # Generate sequence

        # Save the sequence to the file
        save_sequence(sequence, f)

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
SAVE_TO = "../../../data/created_data/seqs_dense{}.tsv"
MID_SAVE_TO = "../../../data/created_data/mid.tsv"


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
    G = nx.Graph()
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



# Function to format and save a sequence
def save_sequence(sequence, file):
    file.write("\t".join(sequence) + "\n")


# Create a subgraph around a random starting node
def create_subgraph(G, start_node, hops=3):
    bfs_nodes = nx.single_source_shortest_path_length(G, start_node, cutoff=hops)
    subgraph_nodes = list(bfs_nodes.keys())
    return G.subgraph(subgraph_nodes)

def beautify_graph_nodes(graph):
    """Pre-beautify all nodes in the graph."""
    mapping = {node: beautify_node(str(node)) for node in graph.nodes}
    return nx.relabel_nodes(graph, mapping)


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



def has_unvisited_edges(node, graph, visited_pairs):
    """Check if a node has any unvisited edges."""
    for neighbor in graph.neighbors(node):
        edge_data = graph.get_edge_data(node, neighbor)
        edge_name = edge_data.get('relationship', 'No relationship') if edge_data else 'No relationship'
        if (node, edge_name) not in visited_pairs:
            return True  # Found an unvisited edge
    return False  # All edges for this node are visited

# Main function to create condensed sequences
def create_condensed_sequences(graph, save_to, num_sequences, edge_count_range=(3, 5), min_component_size=5):
    global visited_pairs
    visited_pairs = set()
    cur_component_visited_nodes = set()  # Track all visited nodes within the current connectivity component
    successful_sequences = 0  # Counter for successful sequences
    # Precompute all original neighbors
    original_neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
    components = [comp for comp in nx.connected_components(graph) if len(comp) >= min_component_size]
    with open(save_to, "w") as f, tqdm(total=num_sequences) as pbar:
        while successful_sequences < num_sequences:
            # Generate nearby nodes, considering visited nodes
            if cur_component_visited_nodes:
                start_nodes = set()
                for node in cur_component_visited_nodes:
                    if has_unvisited_edges(node, graph, visited_pairs):
                        start_nodes.add(node)  # Add the current node if it has any unvisited edges

                    # Check neighbors for their edges
                    for neighbor in graph.neighbors(node):
                        if has_unvisited_edges(neighbor, graph, visited_pairs):
                            start_nodes.add(neighbor)  # Add neighbor if it has any unvisited edges
            else:
                # Start a new component
                cur_component_visited_nodes = set()  # Reset for a new component
                component = random.choice(components)
                start_nodes = list(component)
                print('Go to the new connectivity component: previously selected `cur_component_visited_nodes` turned out to be empty.')
                
            if not start_nodes:
                # Fallback to a random component if no nearby nodes
                cur_component_visited_nodes = set()  # Reset for a new component
                component = random.choice(components)
                start_nodes = list(component)
                print('Go to the new connectivity component: we took the max from previously selected `cur_component_visited_nodes`.')
            
             # Randomly select a starting node
            start_node = random.choice(list(start_nodes))
            subgraph = create_subgraph(graph, start_node, hops=edge_count_range[1])  # Create a subgraph around the node
            # Generate sequence
            sequence = generate_sequence_from_subgraph(subgraph, edge_count_range=edge_count_range)
            
            # Skip too-short sequences
            if len(sequence) <= edge_count_range[0]:
                continue

            # Save the sequence
            save_sequence(sequence, f)

            # Update progress bar and counters
            successful_sequences += 1
            pbar.update(1)

            # Update last sequence nodes
            cur_component_visited_nodes = cur_component_visited_nodes | set(sequence[::2])  # Add all visited nodes


# In[117]:

ROWS=10000


# Example Usage
visited_pairs = set()
graph = beautify_graph_nodes(G.copy())  # Work with a copy of the graph to avoid modifying the original   
create_condensed_sequences(graph, SAVE_TO.format(ROWS), num_sequences=ROWS, edge_count_range=(3, 5), min_component_size=5)

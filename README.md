# Capacity: Measuring Transformer Memorization on Real-World Data

This repository contains the code, experiments, and visualizations for the paper:

Changalidis, A.; Härmä, A. Capacity Matters: a Proof-of-Concept for Transformer Memorization on Real-World Data, 2025, [arXiv:cs.CL/2506.14704](https://arxiv.org/abs/2506.14704) 

Accepted to **ACL 2025 Workshop L2M2: The First Workshop on Large Language Model Memorization**, Vienna, August 1st, 2025.

---

## Overview

This project evaluates how decoder-only transformer architectures memorize real-world structured data. We use subsets of the **SNOMED CT** medical knowledge graph to construct two types of datasets:

* **Triplets**: (Concept, Property, Related Concept)
* **Sequences**: Simulated graph walks encoding multi-hop relationships

The experiments assess the influence of:

* Dataset size
* Number of layers
* Activation functions
* Embedding size (total number of parameters)

Using **Maximum Attainable Capacity (MAC)** as a key metric, the study identifies optimal architecture–data trade-offs for memory-limited scenarios (e.g., edge devices).

---

## Repository Structure

```bash
.
├── img/                  # Figures and plots used in the paper
├── src/                 # Source code and experiments
│   ├── data_management/     # Dataset extraction and preprocessing (requires Python 3.7)
│   │   ├── 0_snomed_starting.ipynb     # Initialize and load SNOMED ontology
│   │   ├── 1_snomed_get_dataset.ipynb  # Extract triplet datasets
│   │   └── 2_unlinked_construction.py  # Generate graph-walk sequences
│   └── transformers/         # Experiments and architecture evaluation
│       ├── 1_iter_trans_big.ipynb               # Initial experiments (triplets)
│       ├── 2_layers_activations*.ipynb/.py      # Layer & activation comparisons
│       ├── 3_param_size*.ipynb/.py              # Parameter size impact
│       └── 4_seqs*.ipynb/.py                    # Experiments on sequence-based datasets
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Figures

All plots summarizing experimental results are located in `img/`. The filenames follow the format:

* `1_datasize_*` — Results of Experiment 1 (dataset size)
* `2_layact_*` — Experiment 2 (layer + activation variations)
* `3_paramsize_*` — Experiment 3 (embedding/parameter size)
* `4_seqs_*` — Experiment 4 (sequences)

Each comes in both `.pdf` and `.png` formats.

---

## Setup Instructions

**Dependencies:**

```bash
pip install -r requirements.txt
```

> For the `data_management` part, **Python 3.7** is required due to dependency on `owlready2`.

You will need access to the SNOMED CT ontology file to run the data processing notebooks.

---

## Reproducing Experiments

Each experiment is defined as follows:

1. **Triplets – Dataset Size Impact**
   File: `1_iter_trans_big.ipynb`
   Tests different dataset sizes with a fixed 1-layer transformer.

2. **Triplets – Layer & Activation Influence**
   Files: `2_layers_activations_*`
   Tests ReLU, GELU, RReLU, Softmax with 1/2/4 layers.

3. **Triplets – Parameter Size**
   Files: `3_param_size_*`
   Tests different embedding sizes for fixed parameter budgets.

4. **Sequences – Graph Path Memorization**
   Files: `4_seqs_*`
   Tests ability to memorize sequences with 4–6 nodes.

---

## Citation

If you use this code or build upon this work, please cite:


```
@misc{changalidis2025capacitymattersproofofconcepttransformer,
      title={Capacity Matters: a Proof-of-Concept for Transformer Memorization on Real-World Data}, 
      author={Anton Changalidis and Aki Härmä},
      year={2025},
      eprint={2506.14704},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.14704}, 
}
```




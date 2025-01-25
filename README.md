# Capacity

The directories and their content:

* `docs` directory -- some documentation and partial reports
* `src` -- source codes of the research:
    * `data_management` -- directory where we manipulated with datasets (!! `python 3.7` required):
        * `0_snomed_starting.ipynb` -- initial launching of the db, saving to data dir
        * `1_snomed_get_dataset` -- creating triplets dataframe and save.
        * `2_unlinked_construction.py` -- creating sequences dataframe and save.
    * `transformers` -- experiments with transformers models:
        * `1_iter_trans_big.ipynb` -- first setup of experiment with triplets;
        * `2_layers_activations*` -- second setup of experiment with triplets (layers, activation function changing) + drawing of accuracies, capacities and losses;
        * `3_param_size*` -- third setup of experiment with triplets (amount of parameters) + drawing of accuracies, capacities and losses;
        * `4_seqs*` -- forth setup of experiment with sequences.
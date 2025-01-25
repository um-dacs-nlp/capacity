# Capacity

The directories and their content:

* `docs` directory -- some documentation and partial reports
* `src` -- source codes of the research:
    * `data_management` -- directory where we manipulated with datasets (!! `python 3.7` required):
        * `0_snomed_starting.ipynb` -- initial launching of the db, saving to data dir
        * `1_snomed_get_dataset` -- creating triplets dataframe and save.
        * `2_unlinked_construction.py` -- creating sequences dataframe and save.
    * `transformers` -- experiments with transformers models:
        * `0.1_base_trans.ipynb`, `0.1_playground.ipynb` -- basic notebooks for tranmsformers
        * `0.2_iter_trans.ipynb`, `0.3_iter_trans_big.ipynb` -- main training and drawing code for now (nov 13) with some results
        * `0.2_iter_trans_slurm.py`, `0.3_iter_trans_big_slurm.py` -- corresponding files to run on slurm
        * `1.0_extract_models.ipynb` -- extracting models and data from pickle, saving to files
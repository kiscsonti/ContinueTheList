# ContinueTheList

This repository contains the code of three solutions to the Continue The List or Extend The List problem.

## Setup
- Setup DBPedia 2016-04 on your local machine (e.g.: virtuoso container) or use the public endpoint (will be slow)
- Run `pip install -r requirements`
- Download the dataset LC-QuAD
- Run preprocess steps in dataset_preprocess folder (LC_QUAD_preprocess.ipynb)
- If you want to run the SentenceBert embedding algo you shall get the embeddings for all of the nodes in the DBPedia --> dataset_preprocess/BERT_getvectors2nodes.py
- In the algorithm folder you can see the 3 methods implemented/tested (+ others that we didn't report on).
- Inside the main folder you can find all the usage examples for the algorithms
- First you shall run the algorithm, then you shall run the evaluation on the output

## Notes
- In this repo you can find additional experiment results as well and statistics that has been omitted from the paper due to irrelevancy.
- This is not a 100% organized repo.

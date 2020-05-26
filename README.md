# Probing Language Models

Repository for the NLP2 assignment on probing language models. By Anna Langedijk and Albert Harkema.

# Code structure

If you want to look at nice plots and many p-values, we suggest you look at `Results.ipynb`.

All the wrapper code for building representations and training is in `Training.ipynb`.
The only reason you may want to look at this is if you want to see the training loops and evaluation.
All relevant models should be in `StructuralProbe.py` and `POSProbe.py`. 
All relevant tasks are in `utils.py`, `tree_utils.py` and `controltasks.py`. 

The notebooks are largely based on existing pickled file. If for any reason you want these pickled files, please contact `annaproxy@protonmail.com`. 

# Dependencies

pytorch, ete3, conllu, transformers, nltk. 



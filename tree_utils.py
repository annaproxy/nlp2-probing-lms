import torch
import math 
import os
import numpy as np
import pickle
from ete3 import Tree as EteTree
from scipy.sparse.csgraph import minimum_spanning_tree
from utils import fetch_sen_reps
from lstm.model import RNNModel 

# In case you want to transform your conllu tree to an nltk.Tree, for better visualisation

def rec_tokentree_to_nltk(tokentree):
    token = tokentree.token["form"]
    tree_str = f"({token} {' '.join(rec_tokentree_to_nltk(t) for t in tokentree.children)})"

    return tree_str


def tokentree_to_nltk(tokentree):
    from nltk import Tree as NLTKTree
    tree_str = rec_tokentree_to_nltk(tokentree)
    return NLTKTree.fromstring(tree_str)


class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)
        
    def __str__(self):
        return self.get_ascii(show_internal=True)
    
    def __repr__(self):
        return str(self)


def rec_tokentree_to_ete(tokentree):
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx
    
def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)
    return FancyTree(f"{newick_str};")

def parse_corpus(filename):
    from conllu import parse_incr

    data_file = open(filename, encoding="utf-8")
    ud_parses = list(parse_incr(data_file))
    
    return ud_parses

def create_gold_distances(corpus):
    all_distances = []

    for item in (corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)

        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))
        
        # Traverse tree in two directions and get all distances
        dists = []
        dists = [node.get_distance(node2) for node in ete_tree.traverse() for node2 in ete_tree.traverse()]            

        # Turn it into a tensor, view, append
        dists = torch.tensor(dists)
        distances = dists.view(sen_len, sen_len)
        all_distances.append(distances)

    return all_distances



def create_mst(distances):
    distances = torch.triu(distances).cpu().detach().numpy()
    
    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.
    
    return mst

def edges(mst):
    edges = set()

    # Your code for retrieving the edges from the MST matrix
    locations = np.argwhere(mst == 1)

    for elem in locations:
        edges.add((elem[0], elem[1]))
        
    return edges

def calc_uuas(pred_distances, gold_distances):
    uuas = None
    
    # Get both MSTs
    pred_mst = create_mst(pred_distances)
    gold_mst = create_mst(gold_distances)
    
    # Get their edges
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    

    # Calculate uuas
    uuas = np.sum([pred_edge in gold_edges for pred_edge in pred_edges]) / len(gold_edges)
    #print(uuas)
#     if not math.isnan(uuas):
#         print("PRED EDGES, GOLD EDGES")
#         print(pred_edges)
#         print(gold_edges)
#         print("LEN GOLD")
#         print(len(gold_edges))
    
    return uuas


'''
Similar to the `create_data` method of the previous notebook, I recommend you to use a method 
that initialises all the data of a corpus. Note that for your embeddings you can use the 
`fetch_sen_reps` method again. However, for the POS probe you concatenated all these representations into 
1 big tensor of shape (num_tokens_in_corpus, model_dim). 

The StructuralProbe expects its input to contain all the representations of 1 sentence, so I recommend you
to update your `fetch_sen_reps` method in a way that it is easy to retrieve all the representations that 
correspond to a single sentence.
''' 

def init_corpus(path, lm, w2i, concat=False, cutoff=None):
    """ Initialises the data of a corpus.
    
    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating 
        memory usage.
    """
    corpus = parse_corpus(path)[:cutoff]
    print("Fetching for", len(corpus))
    embs = fetch_sen_reps(corpus, lm, w2i , concat=concat)    
    gold_distances = create_gold_distances(corpus)
    
    return gold_distances, embs

def create_or_load_structural_data(set_type:str, lm, w2i, cutoff=None):
    model_name = 'RNN' if type(lm) == RNNModel else 'transformer'

    data_file = os.path.join('data', 'en_ewt-ud-'+set_type+'.conllu')
    save_file = os.path.join('corpus', model_name + "_structural" + set_type + ".pickle")

    if os.path.exists(save_file):
        with open(save_file, "rb") as f: 
            return pickle.load(f)
    true_distances, reprs = init_corpus(data_file, lm, w2i, cutoff=cutoff)

    print("Data created,pickling")
    with open(save_file, "wb") as f: 
        pickle.dump({'x':reprs, 'y': true_distances}, f)

    return true_distances, reprs

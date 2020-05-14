import copy
from collections import defaultdict 
import numpy as np 
from tree_utils import tokentree_to_ete
from utils import parse_corpus
import torch
import os
import pickle
import conllu
#
def control_POS_train(data_x, data_y, flattened, dist):
    """
    Function to create control data for the POS tagging task based on a distribution
    """
    control_dict = defaultdict(lambda:None)
    
    # Retrieve 'POS tags' and distribution values
    keys = np.array(list(dist.keys()))
    values = np.array(list(dist.values()))
    
    # Normalize to get probabilities and sample keys
    probs = values / np.sum(values)
    
    # Make dict and get a full new y based on dict
    control_dict = {word.lower():np.random.choice(keys,replace=True,p=probs) for word in flattened if not word.lower() in control_dict}
    new_y = np.array([control_dict[word.lower()] for word in flattened])
    
    return new_y, control_dict

#
def control_POS(control_dict_train, flattened, dist):
    """
    Create test and dev control sets based on word mappings from train dict
    """
    # Retrieve 'POS tags' and distribution values
    keys = np.array(list(dist.keys()))
    values = np.array(list(dist.values()))
    
    # Normalize to get probabilities and sample keys
    probs = values / np.sum(values)
    
    # Initialize control dict and new y
    new_y = []
    control_dict = copy.deepcopy(control_dict_train)
    
    for word in flattened:
        word = word.lower()
        if word not in control_dict:
            control_dict[word] = np.random.choice(keys,replace=True,p=probs)

    new_y = np.array([control_dict[word.lower()] for word in flattened])
    
    return np.array(new_y)

def get_behaviour(behave_dict, token):
    if token in behave_dict:
        return behave_dict[token]
    return np.random.choice(["beginning", "ending"],p=[1/2,1/2])

def fake_gold_distances(corpus, behave_dict):
    all_distances = []
    
    for item in corpus:
        n = len(item)
        modified_heads = np.zeros(n)
        for word in item:
            i = word['id']
            
            behaviour = get_behaviour(behave_dict, word['form'])
            if behaviour == "beginning":
                modified_heads[i-1] = 1 
            elif behaviour == "ending":
                modified_heads[i-1] = n
            
        for i, z in enumerate(item):
            new_head = int(modified_heads[i])
            z['head'] = new_head
            if i == 0 :
                z['head'] = 0
            elif i == (n-1):
                z['head'] = 1
            
        tokentree = item.to_tree()
        test = tokentree_to_ete(tokentree)
        dists = [node.get_distance(node2) for node in test.traverse() for node2 in test.traverse()]

        # Turn it into a tensor, view, append
        dists = torch.tensor(dists)
        boy = int(np.sqrt(len(dists)))
        assert boy == n, "Horrible"
        dists = dists.view(boy,boy)
        all_distances.append(dists)
    return all_distances, behave_dict

def save_or_load_pos_controls(train_x =None, train_y=None, flattened_lists=None, dist=None):
    train_filename = os.path.join('control', 'pos' + 'train' + '.pickle')
    dev_filename = os.path.join('control', 'pos' + 'dev' + '.pickle')
    test_filename = os.path.join('control', 'pos' + 'test' + '.pickle')

    if os.path.exists(train_filename):
        with open(train_filename, 'rb') as f:
            with open(dev_filename, 'rb') as f2:
                with open(test_filename, 'rb') as f3:
                    return pickle.load(f), pickle.load(f2), pickle.load(f3)

    control_y_train, new_d = control_POS_train(train_x, train_y, flattened_lists[0], dist)
    control_y_dev = control_POS(new_d, flattened_lists[1], dist)
    control_y_test  = control_POS(new_d, flattened_lists[2] , dist)

    with open(train_filename, 'wb') as f: 
        pickle.dump(control_y_train,f)
    with open(dev_filename, 'wb') as f: 
        pickle.dump(control_y_dev,f)
    with open(test_filename, 'wb') as f: 
        pickle.dump(control_y_test,f)
    return control_y_train, control_y_dev, control_y_test

def save_or_load_struct_controls(cutoff=None):
    traincorpus = parse_corpus(os.path.join('data', 'en_ewt-ud-'+'train'+'.conllu'))[:cutoff ]
    devcorpus = parse_corpus(os.path.join('data', 'en_ewt-ud-'+'dev'+'.conllu'))[:cutoff]
    testcorpus = parse_corpus(os.path.join('data', 'en_ewt-ud-'+'test'+'.conllu'))[:cutoff]
    
    train_filename = os.path.join('control','struct' + 'train' + '.pickle')
    dev_filename = os.path.join('control','struct' + 'dev' + '.pickle')
    test_filename = os.path.join('control','struct' + 'test' + '.pickle')

    if os.path.exists(train_filename):
        with open(train_filename, 'rb') as f:
            with open(dev_filename, 'rb') as f2:
                with open(test_filename, 'rb') as f3:
                    return pickle.load(f), pickle.load(f2), pickle.load(f3)

    behave_dict = {}
    train_dists, behave_dict = fake_gold_distances(traincorpus, behave_dict)
    dev_dists, behave_dict = fake_gold_distances(devcorpus, behave_dict)
    test_dists, behave_dict = fake_gold_distances(testcorpus, behave_dict)

    with open(train_filename, 'wb') as f: 
        pickle.dump(train_dists,f)
    with open(dev_filename, 'wb') as f: 
        pickle.dump(dev_dists,f)
    with open(test_filename, 'wb') as f: 
        pickle.dump(test_dists,f)

    return train_dists, dev_dists, test_dists
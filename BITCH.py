import torch
import numpy as np
import time

from collections import defaultdict
from typing import List
from conllu import parse_incr, TokenList
from torch import Tensor
from transformers import GPT2Model, GPT2Tokenizer
CUTOFF = None
from lstm.model import RNNModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from utils import create_or_load_pos_data
from save_both import create_or_load_both_data
from controltasks import save_or_load_pos_controls 
from datasets import find_distribution, POSDataset
import torch.utils.data as data 
import time

"""
Change this piece of malevolent code that says CUTOFF = 100 or CUTOFF = 20
"""
CUTOFF = None
"""
"""
def get_transformer_reps(transformer, tokenizer, cutoff=CUTOFF, extra_transformer=None):
    """
    Ugly function that either builds representations for a transformer or retrieves pickled ones
    """
    
    train_x, train_y= create_or_load_both_data("train", 
                                                                   transformer, 
                                                                   tokenizer, 
                                                                   cutoff=CUTOFF,
                                                                   extra_transformer = extra_transformer)
    dev_x, dev_y = create_or_load_both_data("dev", 
                                                             transformer, 
                                                             tokenizer, 
                                                             cutoff=CUTOFF,
                                                             extra_transformer = extra_transformer)
    test_x, test_y= create_or_load_both_data("test", 
                                                                transformer, 
                                                                tokenizer, 
                                                                cutoff=CUTOFF,
                                                                extra_transformer = extra_transformer)

    # Flatten the wordlists so we have one big list of words for all set types
    #flatten_train = [word for sublist in words_train for word in sublist]
    #flatten_dev   = [word for sublist in words_dev for word in sublist]
    #flatten_test  = [word for sublist in words_test for word in sublist]
    
    # Generate a distribution over tags, useful for control task
    #dist = find_distribution(data.DataLoader(POSDataset(train_x, train_y), batch_size=1))
    #print(len(dist))
    #ypos_train_control, ypos_dev_control, ypos_test_control = save_or_load_pos_controls(
    #    train_x, train_y, [flatten_train, flatten_dev, flatten_test], dist)

    #
    return train_x, train_y, \
           dev_x, dev_y, \
           test_x, test_y


# Load
# Transformer XL
from transformers import TransfoXLTokenizer, TransfoXLModel
transfo_XL = TransfoXLModel.from_pretrained('transfo-xl-wt103')
print("I have loaded the transformer XL model")
transfo_XL_tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
print("I have loaded the transformer XL Tokenizer")


# Build TransformerXL
import warnings
warnings.filterwarnings(action='ignore')
train_x_transfo_XL, train_y_transfo_XL,  \
           dev_x_transfo_XL, dev_y_transfo_XL,  \
           test_x_transfo_XL, test_y_transfo_XL = get_transformer_reps(transfo_XL, transfo_XL_tokenizer, extra_transformer='TransformerXL')
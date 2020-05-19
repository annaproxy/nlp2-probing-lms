from lstm.model import RNNModel
import os 
import warnings
warnings.filterwarnings(action='ignore')
import pickle 
from typing import List 
from torch import Tensor
from tqdm import tqdm
from conllu import parse_incr, TokenList
from tree_utils import create_gold_distances
import torch 
def parse_corpus(filename: str) -> List[TokenList]:
    """Parses a conllu file into a List of TokenLists"""
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    return ud_parses
last =0
pos_w2i = dict()

def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer) -> List[Tensor]:
    """
    Returns a list of length len(ud_parses)
    """
        
    global pos_w2i, last
    model.eval()
    model.cuda()
    doing_lstm = type(model) == RNNModel
    print(f"Doing LSTM: {doing_lstm}")
    sentences_result = []
    global_words = []
    pos_result = []
    for sentence_nr, sentence in tqdm(enumerate(ud_parses)):
        sentence_words = []
         
        # First build string sentence repr with spaces and such
        for i, token in enumerate(sentence):
            postag = token['upostag']
            if postag in pos_w2i:
                posindex = pos_w2i[postag]
            else:
                posindex = last
                pos_w2i[postag] = last 
                last += 1
            pos_result.append(posindex)
            if token['misc'] is not None:
                # SpaceAfter = False
                next_word = token['form']
            else:
                # SpaceAfter = True
                next_word = token['form'] + ' '
            sentence_words.append(next_word)
        # Now build model representation!
        
        # Also add to global_words to retain word representations
        global_words.append(sentence_words)
        
        # In case of LSTM
        if doing_lstm:
            the_input = torch.tensor([tokenizer[z.strip()] for z in sentence_words]).unsqueeze(0)
            with torch.no_grad():
                final = model(the_input, model.init_hidden(1))
            final = final.squeeze(0)
            assert len(final) == len(sentence_words), "Something is wrong.."
            sentences_result.append(final)
                
        # In case of Transformer
        else:
            representation = []
            sizes = [] 
            for i,word in enumerate(sentence_words):
                if i>0 and sentence_words[i-1][-1] == ' ':
                    e = tokenizer.encode(' ' + word.strip())
                    representation += e
                    sizes.append(len(e))
                else:
                    e = tokenizer.encode(word.strip())
                    representation += e
                    sizes.append(len(e))
            the_input = torch.tensor(representation).cuda()
            with torch.no_grad():
                the_input = the_input.unsqueeze(0)
                result = model(the_input)[0]
                result = result.squeeze(0)
            final_repr = []
            
            i = 0
            for size in sizes:
                to_append = torch.mean(result[i:i+size], dim=0)
                final_repr.append(to_append)
                i += size
            
            assert len(final_repr) == len(sentence_words), "Something is wrong"
            sentences_result.append(torch.stack(final_repr).squeeze(1))
    # Return catted, noncatted
    return torch.cat([s for s in sentences_result]),  torch.tensor(pos_result), [s for s in sentences_result] #, global_words

def create_data(filename: str, lm, w2i, pos_vocab=None, cutoff=None):
    """Create whole dataset """
    global pos_w2i
    ud_parses = parse_corpus(filename)[:cutoff]
    print("Creating data for", len(ud_parses))
    sen_reps, pos_tags, lijstje = fetch_sen_reps(ud_parses, lm, w2i)

    return sen_reps, pos_tags, lijstje, ud_parses


def create_or_load_both_data(set_type:str, lm, w2i, pos_vocab=None, cutoff=None, extra_transformer = None):
    """
    Args:
        set_type: (train,dev,test)
        lm: language model to use
        w2i: corresponding tokenizer/dictionary
        pos_vocab: existing pos_vocab object, set to None for the very first set you load
    Returns:
        x,y: Tensors of size pos_length*repr_size and pos_length
        vocab: vocab to use for the next iteration if first time saving, None otherwise
        words: all words to use for control task
    """
    # Remember original set type, may be overwritten with additional transformer information
    original_set_type = set_type
    model_name = 'RNN' if type(lm) == RNNModel else 'transformer'
    if extra_transformer == 'BART':
        #model_name += 'BART'
        set_type   += '_BART'
    if extra_transformer == 'XLNet':
        #model_name += 'XLNet'
        set_type   += '_XLNet'
    if extra_transformer == 'TransformerXL':
        #model_name += 'XLNet'
        set_type   += '_TransformerXL'
    if extra_transformer == 'T5':
        #model_name += 'T5'
        set_type   += '_T5'

    save_filename = os.path.join('corpus', model_name + '_pos'+set_type+'.pickle')
    #words_filename = os.path.join('words', set_type+'.pickle')
    if os.path.exists(save_filename):
        with open(save_filename, "rb") as f: 
            l = pickle.load(f)
        #with open(words_filename, "rb") as f:
        #    words = pickle.load(f)
        return l['x'], l['y']#words

    # If not exists
    set_type = original_set_type
    x,y, lijstje, corpus = create_data(
            os.path.join('data', 'en_ewt-ud-'+set_type+'.conllu'),
            lm,  
            w2i,
            pos_vocab,
            cutoff
        )

    print("Data created. Pickling now")

    if not os.path.exists(save_filename):
        with open(save_filename, "wb") as f:
            pickle.dump({"x":x, "y":y}, f)
    if extra_transformer == 'TransformerXL':
        #model_name += 'XLNet'
        set_type   += '_TransformerXL'
    save_file = os.path.join('corpus', model_name + "_structural" + set_type + ".pickle")

    true_distances = create_gold_distances(corpus)
    print("Data created,pickling")
    if not os.path.exists(save_file):
        with open(save_file, "wb") as f: 
            pickle.dump((true_distances, lijstje), f)
    return x,y
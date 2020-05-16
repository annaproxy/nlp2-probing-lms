import pickle 

import torch
import os
from torch import Tensor 
from typing import List 
from lstm.model import RNNModel 
from conllu import parse_incr, TokenList
from tqdm import tqdm

pos_w2i = dict()
pos_i2w = dict()
last = 0

from lstm.model import RNNModel

# READ DATA
# If stuff like `: str` and `-> ..` seems scary, fear not! 
# These are type hints that help you to understand what kind of argument and output is expected.
def parse_corpus(filename: str) -> List[TokenList]:
    """Parses a conllu file into a List of TokenLists"""
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    return ud_parses

def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, concat=True, get_pos = False) -> List[Tensor]:
    """
    Returns a list of length len(ud_parses)
    """
    if get_pos:
        global last, pos_w2i, pos_i2w
        pos_result = []
        
    model.eval()
    doing_lstm = type(model) == RNNModel
    print(f"Doing LSTM: {doing_lstm}")
    sentences_result = []
    global_words = []
    
    for sentence_nr, sentence in tqdm(enumerate(ud_parses)):
        sentence_words = []
        
        # First build string sentence repr with spaces and such
        for i, token in enumerate(sentence):
            if get_pos:
                postag = token['upostag']
                if postag in pos_w2i:
                    posindex = pos_w2i[postag]
                else:
                    posindex = last
                    pos_w2i[postag] = last 
                    pos_i2w[last] = postag 
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
            the_input = torch.tensor(representation)
            with torch.no_grad():
                result = model(the_input)[0]
            final_repr = []
            
            i = 0
            for size in sizes:
                to_append = torch.mean(result[i:i+size], dim=0)
                final_repr.append(to_append)
                i += size
            
            assert len(final_repr) == len(sentence_words), "Something is wrong"
            sentences_result.append(torch.stack(final_repr).squeeze(1))
           
    if concat:
        yes = torch.cat([s for s in sentences_result], dim=0)
        if get_pos: return yes, torch.tensor(pos_result), global_words
        return yes
    
    # Assume concat means structural probe, means no pos
    return [s for s in sentences_result] #, global_words


def bitch(x): return x+5
# I provide the following sanity check, that compares your representations against a pickled version of mine.
# Note that I use the DistilGPT-2 LM here. For the LSTM I used 0-valued initial states.
def assert_sen_reps(transformer, tokenizer, lstm, vocab):
    with open('distilgpt2_emb1.pickle', 'rb') as f:
        distilgpt2_emb1 = pickle.load(f)

    with open('lstm_emb1.pickle', 'rb') as f:
        lstm_emb1 = pickle.load(f)
    
    corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')[:1]

    own_distilgpt2_emb1 = fetch_sen_reps(corpus, transformer, tokenizer)
    own_lstm_emb1 = fetch_sen_reps(corpus, lstm, vocab)
    print(distilgpt2_emb1.shape, own_distilgpt2_emb1.shape)
    
    assert distilgpt2_emb1.shape == own_distilgpt2_emb1.shape
    assert lstm_emb1.shape == own_lstm_emb1.shape
    assert torch.allclose(distilgpt2_emb1, own_distilgpt2_emb1,atol=1e-05), "GPT2 embeddings don't match!"
    assert torch.allclose(lstm_emb1, own_lstm_emb1,atol=1e-05), "LSTM embeddings don't match!"

    print("All is well!")

def create_data(filename: str, lm, w2i, pos_vocab=None, cutoff=None):
    """Create whole dataset """
    global pos_w2i
    ud_parses = parse_corpus(filename)[:cutoff]
    print("Creating data for", len(ud_parses))
    sen_reps, pos_tags, global_words = fetch_sen_reps(ud_parses, lm, w2i, concat=True, get_pos=True)
    pos_vocab = pos_w2i

    return sen_reps, pos_tags, pos_vocab, global_words

def create_or_load_pos_data(set_type:str, lm, w2i, pos_vocab=None, cutoff=None):
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
    model_name = 'RNN' if type(lm) == RNNModel else 'transformer'

    save_filename = os.path.join('corpus', model_name + '_pos'+set_type+'.pickle')
    words_filename = os.path.join('words', set_type+'.pickle')
    if os.path.exists(save_filename):
        with open(save_filename, "rb") as f: 
            l = pickle.load(f)
        with open(words_filename, "rb") as f:
            words = pickle.load(f)
        return l['x'], l['y'], None, words

    # If not exists
    x,y,vocab,words = create_data(
            os.path.join('data', 'en_ewt-ud-'+set_type+'.conllu'),
            lm,  
            w2i,
            pos_vocab,
            cutoff
        )

    print("Data created. Pickling now")

    # Pickle corpus and true y-labels
    if not os.path.exists("corpus"):
        os.makedirs("corpus")
    with open(save_filename, "wb") as f:
        pickle.dump({"x":x, "y":y}, f)

    # Pickle words for control task
    if not os.path.exists("words"):
        os.makedirs("words") 
    with open(words_filename,"wb") as fp: 
        pickle.dump(words, fp)

    return x,y,vocab,words
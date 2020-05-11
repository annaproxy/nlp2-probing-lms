
import torch.utils.data as data
from collections import defaultdict
import torch 

class POSDataset(data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

class StructuralDataset(data.Dataset):
    def __init__(self, gold_distances, embs):
        self.gold_distances = gold_distances
        self.embs = embs
        
    def __len__(self):
        return len(self.embs)

    def __getitem__(self, index):
        return self.gold_distances[index], self.embs[index], len(self.gold_distances[index])


def pad_batch(x):
    batch_size = len(x)
    dists, embs, lengths =  list(zip(*x))
    try:
        a = embs[0].shape #print(embs[0].shape)
    except:
        print("weird shit? Why is this a list sometimes?")
        embs = embs[0]
        #print(len(embs[0]), embs[0][0].shape, embs[0][2].shape)
    max_length = max(lengths)

    padded_dists = torch.zeros((batch_size, max_length, max_length)) - 1
    padded_embs = torch.zeros((batch_size, max_length, embs[0].shape[-1])) - 1
    for i, l in enumerate(lengths):
        padded_embs[i, 0:l, :] = embs[i][0:l]
        padded_dists[i, 0:l, 0:l] = dists[i][0:l]

    return padded_dists, padded_embs, torch.tensor(lengths)


def find_distribution(loader):
    result = defaultdict(lambda:0)
    for _, y in loader:
        result[y.item()] += 1
    return result
import torch.nn as nn
import copy
class POSProbe(nn.Module):
    def __init__(self, repr_size, pos_size, hidden_size = 0, dropout=0):
        super().__init__()
        if hidden_size == 0:
            self.linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(repr_size, pos_size))
        else:
            self.linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(repr_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, pos_size)
            )
        
    def forward(self, x):
        return self.linear(x)
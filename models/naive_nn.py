import torch.nn as nn
import torch.nn.functional as F


class NaiveNN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(NaiveNN, self).__init__()
        self.embedding_layer = nn.Embedding(in_dim, h_dim)
        self.hidden_layer = nn.Linear(h_dim, in_dim)

    def forward(self, input):
        embed = self.embedding_layer(input)
        hidden = self.hidden_layer(embed)
        return F.log_softmax(hidden, dim=1)

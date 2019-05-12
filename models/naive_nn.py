import torch.nn as nn
import torch.nn.functional as F


class NaiveNN(nn.Module):
    def __init__(self, in_dim, batch_size, f_dim, h_dim):
        super(NaiveNN, self).__init__()
        self.input_layer = nn.Linear(f_dim, int(h_dim / 2))  # output -> (in_dim x h_dim/2)
        self.hidden_layer = nn.Linear(int(h_dim / 2), h_dim)  # output -> (in_dim x h_dim)

        #  flatten -> (in_dimm * h_dim)
        #  output -> (in_dimm * h_dim / 100)
        self.smaller = nn.Linear(batch_size * h_dim, int(batch_size * h_dim / 100))
        self.out = nn.Linear(int(batch_size * h_dim / 100), batch_size)

    def forward(self, inputs):
        # (1) input
        # print()
        # print()
        # print()
        # print(inputs.shape)

        # (2) 1st hidden layer
        # print(self.input_layer.weight.shape)
        embed = self.input_layer(inputs)
        # print(embed.shape)

        # (3) 2nd hidden layer
        # print(self.hidden_layer.weight.shape)
        hidden = self.hidden_layer(embed)
        # print(hidden.shape)

        # (4) 3rd hidden layer (flattened)
        flattened = hidden.flatten()
        # print(flattened.shape)
        # print(self.smaller.weight.shape)
        smaller = self.smaller(flattened)
        # print(smaller.shape)

        # (5) output layer 
        # print(self.out.weight.shape)
        output = self.out(smaller)
        # print(output.shape)

        # print(F.log_softmax(output).shape)
        # exit(11)
        return F.softmax(output, dim=0)

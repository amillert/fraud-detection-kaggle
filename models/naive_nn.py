import torch.nn as nn
import torch.nn.functional as F


class NaiveNN(nn.Module):
    def __init__(self, batch_size, f_dim, h_dim):
        super(NaiveNN, self).__init__()
        # input         -> (BATCH x f_dim)
        # input_layer   -> (f_dim x h_dim)
        self.hidden_layer = nn.Linear(f_dim, h_dim)
        # result        -> (BATCH x h_dim)

        # flattened     -> (1 x BATCH*h_dim)
        # flattened_layer -> (BATCH*h_dim x BATCH*h_dim/100)
        self.flattened_layer = nn.Linear(batch_size * h_dim, int(batch_size * h_dim / 100))
        # result        -> (1 x BATCH*h_dim/100)
        self.output_layer = nn.Linear(int(batch_size * h_dim / 100), batch_size)
        # result       -> (BATCH)

    def forward(self, input):
        # (1) input
        # print(input.shape)

        # (2) 1st hidden layer
        # print(self.input_layer.weight.shape)
        # print(self.hidden_layer.weight.detach().numpy().mean())
        positive_out = self.hidden_layer(input)
        # print(positive_out.shape)

        # (3) 2nd hidden layer (flattened)
        positive_out = positive_out.flatten()
        # print(positive_out.shape)
        # print(self.flattened_layer.weight.shape)
        positive_out = self.flattened_layer(positive_out)
        # print(positive_out.shape)

        # (4) 3rd output layer 
        # print(self.out.weight.shape)
        positive_out = self.output_layer(positive_out)
        # print(positive_out.shape)

        # (5) activation function
        # print(F.log_softmax(positive_out).shape)
        # exit(11)
        return F.softmax(positive_out, dim=0)

import torch.nn as nn
import torch
import numpy as np

class TimeEncode(nn.Module):
    # time encoding proposed by TGAT
    def __init__(self, dimension, dropout):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))  # np.linspace(start,end,num)
            .float().reshape(dimension, -1))  # 这是什么encoding方式？
        self.w.bias = nn.Parameter(torch.zeros(dimension).float())
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=1)
        t = self.w(t)
        t = self.dropout(t)
        # output has shape [bs,seq_len,dimension]
        output = torch.cos(t)
        return output
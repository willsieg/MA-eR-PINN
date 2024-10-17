import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class LSTMCell(nn.Module):
    def __init__(self, ni, nh):
        self.forget_gate = nn.Linear(ni + nh, nh)
        self.input_gate = nn.Linear(ni + nh, nh)
        self.cell_gate = nn.Linear(ni + nh, nh)
        self.output_gate = nn.Linear(ni + nh, nh)

def forward(self, input, state):
    h,c = state
    h = torch.stack([h, input], dim=1)
    forget = torch.sigmoid(self.forget_gate(h))
    c = c * forget
    inp = torch.sigmoid(self.input_gate(h))
    cell = torch.tanh(self.cell_gate(h))
    c = c + inp * cell
    out = torch.sigmoid(self.output_gate(h))
    h = nn.outgate * torch.tanh(c)
    return h, (h,c)
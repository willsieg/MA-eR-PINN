import torch
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn as nn
torch.set_default_dtype(torch.float32); torch.set_printoptions(precision=6, sci_mode=True)
SEED = 42; torch.manual_seed(SEED)

def print_info(model):
    print(f"{'-'*60}\n", model, f"{'-'*60}\nModel state_dict:")
    for param_tensor in model.state_dict(): 
        print(f"{param_tensor}:\t {model.state_dict()[param_tensor].size()}") 
    # --> Note torch.Size([4*hidden_size, input_size]) for LSTM weights because of i,o,f,g params concatenated


####################################################################################################
class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(LSTM1, self).__init__()

        self.input_size = input_size    # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers    # number of layers
        self.dropout = dropout

        # LSTM CELL --------------------------------
        self.lstm = nn.LSTM(
            self.input_size,            # The number of expected features in the input x
            self.hidden_size,           # The number of features in the hidden state h
            self.num_layers,            # Number of recurrent layers for stacked LSTMs. Default: 1
            batch_first=True,           # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Default: False
            bias=True,                  # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            dropout=self.dropout,       # usually: [0.2 - 0.5], introduces a Dropout layer on the outputs of each LSTM layer except the last layer, (dropout probability). Default: 0
            bidirectional=False,        # If True, becomes a bidirectional LSTM. Default: False
            proj_size=0,                # If > 0, will use LSTM with projections of corresponding size. Default: 0
            device=device)

        # LAYERS -----------------------------------
        self.relu = nn.ReLU()
        self.fc_test = nn.Linear(hidden_size, 1)

    def forward(self, packed_input, batch_size=None):
        # Propagate input through LSTM
        packed_out, _ = self.lstm(packed_input)
        #print(f"LSTM: Output after LSTM: {packed_out.data.shape}, {type(packed_out)}")

        # Unpack the output
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        #print(f"             after packing: {out.shape}, {type(out)}")

        # Output layers
        out = self.relu(out)  # relu
        #print(f"             after relu: {out.shape}, {type(out)}")

        out = self.fc_test(out)  # Use all outputs for prediction
        #print(f"             after fc: {out.shape}, {type(out)}")
        
        return out


####################################################################################################
class LSTM1_keep_hidden_cell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(LSTM1_keep_hidden_cell, self).__init__()

        self.input_size = input_size    # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers    # number of layers
        self.dropout = dropout
        self.device = device

        # LAYERS -----------------------------------
        self.relu = nn.ReLU()
        self.fc_test = nn.Linear(hidden_size, 1)

        # LSTM CELL --------------------------------
        self.lstm = nn.LSTM(self.input_size,self.hidden_size, self.num_layers, batch_first=True, bias=True, dropout=self.dropout, 
                            bidirectional=False, proj_size=0, device=self.device)
    
    def forward(self, packed_input, hidden=None, cell=None):
        # Initialize hidden state with zeros if not provided
        if hidden is None or cell is None:
            hidden = torch.zeros(self.num_layers, packed_input.batch_sizes[0].item(), self.hidden_size).to(self.device)
            cell = torch.zeros(self.num_layers, packed_input.batch_sizes[0].item(), self.hidden_size).to(self.device)

        packed_out, (hidden, cell) = self.lstm(packed_input, (hidden, cell))
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.relu(out)  # relu
        out = self.fc_test(out)  # Use all outputs for prediction
        return out, (hidden, cell)
        

####################################################################################################
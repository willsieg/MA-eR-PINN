import torch
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn as nn

def print_info(model):
    print(f"{'-'*60}\n", model, f"{'-'*60}\nModel state_dict:")
    #for param_tensor in model.state_dict(): 
    #    print(f"{param_tensor}:\t {model.state_dict()[param_tensor].size()}") 
    # --> Note torch.Size([4*hidden_size, input_size]) for LSTM weights because of i,o,f,g params concatenated


class LSTM1_packed_old_version(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(LSTM1_packed_old_version, self).__init__()
        # LSTM CELL --------------------------------
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0,device=device)
        # LAYERS -----------------------------------
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, packed_input, batch_size=None):
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.relu(out)  # relu
        out = self.dropout_layer(out)  # dropout
        out = self.fc1(out)  # fully connected layer 1
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)  # relu
        out = self.fc2(out)  # fully connected layer 2
        return out

    # Define the weight initialization function for LSTM and other layers
    def initialize_weights_lstm(self, init_type):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')  # HE INIT
                elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)  # NORMAL INIT
                elif init_type == 'default': continue  # TORCH DEFAULT INIT
            elif 'weight' in name:
                if param.dim() >= 2:  # Ensure the tensor has at least 2 dimensions
                    if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')  # HE INIT for FC layers
                    elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)  # NORMAL INIT for FC layers
                    elif init_type == 'default': continue  # TORCH DEFAULT INIT
            elif 'bias' in name and init_type != 'default': nn.init.constant_(param.data, 0)  # Initialize biases to 0

class DeepLSTM_v2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(DeepLSTM_v2, self).__init__()

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0,device=device) # LSTM Dropout = 0 !
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()   # nn.LeakyReLU(negative_slope=0.01)

    def forward(self, packed_input, batch_size=None):
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    # Define the weight initialization function for LSTM and other layers
    def initialize_weights_lstm(self, init_type):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')  # HE INIT
                elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)  # NORMAL INIT
                elif init_type == 'default': continue  # TORCH DEFAULT INIT
            elif 'weight' in name:
                if param.dim() >= 2:  # Ensure the tensor has at least 2 dimensions
                    if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')  # HE INIT for FC layers
                    elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)  # NORMAL INIT for FC layers
                    elif init_type == 'default': continue  # TORCH DEFAULT INIT
            elif 'bias' in name and init_type != 'default': nn.init.constant_(param.data, 0)  # Initialize biases to 0

class DeepLSTM_v3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(DeepLSTM_v3, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, device=device)  # LSTM Dropout = 0 !
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, packed_input, batch_size=None):
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.fc4(out)
        return out

    # Define the weight initialization function for LSTM and other layers
    def initialize_weights_lstm(self, init_type):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')  # HE INIT
                elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)  # NORMAL INIT
                elif init_type == 'default': continue  # TORCH DEFAULT INIT
            elif 'weight' in name:
                if param.dim() >= 2:  # Ensure the tensor has at least 2 dimensions
                    if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')  # HE INIT for FC layers
                    elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)  # NORMAL INIT for FC layers
                    elif init_type == 'default': continue  # TORCH DEFAULT INIT
            elif 'bias' in name and init_type != 'default': nn.init.constant_(param.data, 0)  # Initialize biases to 0
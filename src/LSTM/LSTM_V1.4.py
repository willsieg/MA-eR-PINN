# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
'''
---------------------------------------------------------------------
LSTM Training
Version: V1.4
Modified: 14.10.2024
---------------------------------------------------------------------
notebook can be converted to python script using: 
jupytext --to py FILENAME.ipynb
---------------------------------------------------------------------
'''

global IS_NOTEBOOK
IS_NOTEBOOK = False
try:    # if running in IPython
    shell = get_ipython().__class__.__name__ # type: ignore 
    # %reset -f -s
    # %matplotlib inline
    from IPython.display import display, HTML, Javascript
    from IPython.core.magic import register_cell_magic
    @register_cell_magic    # cells can be skipped by using '%%skip' in the first line
    def skip(line, cell): return
    from tqdm.notebook import tqdm # type: ignore
    IS_NOTEBOOK = True
except (NameError, ImportError):    # if running in script
    from tqdm import tqdm
    from tabulate import tabulate
    print("running in script mode")

# +
# IMPORTS ---------------------------------------------------------------------
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from copy import deepcopy
from datetime import datetime

#from sklearn.metrics import mean_squared_error
#from torchinfo import summary
#import pickle
#import random
#from scipy.signal import savgol_filter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset
torch.manual_seed(1);
# -

# DEVICE SELECTION ---------------------------------------------------------------------
global DEVICE
print(f"{'-'*60}\nTorch version: ", torch.__version__)
print('Cuda available: ',torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda") 
    #DEVICE = torch.device("cuda:0")   # or overwrite with explicit Core number
    print(f'Current Device: {torch.cuda.current_device()},  Total Count: {torch.cuda.device_count()}')
else:
    DEVICE = ("cpu")
print(f"   --> Using {DEVICE} device")

# SET SOURCE PATHS HERE
# ___

# +
# FILE SOURCES ---------------------------------------------------------------
parquet_folder = "/home/sieglew/data/processed" # Trip parquet files
#parquet_folder = r"C:\Users\SIEGLEW\OneDrive - Daimler Truck\MA\Code\SIEGLEW_COPY\data\processed" # Trip parquet files

save_model_folder = "/home/sieglew/pth" # Save model files

# +
# INPUT & TARGET SPECIFICATION ---------------------------------------------------
columns = ["signal_time", 
            "hirestotalvehdist_cval_icuc", "vehspd_cval_cpc", "altitude_cval_ippc", "bs_roadincln_cval", "ambtemp_cval_pt", "hv_batpwr_cval_bms1", 
            "hv_batmomavldischrgen_cval_1"]

# Time and Target column taken out
input_columns = columns[1:-1]
target_column = columns[-1]

# +
# PREPARE TRAIN & TEST SET ---------------------------------------------------
all_files = [os.path.join(parquet_folder, f) for f in os.listdir(parquet_folder) if f.endswith(".parquet")]

# slect Only first 100 Input-files in total
files = all_files[:100]

# Train & Test Sets
train_files, test_files = train_test_split(files, test_size=0.2, random_state=1)
# -

# FEATURE NORMALIZATION/SCALING -----------------------------------------------------------------
scaler = StandardScaler()   # Standardize features by removing the mean and scaling to unit variance
target_scaler = StandardScaler()  #MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range


# DATASET DEFINITION -----------------------------------------------------------------------
class TripDataset(Dataset):
    def __init__(self, file_list, scaler, target_scaler):
        self.file_list = file_list
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.data = []
        self.targets = []

        for file in self.file_list:
            df = pd.read_parquet(file, columns=columns, engine='fastparquet')
            df.fillna(df.median(), inplace=True)  # Basic Median Filling 
            
            X = df[input_columns].values
            y = df[target_column].values.reshape(-1, 1)     # reshape to match the shape of the input
            
            # Normalize inputs
            X = self.scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y).squeeze()
            
            # Append to data
            self.data.append(X)
            self.targets.append(y)

    def __len__(self):
        return sum(len(target) for target in self.targets)

    def __getitem__(self, index):
        # Find which file the index belongs to
        for i, target in enumerate(self.targets):
            if index < len(target):
                return (
                    torch.tensor(self.data[i][index], dtype=torch.float32).unsqueeze(0),  # Add time dimension
                    torch.tensor(target[index], dtype=torch.float32)
                )
            index -= len(target)
        raise IndexError("Index out of range")


# +
# GENERATE DATALOADERS  ---------------------------------------------------------------
batch_size = 4096

train_dataset = TripDataset(train_files, scaler, target_scaler)
test_dataset = TripDataset(test_files, scaler, target_scaler)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the size of the datasets
print(f"{'-'*60}\nTrain size:  {len(train_dataset)}")
print(f'Test size:   {len(test_dataset)}')


# -

# LSTM NETWORK -----------------------------------------------------------------------
class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device = DEVICE): #, num_classes, seq_length):
        super(LSTM1, self).__init__()

        self.input_size = input_size    # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers    # number of layers
        #self.num_classes = num_classes  # number of classes
        #self.seq_length = seq_length    # sequence length

        # LSTM CELL --------------------------------
        self.lstm = nn.LSTM(
            input_size = input_size,    # The number of expected features in the input x
            hidden_size = hidden_size,  # The number of features in the hidden state h
            num_layers = num_layers,    # Number of recurrent layers for stacked LSTMs. Default: 1
            batch_first = True,         # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Default: False
            bias = True,                # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            dropout = 0.2,              # usually: [0.2 - 0.5] , If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
            bidirectional = False,      # If True, becomes a bidirectional LSTM. Default: False
            proj_size = 0,              # If > 0, will use LSTM with projections of corresponding size. Default: 0
            device = DEVICE,
            dtype=torch.float32
            ) 
        
        #self.fc_1 =  nn.Linear(hidden_size, 128)  # fully connected 1
        #self.fc = nn.Linear(128, num_classes)     # fully connected last layer
        self.relu = nn.ReLU()
        self.fc_test =  nn.Linear(hidden_size, 1)

    
    def forward(self,input, batch_size = None):
        '''
        # initial hidden and internal states
        h_0 = torch.zeros(self.num_layers, input.size(0) if batch_size is None else batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, input.size(0) if batch_size is None else batch_size, self.hidden_size)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(input, (h_0, c_0)) # lstm with input, hidden, and internal state
        

        out = self.relu(hn.view(-1, self.hidden_size)) # reshaping the data for Dense layer next
        out = self.fc_1(out) # first Dense
        out = self.relu(out) # relu
        out = self.fc(out) # Final Output'''

        out, _ = self.lstm(input)
        out = self.relu(out) # relu
        out = self.fc_test(out[:, -1, :]) 

        return out

# ___

# +
# MODEL CONFIGURATION -----------------------------------------------------------------------

# LAYERS --------------------------------
input_size = len(input_columns)     # expected features in the input x
hidden_size = 64                    # features in the hidden state h
num_layers = 4                      # ecurrent layers for stacked LSTMs. Default: 1
num_classes = 1                     # output classes (=1 for regression)

# INSTANTIATE MODEL --------------------
model = LSTM1(input_size, hidden_size, num_layers).to(DEVICE)  #, num_classes, X_train_T_final.shape[1]
print(f"{'-'*60}\n",model)

# +
# TRAINING CONFIGURATION -----------------------------------------------------------------------
global NUM_EPOCHS

# HYPERPARAMETERS -----------------------
NUM_EPOCHS = 20
learning_rate = 1e-2 # 0.001 lr

# OPTIMIZER -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,
    weight_decay = 1e-5      # weight decay coefficient (default: 1e-2)
    #betas = (0.9, 0.95),    # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    #eps = 1e-8,             # term added to the denominator to improve numerical stability (default: 1e-8)
)

# LOSS FUNCTION ---------------------------
def loss_func(model_output, target):
    return F.mse_loss(model_output, target) # mean-squared error for regression

# or define criterion function:
criterion = nn.MSELoss()


# +
# print Model and Optimizer state_dicts
def print_state_dicts(model, optimizer=None):
    print(f"{'-'*60}\nModel state_dict:")
    for param_tensor in model.state_dict():
        print(f"{param_tensor}:\t {model.state_dict()[param_tensor].size()}")
        
    if optimizer:
        print("\nOptimizer state_dict:")
        for var_name in optimizer.state_dict():
            if var_name == 'param_groups':
                print(optimizer.state_dict()[var_name][0])

print_state_dicts(model, optimizer)
print(f"{'-'*60}\n{'-'*60}\n")


# -

# ___

# +
# Initialize a HTML table for performance tracking (if running in a notebook)
def initialize_table():
    table_html = """
    <table id="training_table" style="width:50%; border-collapse: collapse;">
        <thead style="position: sticky; top: 0; z-index: 1;">
            <tr>
                <th style="font-weight:bold; width:15%; text-align:left; padding: 10px; background-color: #404040;">Epoch</th>
                <th style="font-weight:bold; width:25%; text-align:left; padding: 10px; background-color: #404040;">Iteration</th>
                <th style="font-weight:bold; width:30%; text-align:left; padding: 10px; background-color: #404040;">Batch Loss</th>
                <th style="font-weight:bold; width:30%; text-align:left; padding: 10px; background-color: #404040;">Train Loss</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
    </table>
    <script>
        function addRow(epoch, step, loss, running_loss) {
            var table = document.getElementById("training_table").getElementsByTagName('tbody')[0];
            var row = table.insertRow(-1);
            var cell1 = row.insertCell(0);
            var cell2 = row.insertCell(1);
            var cell3 = row.insertCell(2);
            var cell4 = row.insertCell(3);
            cell1.style.textAlign = "left";
            cell2.style.textAlign = "left";
            cell3.style.textAlign = "left";
            cell4.style.textAlign = "left";
            cell1.innerHTML = epoch;
            cell2.innerHTML = step;
            cell3.innerHTML = loss;
            cell4.innerHTML = running_loss;
            var scrollableDiv = document.getElementById("scrollable_table");
            scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
        }
    </script>
    """

    return """<div id="scrollable_table" style="height: 300px; overflow-y: scroll;">""" + table_html + """</div>"""

# Initialize a list for performance tracking (if running in script mode)
training_table = []

# Function to add a row to the performance table
def add_row(epoch, iteration, batch_loss, train_loss):
    training_table.append([epoch, iteration, batch_loss, train_loss])

# Function to print the performance table
header_printed = False
def print_row():
    global header_printed
    headers = ["Epoch", "Iteration", "Batch Loss", "Train Loss"]
    col_widths = [12, 12, 12, 12]  # Define fixed column widths

    def format_row(row):
        return [str(item).ljust(width) for item, width in zip(row, col_widths)]

    if not header_printed:
        formatted_headers = format_row(headers)
        tqdm.write(tabulate([training_table[-1]], headers=formatted_headers, tablefmt="plain", colalign=("left", "left", "left", "left")))
        header_printed = True
    else:
        formatted_row = format_row(training_table[-1])
        tqdm.write(tabulate([training_table[-1]], headers=format_row(["", "", "", ""]), tablefmt="plain", colalign=("left", "left", "left", "left")))


# -

# TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
def train_model(model, optimizer, loss_func, train_loader):

    # output info on training process
    print(f"Training Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
        f"Model: {model.__class__.__name__}\t\tParameters on device: {next(model.parameters()).device}\n{'-'*60}\n"
        f"Train/Batch size:\t{len(train_loader.dataset)} / {train_loader.batch_size}\n"
        f"Optimizer:\t\t{optimizer.__class__.__name__}\nLR:\t\t\t{optimizer.param_groups[0]['lr']} \n{'-'*60}")
    
    if IS_NOTEBOOK: display(HTML(initialize_table()))

    train_losses = [] # collect loss
    start_time = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()   # set model to training mode
        running_loss = 0.0
        num_iterations = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        header_printed = False
        
        with tqdm(enumerate(train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
            for iter, (inputs, targets) in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{NUM_EPOCHS}")

                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)     # Move data to the GPU

                # zero gradients > forward pass > obtain loss function > apply backpropagation > update weights:
                optimizer.zero_grad()
                outputs = model(inputs) 
                loss = criterion(outputs.squeeze(), targets) 
                loss.backward() 
                # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # optional: Gradient Value Clipping
                optimizer.step()

                # Update the performance table
                if iter % (num_iterations//4) == 0 and iter != num_iterations//4*4:
                    if IS_NOTEBOOK:
                        display(Javascript(f"""addRow("", "{iter}", "{loss.item():.4f}", "");"""))
                    else:
                        add_row(f" ", f"{iter}",f"{loss.item():.4f}", " "); print_row()
                elif iter == 1:
                    if IS_NOTEBOOK:
                        display(Javascript(f"""addRow("<b>{epoch}/{NUM_EPOCHS}", "{iter}/{num_iterations}", "{loss.item():.4f}", "");"""))
                    else:
                        add_row(f"{epoch}/{NUM_EPOCHS}", f"{iter}/{num_iterations}",f"{loss.item():.4f}", " "); print_row()

                # Update running loss and progress bar
                running_loss += loss.item() # acculumate loss for epoch
                tepoch.set_postfix(loss=loss.item()); tepoch.update(1)

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Update the performance table
        if IS_NOTEBOOK:
            display(Javascript(f"""addRow("", "{iter}", "{loss.item():.4f}", "<b>{avg_train_loss:.4f}");"""))
        else:
            add_row(f" ", f"{iter}",f"{loss.item():.4f}", f"{avg_train_loss:.4f}"); print_row()

    end_time = time.perf_counter()
    print(f"{'-'*60}\nTraining Completed.\tExecution Time: ", f"{(end_time - start_time):.2f}", f"s\n")

    return train_losses


# +
# NETWORK TRAINING -----------------------------------------------------------------

train_losses = train_model(
    model = model, 
    optimizer = optimizer, 
    loss_func = criterion, 
    train_loader = train_loader
    )

# +
# plot training performance:
plt.style.use('ggplot')
fig, ax1 = plt.subplots(figsize=(8,3))

# Set x-axis to integers starting from 1
ax1.set_xlabel('Epoch')
ax1.set_xticks(range(1, NUM_EPOCHS + 1))

plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='train_loss')
fig.tight_layout(); plt.legend();

# +
# EVALUATION -----------------------------------------------------------------
model.eval() # set model to evaluation mode
test_loss = 0

with torch.no_grad():

    for iter, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)

        # Optional: Inverse-transform outputs and targets for evaluation
        # You can use `scaled_outputs` and `scaled_targets` for error metrics in the original scale if needed
        scaled_outputs = target_scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
        scaled_targets = target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))

        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss:  {test_loss:.4f}")
print(f"Iterations: {iter}")

# +
# SAVE MODEL  -----------------------------------------------------------------

# create unique model name
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
model_name = f"{model.__class__.__name__}_{timestamp}"
model_destintion_path = f"{save_model_folder}/{model_name}.pth"

# save state_dict
torch.save(model.state_dict(), model_destintion_path)
print(f"Model saved to:\t {model_destintion_path}")

# +
# keep best performing model:
#best_model_state = deepcopy(model.state_dict())

# +
# %%skip
# PLOT RESULTS -----------------------------------------------------------------

# Reverse Transformation of the latest output and target
scaled_outputs = target_scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
scaled_targets = target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))


plt.figure(figsize=(12,4)) 

plt.xlabel('Time in s')
plt.ylabel('Battery Energy in kWh')

plt.plot(scaled_outputs, label='Actual Data') # actual plot
plt.plot(scaled_targets, label='Predicted Data') # predicted plot
plt.title('Time-Series Prediction')
plt.legend();

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
Version: V1.6
Modified: 31.10.2024
William Siegle
---------------------------------------------------------------------
notebook can be converted to python script using: 
(python -m) jupytext --to py FILENAME.ipynb
---------------------------------------------------------------------
'''

global IS_NOTEBOOK
IS_NOTEBOOK = False
try:    # if running in IPython
    shell = get_ipython().__class__.__name__ # type: ignore 
    # #%reset -f -s
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
    print(f"{'-'*60}\nRunning in script mode")

# +
# IMPORTS ---------------------------------------------------------------------
import sys
import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#from torchinfo import summary
#import pickle
#import random
#from scipy.signal import savgol_filter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchmetrics.functional import mean_squared_error
torch.manual_seed(1);
# -

# ___
# LOCATE DEVICES & SYSTEM FOLDERS

# DEVICE SELECTION ---------------------------------------------------------------------
global DEVICE
print(f"{'-'*60}\nTorch version: ", torch.__version__)
print('Cuda available: ',torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda") 
    #DEVICE = torch.device("cuda:1")   # or overwrite with explicit Core number
    print(f'Currently Selected Device: {torch.cuda.current_device()},  Total Count: {torch.cuda.device_count()}')
else:
    DEVICE = ("cpu")
print(f"   --> Using {DEVICE} device")

# ------------ LOCATE REPOSITORY/DATASTORAGE IN CURRENT SYSTEM ENVIRONMENT  --------------
global ROOT, DATA_PATH                                #|
ROOT = Path('../..').resolve()                      #|
print(f"{'-'*60}\n{ROOT}:\t{', '.join([_.name for _ in ROOT.glob('*/')])}")             #|
sys.path.append(os.path.abspath(ROOT))                                                  #|
from data import get_data_path  # paths set in "data/__init__.py"                       #|
DATA_PATH = get_data_path()                                                             #|
print(f"{DATA_PATH}:\t\t{', '.join([_.name for _ in DATA_PATH.glob('*/')])}")           #|
# ----------------------------------------------------------------------------------------

# FILE SOURCES ---------------------------------------------------------------
input_folder = Path(DATA_PATH, "final", "trips_processed_resampled") # Trip parquet files
pth_folder = Path(ROOT, "src", "models", "pth")
print(f"{'-'*60}\nInput Data: {input_folder}\nStored model: {pth_folder}")

# ___
# DATA PREPROCESSING

# +
# PREPARE TRAIN & TEST SET ---------------------------------------------------
all_files = [Path(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".parquet")]
files = all_files
print(f"{'-'*60}\nTotal Files: {len(files)}\n{'-'*60}")
# ---------------------------------------------------
df = pd.read_parquet(Path(input_folder, random.choice(files)), engine='fastparquet')
all_signals = df.columns
assert len(all_signals) == 58

# get df stats:
'''df.info()
from scipy.stats import shapiro
nd = []
for sig in df.columns:
    if np.ptp(df[sig]) != 0:
        _ , p = shapiro(df[sig])
        if p > 0.05:
            nd.append(sig)
print(f"{'-'*60}\nNormal Distributed Signals: {len(nd)}\n{'-'*60}")''';

# +
# INPUT & TARGET SPECIFICATION ---------------------------------------------------
# these signals are required for the physical Model calculation:
base_signals = ["signal_time", 
            "hirestotalvehdist_cval_icuc", "vehspd_cval_cpc", "altitude_cval_ippc", "airtempoutsd_cval_cpc", "hv_batpwr_cval_bms1", "emot_pwr_cval",
            "bs_roadincln_cval", "roadgrad_cval_pt"]

# these signals have to be dropped in order for appropriate training:
columns_to_drop = ["hv_bat_soc_cval_bms1", "latitude_cval_ippc", "longitude_cval_ippc", "signal_time", "signal_ts"]

# ---------------------------------------------------
target_column = "hv_batmomavldischrgen_cval_1"
input_columns = all_signals.drop(columns_to_drop + [target_column])


# -

# DATASET DEFINITION -----------------------------------------------------------------------
class TripDataset(Dataset):
    def __init__(self, file_list, scaler, target_scaler, fit=False):
        self.file_list = file_list
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.fit = fit
        self.data = []
        self.targets = []

        for file in self.file_list:
            # DATA PREPROCESSING -----------------------------------------------------------
            df = pd.read_parquet(file, engine='fastparquet')

            # assigning inputs and targets and reshaping ---------------
            X = df[input_columns].values
            y = df[target_column].values.reshape(-1, 1)     # reshape to match the shape of the input
            
            if self.fit == True:
                # Normalize inputs
                X = self.scaler.fit_transform(X)
                y = self.target_scaler.fit_transform(y).squeeze()
            else:
                X = self.scaler.transform(X)    
                y = self.target_scaler.transform(y).squeeze()
            
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


# FEATURE NORMALIZATION/SCALING -----------------------------------------------------------------
scaler = StandardScaler()   # Standardize features by removing the mean and scaling to unit variance
target_scaler = MinMaxScaler(feature_range=(0, 1))  #MinMaxScaler(feature_range=(0, 1))  # Transform features by scaling each feature to a given range

# DATA SET SPLITTING -----------------------------------------------------------------------
# train_subset, test_subset = train_test_split(files, test_size=0.2, random_state=1)
train_subset, val_subset, test_subset = random_split(files, [0.8, 0.1, 0.1])

# +
# GENERATE DATALOADERS  ---------------------------------------------------------------
batch_size = 1024 # [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# TRAIN  ------------------------------------------------------------
train_dataset = TripDataset(train_subset, scaler, target_scaler, fit=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# VAL ------------------------------------------------------------
val_dataset = TripDataset(val_subset, scaler, target_scaler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# TEST ------------------------------------------------------------
test_dataset = TripDataset(test_subset, scaler, target_scaler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the size of the datasets
print(f"{'-'*60}\nTrain size:  {len(train_dataset)}")
print(f'Val. size:   {len(val_dataset)}')
print(f'Test size:   {len(test_dataset)}')


# -

# ___
# NETWORK ARCHITECTURE

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
            self.input_size,            # The number of expected features in the input x
            self.hidden_size,           # The number of features in the hidden state h
            self.num_layers,            # Number of recurrent layers for stacked LSTMs. Default: 1
            batch_first = True,         # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Default: False
            bias = True,                # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            dropout = 0.2,              # usually: [0.2 - 0.5] ,introduces a Dropout layer on the outputs of each LSTM layer except the last layer, (dropout probability). Default: 0
            bidirectional = False,      # If True, becomes a bidirectional LSTM. Default: False
            proj_size = 0,              # If > 0, will use LSTM with projections of corresponding size. Default: 0
            device = DEVICE,
            dtype = torch.float32
            ) 
        
        # --------------------------------
        #self.fc_1 =  nn.Linear(hidden_size, 128)  # fully connected 1
        #self.fc = nn.Linear(128, num_classes)     # fully connected last layer
        # --------------------------------
        self.relu = nn.ReLU()
        self.fc_test =  nn.Linear(hidden_size, 1)

    
    def forward(self, input, batch_size = None):
        '''        
        # initial hidden and internal states
        # --------------------------------
        h_0 = torch.zeros(self.num_layers, input.size(0) if batch_size is None else batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, input.size(0) if batch_size is None else batch_size, self.hidden_size)  
        # --------------------------------
        out = self.relu(hn.view(-1, self.hidden_size)) # reshaping the data for Dense layer next
        out = self.fc_1(out) # first Dense
        out = self.relu(out) # relu
        out = self.fc(out) # Final Output
        '''

        # Propagate input through LSTM
        # --------------------------------
        # output, (hn, cn) = self.lstm(input, (h_0, c_0)) # lstm with input, hidden, and internal state
        # input shape:      (batch_size, seq_length, input_size)
        # output shape:     (batch_size, seq_length, hidden_size)
        # --------------------------------
        out, _ = self.lstm(input)


        # ouput layers
        # --------------------------------
        out = self.relu(out) # relu
        out = self.fc_test(out[:, -1, :])  
        #out = self.fc_test(out)

        return out

# +
# MODEL CONFIGURATION -----------------------------------------------------------------------

# LAYERS --------------------------------
input_size = len(input_columns)     # expected features in the input x
hidden_size = 64                    # features in the hidden state h
num_layers = 2                      # recurrent layers for stacked LSTMs. Default: 1
num_classes = 1                     # output classes (=1 for regression)

# INSTANTIATE MODEL --------------------
model = LSTM1(input_size, hidden_size, num_layers).to(DEVICE)  #, num_classes, X_train_T_final.shape[1]
print(f"{'-'*60}\n",model)
# -

# ___
# TRAINING SETUP

# +
# TRAINING CONFIGURATION -----------------------------------------------------------------------
global NUM_EPOCHS

# HYPERPARAMETERS -----------------------
NUM_EPOCHS = 20
learning_rate = 1e-3 # 0.001 lr

# OPTIMIZER -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,
    weight_decay = 1e-5      # weight decay coefficient (default: 1e-2)
    #betas = (0.9, 0.95),    # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    #eps = 1e-8,             # term added to the denominator to improve numerical stability (default: 1e-8)
)

# LOSS FUNCTION ---------------------------
def loss_fn(model_output, target):
    loss = F.mse_loss(model_output, target) # mean-squared error for regression
    return loss

# or define criterion function:
criterion = nn.MSELoss()
#criterion = nn.MSELoss(reduction='mean')

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
    <table id="training_table" style="width:60%; border-collapse: collapse;">
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

# -----------------------
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
    col_widths = [14, 14, 14, 14]  # Define fixed column widths

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
def train_model(model, optimizer, loss_fn, train_loader, val_loader = None):

    def validate_model(model, val_loader, loss_fn):
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = loss_fn(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)  # Calculate average validation loss
        return val_loss

    # output info on training process
    print(f"Training Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
        f"Model: {model.__class__.__name__}\t\tParameters on device: {next(model.parameters()).device}\n{'-'*60}\n"
        f"Train/Batch size:\t{len(train_loader.dataset)} / {train_loader.batch_size}\n"
        f"Loss:\t\t\t{loss_fn}\nOptimizer:\t\t{optimizer.__class__.__name__}\nLR:\t\t\t"
        f"{optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")
    
    if IS_NOTEBOOK: display(HTML(initialize_table()))

    # TRAINING LOOP:
    train_losses, val_losses = [], [] # collect loss
    start_time = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()   # set model to training mode
        running_loss = 0.0
        num_iterations = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        header_printed = False
        
        with tqdm(enumerate(train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
            for iter, (inputs, targets) in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{NUM_EPOCHS}")

                # -------------------------------------------------------------
                # Move data to the GPU
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)  
                # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                optimizer.zero_grad()
                outputs = model(inputs) 
                loss = loss_fn(outputs.squeeze(), targets) 
                loss.backward() 
                # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # optional: Gradient Value Clipping
                optimizer.step()

                # -------------------------------------------------------------
                # Update the performance table
                if iter % (num_iterations//4) == 0 and iter != num_iterations//4*4:
                    add_row(f" ", f"{iter}",f"{loss.item():.6f}", " ")
                    if IS_NOTEBOOK:
                        display(Javascript(f"""addRow("", "{iter}", "{loss.item():.6f}", "");"""))
                    else:
                        print_row()
                elif iter == 1:
                    add_row(f"{epoch}/{NUM_EPOCHS}", f"{iter}/{num_iterations}",f"{loss.item():.6f}", " ")
                    if IS_NOTEBOOK:
                        display(Javascript(f"""addRow("<b>{epoch}/{NUM_EPOCHS}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");"""))
                    else:
                        print_row()
                        
                # -------------------------------------------------------------
                # Update running loss and progress bar
                running_loss += loss.item() # acculumate loss for epoch
                tepoch.set_postfix(loss=loss.item()); tepoch.update(1)

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Update the performance table
        add_row(f" ", f"{iter}",f"{loss.item():.6f}", f"{avg_train_loss:6f}")
        if IS_NOTEBOOK:
            display(Javascript(f"""addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");"""))
        else:
            print_row()

        # VALIDATION
        if val_loader:
            val_loss = validate_model(model, val_loader, loss_fn)
            val_losses.append(val_loss)
            # Update the performance table
            add_row(f" ", f"Validation Loss:",f"{val_loss:.6f}", f"")
            if IS_NOTEBOOK:
                display(Javascript(f"""addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.4f}", "");"""))
            else:
                print_row()

        # save epochs results:


    print(f"{'-'*60}\nTraining Completed.\tExecution Time: ", f"{(time.perf_counter() - start_time):.2f}", f"s\n")
    output = {"train_losses": train_losses, "val_losses": val_losses, "epoch": epoch, "training_table": training_table}
    return output

# ___
# NETWORK TRAINING

# NETWORK TRAINING -----------------------------------------------------------------
trained = train_model(
    model = model, 
    optimizer = optimizer, 
    loss_fn = criterion, 
    train_loader = train_loader,
    val_loader = val_loader
    )

# +
# SAVE MODEL  -----------------------------------------------------------------
# create unique model name
model_name = f'{model.__class__.__name__}_{datetime.now().strftime("%y%m%d_%H%M%S")}'
model_destination_path = Path(pth_folder, model_name + ".pth")

# SAVE STATE_DICT FOR MODEL INFERENCE -----------------
state = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_func': criterion, 
    'train_loader': train_loader, 
    'val_loader': val_loader, 
    'training_table': trained["training_table"],
    'train_losses': trained["train_losses"],
    'val_losses': trained["val_losses"],
    'epoch': trained["epoch"]
}
torch.save(state, model_destination_path)
print(f"Model saved to:\t {model_destination_path}")
# -

from pathlib import Path
#model_destination_path = Path(pth_folder, "LSTM1_241102_232814.pth")
state = torch.load(model_destination_path, weights_only=False)
model.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])
model.eval(); # set model to evaluation mode for inference

training_table = state["training_table"]
train_losses = state["train_losses"]
val_losses = state["val_losses"]
epoch = state["epoch"]

# +
# get DataFrame of training metrics:
training_df = pd.DataFrame(training_table, columns=["Epoch", "Iteration", "Batch Loss", "Train Loss"])
# Extract the 'Train Loss' column and compare with the train_losses list
train_loss_column = training_df['Train Loss'].replace(['',' '], np.nan).dropna().astype(float).values
if any(abs(train_loss_column - train_losses) > 1e-3): print("Extracted and original Train Losses are not equal. Please check metrics table.")

# -------------------------------------
# plot training performance:
plt.style.use('ggplot')
fig, ax1 = plt.subplots(figsize=(8,3))

# Set x-axis to integers starting from 1
ax1.set_xlabel('Epochs')
ax1.set_xticks(range(1, NUM_EPOCHS + 1))

plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='train_loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='val_loss')
plt.yscale('log')
fig.tight_layout(); plt.legend();
# -

# ___
# EVALUATION / POST-PROCESSING

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
print(f"Iterations: {iter}/{math.floor(len(test_loader.dataset) / test_loader.batch_size)}")

# +
test_files = list(test_loader.dataset.file_list)
test_dataset = TripDataset(random.sample(test_files,1), scaler, target_scaler)
test_loader_2 = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
y_pred = []
with torch.no_grad():

    for iter, (inputs, targets) in enumerate(test_loader_2):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        y_pred.append(torch.mean(outputs).item())

y_pred = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
y_true = target_scaler.inverse_transform(np.array(test_loader_2.dataset.targets[0] ).reshape(-1, 1))

###############################################
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
plt.figure(figsize=(18,4)) 

plt.xlabel('Time in s')
plt.ylabel('Battery Energy in kWh')

plt.plot(y_true, label='Actual Data') # actual plot
plt.plot(np.arange(0, len(y_true), 1), y_pred, label='Predicted Data') # predicted plot
plt.title('Time-Series Prediction')
plt.legend();
# -

model.eval()
with torch.no_grad():
    # Randomly select a sequence from the test dataset
    seq = random.randint(0, len(test_loader.dataset.data) - 1)

    # Get the inputs and targets for the selected sequence
    test_inputs = torch.tensor(test_loader.dataset.data[seq], dtype=torch.float32).unsqueeze(0).to(DEVICE).contiguous()
    test_targets = torch.tensor(test_loader.dataset.targets[seq], dtype=torch.float32).to(DEVICE).contiguous()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Perform inference
    outputs = model(test_inputs)
    inference_loss = criterion(outputs, test_targets)
    print(f"Inference Loss:  {inference_loss.item():.4f}")

    # Inverse-transform the outputs and targets for evaluation
    y_pred = target_scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
    y_true = target_scaler.inverse_transform(test_targets.detach().cpu().numpy().reshape(-1, 1))

    # Print the first few predictions and true values for comparison
    print("Predicted values:", y_pred[:5].flatten())
    print("True values:", y_true[:5].flatten())

# +
# #%%skip
# PLOT RESULTS -----------------------------------------------------------------

# Reverse Transformation of the latest output and target
scaled_outputs = target_scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
scaled_targets = target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))
# -

y_true = target_scaler.inverse_transform(np.array(test_loader.dataset.targets[0] ).reshape(-1, 1))
y_true.shape

y_pred.shape

# +
from scipy.signal import savgol_filter

plt.figure(figsize=(18,4)) 

# Original and Smoothed plot
plt.xlabel('Time in s')
plt.ylabel('Battery Energy in kWh')
plt.plot(y_true, label='Actual Data') # actual plot
#plt.plot(np.arange(0, len(y_true), 10), y_pred, label='Predicted Data') # predicted plot

# Smoothed predicted plot
smoothed_y_pred = savgol_filter(y_pred.flatten(), window_length=100, polyorder=3)
plt.plot(np.arange(0, len(y_true), 10), smoothed_y_pred, label='Smoothed Predicted Data')

plt.title('Time-Series Prediction')
plt.legend()
plt.tight_layout()
plt.show()


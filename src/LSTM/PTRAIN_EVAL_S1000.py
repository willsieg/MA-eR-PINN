# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: sieglew
#     language: python
#     name: python3
# ---

# %reset -f -s
# %matplotlib inline
'''------------------------------------------------------------------
MA-eR-PINN: eRange Prediction using Physics-Informed Neural Networks
---------------------------------------------------------------------
Version: V2.1      Modified: 15.01.2025        William Siegle
---------------------------------------------------------------------
PTRAIN - Standard Pipeline Framework for Training the PINN
+ OPTUNA - Hyperparameter Optimization using Optuna
------------------------------------------------------------------''';

# +
# MA-eR-PINN: CONFIGURATION FILE -------------------------------------------------
from pathlib import Path
CONFIG = {
    # SYSTEM: ---------------------------------------------------------------------
    "GPU_SELECT":       0,
    "ROOT":             Path('../..').resolve(),
    "INPUT_LOCATION":   Path("TripSequences", "trips_processed_pinn_4"), 
    "OUTPUT_LOCATION":  Path("src", "models", "pth"),
    "SEED"  :           14,
    "MIXED_PRECISION":  True,
    "EARLY_STOPPING":   False,

    # DATA PREPROCESSING: ---------------------------------------------------------
    "TRAIN_VAL_TEST":   [0.7, 0.15, 0.15], # [train, val, test splits]
    "MAX_FILES":        None, # None: all files
    "MIN_SEQ_LENGTH":   600, # minimum sequence length in s to be included in DataSets
    "SCALERS":          {'feature_scaler': 'MinMaxScaler()','target_scaler': 'MinMaxScaler()','prior_scaler': 'MinMaxScaler()'},

    # FEATURES: -------------------------------------------------------------------
    "FEATURES":         ['accelpdlposn_cval','actdrvtrnpwrprc_cval','actualdcvoltage_pti1','actualspeed_pti1','actualtorque_pti1',
                        'airtempinsd_cval_hvac','airtempinsd_rq','airtempoutsd_cval_cpc','altitude_cval_ippc','brc_stat_brc1','brktempra_cval',
                        'bs_brk_cval','currpwr_contendrnbrkresist_cval','elcomp_pwrcons_cval','epto_pwr_cval','hv_bat_dc_momvolt_cval_bms1',
                        'hv_batavcelltemp_cval_bms1','hv_batcurr_cval_bms1','hv_batisores_cval_e2e','hv_batmaxchrgpwrlim_cval_1',
                        'hv_batmaxdischrgpwrlim_cval_1','hv_curr_cval_dcl1','hv_dclink_volt_cval_dcl1','hv_ptc_cabin1_pwr_cval','hv_pwr_cval_dcl1',
                        'lv_convpwr_cval_dcl1','maxrecuppwrprc_cval','maxtracpwrpct_cval','motortemperature_pti1','powerstagetemperature_pti1',
                        'rmsmotorcurrent_pti1','roadgrad_cval_pt','selgr_rq_pt','start_soc','txoiltemp_cval_tcm','vehspd_cval_cpc','vehweight_cval_pt'],                 
    "TARGETS":          ['hv_bat_soc_cval_bms1'],
    "PRIORS":           ['emot_soc_pred'],  

    # MODEL: -----------------------------------------------------------------------
    "HIDDEN_SIZE":      100,    # features in the hidden state h
    "NUM_LAYERS":       4,      # recurrent layers for stacked LSTMs. Default: 1
    "DROPOUT":          0.05,  # usually: [0.2 - 0.5]
    
    # TRAINING & OPTIMIZER: --------------------------------------------------------
    "NUM_EPOCHS":       3000,         # max epochs
    "BATCH_SIZE":       256,         # [2, 4, 8, 16, 32, 64, 128, 256]
    "LEARNING_RATE":    0.0003,     # 0.001 lr
    "OPTIMIZER":        'adam',     # ('adam', 'sgd', 'adamw')
    "WEIGHT_DECAY":     1e-7,       # weight decay coefficient (default: 1e-2)
    "WEIGHT_INIT_TYPE": 'he',       # ('he', 'normal', 'default')
    "CLIP_GRAD":        None,       # default: None
    "LRSCHEDULER":      "torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)",  # constant LR for 1.0 as multiplicative factor

    # LOSS FUNCTION: ---------------------------------------------------------------
    "LPSCHEDULER":      "ParameterScheduler(initial_value=1.0, schedule_type='reverse_sigmoid', total_epochs=2800)",
    #       'constant': [], 'time_based': ['decay_rate'], 'step_based': ['drop_rate', 'epochs_drop'], 'exponential': ['decay_rate'], 'linear': ['absolute_reduction'], 
    #       'cosine_annealing': ['total_epochs'], 'cyclic': ['base_lr', 'max_lr', 'step_size'], 'reverse_sigmoid': ['total_epochs']

}

# +
# LOSS FUNCTION MODULES ----------------------------------------------------------------  
global LOSS_FN
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, y_phys, l_p):
        mse_loss_value = self.mse_loss(y_pred, y_true)                      # loss w.r.t. data
        phys_loss_value = self.mse_loss(y_pred, y_phys)                     # loss w.r.t. physical model
        total_loss = (1 - l_p) * mse_loss_value + l_p * phys_loss_value     # total loss, weighted by l_p-factor
        loss_components = (mse_loss_value, phys_loss_value)
        return total_loss, loss_components

class CustomLoss_2(nn.Module):
    def __init__(self):
        super(CustomLoss_2, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, y_phys, l_p):
        mse_loss_value = self.mse_loss(y_pred, y_true)                      # loss w.r.t. data
        phys_loss_value = self.mse_loss(y_pred, y_phys)                     # loss w.r.t. physical model
        total_loss = mse_loss_value + l_p * phys_loss_value     # total loss, weighted by l_p-factor
        return total_loss

class CustomLoss_3(nn.Module):      # L_MSE + l*L_phys  + l*L_constraint
    def __init__(self):
        super(CustomLoss_3, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, y_phys, l_p):
        mse_loss_value = self.mse_loss(y_pred, y_true)                      # loss w.r.t. data
        phys_loss_value = self.mse_loss(y_pred, y_phys)                     # loss w.r.t. physical model
        constraint = torch.mean((torch.clamp(y_pred - y_phys, min=0)) ** 2) # penalty for y_pred > y_phys
        total_loss = mse_loss_value + l_p * constraint                 # total loss, weighted by l_p-factor
        loss_components = (mse_loss_value, phys_loss_value, constraint)
        return total_loss, loss_components




LOSS_FN = CustomLoss()
# -

# ___
# SETUP: Locate devices & system folders

# +
# LOCATE REPOSITORY/DATASTORAGE IN CURRENT SYSTEM ENVIRONMENT  ---------------------------
import sys, os
for key in CONFIG: globals()[key] = CONFIG[key]
if 'ROOT' not in globals(): ROOT = Path('../..').resolve()
sys.path.append(os.path.abspath(ROOT)); print(ROOT)

# INTERNAL MODULE IMPORTS ----------------------------------------------------------------
from src.__init__ import *
from src.utils.data_utils import *
from src.utils.preprocess_utils import *
from src.utils.eval_utils import *
from src.utils.Trainers import *
from src.utils.scheduler_utils import *
from src.models.lstm_models import *

# SETUP ENVIRONMENT ---------------------------------------------------------------------
DATA_PATH, IS_NOTEBOOK, DEVICE, LOG_FILE_NAME, TS = setup_environment(CONFIG, ROOT, SEED, GPU_SELECT)
if not IS_NOTEBOOK: output_file = open(f"{LOG_FILE_NAME}", "w"); sys.stdout = Tee(sys.stdout, output_file); sys.stderr = Tee(sys.stderr, output_file)

# FILE SOURCES ---------------------------------------------------------------
input_folder = Path(DATA_PATH, INPUT_LOCATION) # Trip parquet files
pth_folder = Path(ROOT, OUTPUT_LOCATION, f"{TS}")
pth_folder.mkdir(parents=True, exist_ok=True)
files, trip_lengths, indices_by_length, sorted_trip_lengths, all_signals, files_df = prepare_data(input_folder, pth_folder, MAX_FILES, MIN_SEQ_LENGTH, ROOT)
# -

# ___
# DATA SELECTION & PREPROCESSING

# +
# FEATURE SELECTION & SCALING ----------------------------------------------------------------------------
INPUT_COLUMNS = FEATURES; TARGET_COLUMN = TARGETS; PRIOR_COLUMN = PRIORS
print(f"{'-'*60}\nInput Signals:\t{len(FEATURES)}\nTarget Signals:\t{len(TARGETS)}\nPhysical Prior Signals:\t{len(PRIORS)}\n{'-'*60}")
scaler, target_scaler, prior_scaler = eval(SCALERS['feature_scaler']), eval(SCALERS['target_scaler']), eval(SCALERS['prior_scaler'])

# DATA SET SPLITTING AND SORTING ----------------------------------------------------------------
train_subset, val_subset, test_subset = random_split(files, TRAIN_VAL_TEST)

# DATALOADER SETTINGS ------------------------------------------------------------------
dataloader_settings = {'batch_size': 1, 'shuffle': True, 'collate_fn': collate_fn_PINN, 'num_workers': 8,
 'prefetch_factor': 4, 'persistent_workers': True, 'pin_memory': False if DEVICE.type == 'cpu' else True}

# PREPARE TRAIN, VAL & TEST DATALOADERS  ------------------------------------------------------------
train_subset, train_dataset, train_dataset_batches, train_loader = prepare_dataloader_PINN(train_subset, indices_by_length, \
    BATCH_SIZE, INPUT_COLUMNS, TARGET_COLUMN, PRIOR_COLUMN, scaler, target_scaler, prior_scaler, dataloader_settings, fit=True, drop_last=True)

val_subset, val_dataset, val_dataset_batches, val_loader = prepare_dataloader_PINN(val_subset, indices_by_length, \
    BATCH_SIZE, INPUT_COLUMNS, TARGET_COLUMN, PRIOR_COLUMN, scaler, target_scaler, prior_scaler, dataloader_settings, drop_last=False)

test_subset, test_dataset, test_dataset_batches, test_loader = prepare_dataloader_PINN(test_subset, indices_by_length, \
    BATCH_SIZE, INPUT_COLUMNS, TARGET_COLUMN, PRIOR_COLUMN, scaler, target_scaler, prior_scaler, dataloader_settings, drop_last=False)

# print dataset info
subset_files = print_dataset_sizes(train_dataset, val_dataset, test_dataset, train_subset, val_subset, test_subset, files)

# -----------------------------------------------------------------------------------
# Load dataloaders instead
#train_loader = torch.load('train_loader.pth')
#val_loader = torch.load('val_loader.pth')
#test_loader = torch.load('test_loader.pth')

# optional visualizations of padding preprocessing:
if IS_NOTEBOOK and False: 
    check_batch_PINN(train_loader)
    visualize_padding(BATCH_SIZE, trip_lengths, sorted_trip_lengths, train_loader, val_loader, test_loader)


# -

# ___
# MODEL & TRAINING CONFIGURATIONS

# +
# LSTM NETWORK -----------------------------------------------------------------------

class LSTM1_packed_old_version(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device=DEVICE):
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
    def __init__(self, input_size, hidden_size, num_layers, dropout, device=DEVICE):
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
    def __init__(self, input_size, hidden_size, num_layers, dropout, device='cpu'):
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


# -

# ___
# TRAINING

# +
# INSTANTIATE MODEL --------------------
model = DeepLSTM_v3(len(INPUT_COLUMNS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
model.initialize_weights_lstm(WEIGHT_INIT_TYPE); print_info(model)

# SET OPTIMIIZER, SCHEDULER AND LOSS MODULES ---------------------------
if OPTIMIZER=='adam': optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adamw': optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
elif OPTIMIZER=='sgd': optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch): return 1.0
lr_scheduler = eval(LRSCHEDULER); l_p_scheduler = eval(LPSCHEDULER)

# TRAIN -----------------------------------------------------------------
TRAINER = PTrainer_PINN(
    model = model, 
    optimizer = optimizer, 
    lr_scheduler = lr_scheduler,
    l_p_scheduler = l_p_scheduler,
    loss_fn = LOSS_FN, 
    train_loader = train_loader, 
    val_loader = val_loader, 
    test_loader = test_loader, 
    num_epochs = NUM_EPOCHS, 
    device = DEVICE, 
    is_notebook = IS_NOTEBOOK, 
    use_mixed_precision = MIXED_PRECISION, 
    use_early_stopping = EARLY_STOPPING, 
    clip_value = CLIP_GRAD, 
    log_file = Path(LOG_FILE_NAME).with_name(f"{TS}_log.txt"),
    config = CONFIG)

RESULTS = TRAINER.train_model()
# -

plot_training_performance(RESULTS)

# ___
# SAVE CHECKPOINT

# SAVE MODEL -----------------------------------------------------------------
CHECKPOINT, model_destination_path = save_checkpoint(TRAINER, train_loader, val_loader, test_loader, RESULTS, CONFIG, subset_files, pth_folder, TS)

# ___
# LOAD CHECKPOINT

# +
# model_destination_path = Path(pth_folder, "LSTM1_packed_old_version_241216_082030.pth")
# -

# LOAD MODEL -----------------------------------------------------------------
CHECKPOINT = load_checkpoint(model_destination_path, DEVICE)
for key in CHECKPOINT.keys(): globals()[key] = CHECKPOINT[key]
# load model and optimizer states --------------------------------------------
model.load_state_dict(model_state_dict)
optimizer.load_state_dict(optimizer_state_dict)
model.eval()  # set model to evaluation mode for inference

# ___
# EVALUATION

# +
# EVALUATION -----------------------------------------------------------------
# get file list of test subset
test_files = CHECKPOINT["test_files"]; print(f"{'-'*60}\nTest subset: {len(test_files)} files\n{'-'*60}")
# -------------------------------------
# evaluate model on test set
test_loss, all_outputs, all_targets, all_priors, all_original_lengths = TRAINER.evaluate_model()
# -------------------------------------
# Inverse-transform on all outputs and targets for evaluation
scaled_outputs = [target_scaler.inverse_transform(output_sequence.reshape(1, -1)).squeeze() for output_sequence in all_outputs]
scaled_targets = [target_scaler.inverse_transform(target_sequence.reshape(1, -1)).squeeze() for target_sequence in all_targets]
scaled_priors = [prior_scaler.inverse_transform(prior_sequence.reshape(1, -1)).squeeze() for prior_sequence in all_priors]

# concatenate:
all_y_true, all_y_pred, all_y_phys = np.concatenate(scaled_targets), np.concatenate(scaled_outputs), np.concatenate(scaled_priors)

# calculate evaluation metrics
print(f"Test Loss:\t\t{test_loss:.6f}")
metrics = calculate_metrics(all_y_true, all_y_pred) # [rmse, mae, std_dev, mape, r2, max_error]
print("Metrics (unweighted):")
mean_metrics = calculate_metrics_per_sequence(scaled_targets, scaled_outputs)
# -

# ___
# PLOT RESULTS

# +
# get random sample sequence from test set
sample_int = random.randint(1, len(test_files)-1)
y_true, y_pred, y_phys = scaled_targets[sample_int], scaled_outputs[sample_int], scaled_priors[sample_int]

# ----------->  add specific file selector for easier comparison
###############################################
# PLOT PREDICTION -----------------------------------------------------------------
# -----------
fig = plt.figure(figsize=(16,4)); plt.xlabel('Time in s', fontsize=14); plt.ylabel('SOC in %', fontsize=14); plt.title('Battery State of Charge: Prediction vs. Actual Data') 
plt.plot(y_true, label='Actual Data', linewidth=2)                                             # actual plot
plt.plot(y_phys, label='Physical Prior', linewidth=2, linestyle='--')                          # physical prior
plt.plot(np.arange(0, len(y_true), 1), y_pred, label='Predicted Data', linewidth=2)            # predicted plot
plt.ylim(0, 100); ax = plt.gca(); ax.set_xlim(left=-len(y_true)/100)
plt.tight_layout(pad=0.8); fig.patch.set_facecolor('white'); fig.set_facecolor('white'); plt.legend(); plt.grid(True)
plt.text(0.007, 0.05, f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}\nStd Dev: {np.std(y_true - y_pred):.4f}\nModel ID: {model_name_id}",\
     transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# -----------
fig = plt.figure(figsize=(16,4)); plt.xlabel('Time in s', fontsize=14); plt.ylabel('SOC in %', fontsize=14)
plt.plot(savgol_filter(y_true.flatten(), window_length=60, polyorder=3), label='Actual Data (Smoothed)', linewidth=2)   # actual plot
plt.plot(np.arange(0, len(y_true), 1), savgol_filter(y_pred.flatten(), window_length=300, polyorder=3), label='Predicted Data (Smoothed)', linewidth=2, color='r')   # predicted plot
plt.ylim(0, 100); ax = plt.gca(); ax.set_xlim(left=-len(y_true)/100)
plt.tight_layout(pad=0.8); fig.patch.set_facecolor('white'); fig.set_facecolor('white'); plt.legend(); plt.grid(True)
# -

if not IS_NOTEBOOK: sys.stdout.close(); sys.stderr.close()

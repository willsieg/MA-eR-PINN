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

# %reset -f -s

# +
'''------------------------------------------------------------------
eRange Prediction with LSTM
---------------------------------------------------------------------
Version: V1.7       Modified: 06.11.2024        William Siegle
---------------------------------------------------------------------
(python -m) jupytext --to py FILENAME.ipynb
------------------------------------------------------------------'''
#import pathlib
from pathlib import Path, WindowsPath, PosixPath

# SETTINGS ------------------------------------------------------------------------
CONFIG = {
    # SYSTEM: ---------------------------------------------------------------------
    "GPU_SELECT":       2, # {0,1,2,3, None: CPU only}
    "ROOT":             Path('../..').resolve(),
    "INPUT_LOCATION":   Path("TripSequences", "trips_processed_pinn"), 
    "OUTPUT_LOCATION":  Path("src", "models", "pth"),
    "SEED"  :           55,
    "PLOT_ACTIVE":      True,

    # DATA PREPROCESSING: ---------------------------------------------------------
    "TRAIN_VAL_TEST":   [0.8, 0.15, 0.05], # [train, val, test splits]
    "MAX_FILES":        None, # None: all files
    "SCALERS":          {'feature_scaler': 'MaxAbsScaler()',
                         'target_scaler': 'StandardScaler()',
                         'prior_scaler': 'StandardScaler()'},
    "MIN_SEQ_LENGTH":   600, # minimum sequence length in s to be included in DataSets

    # FEATURES: -------------------------------------------------------------------
    "FEATURES":         ["vehspd_cval_cpc", "altitude_cval_ippc", "airtempoutsd_cval_cpc", 
                        'roadgrad_cval_pt', "vehweight_cval_pt", "accelpdlposn_cval", "bs_brk_cval", "elcomp_pwrcons_cval",
                        "motortemperature_pti1", "powerstagetemperature_pti1","epto_pwr_cval",
                        'airtempinsd_cval_hvac', 'brktempra_cval', 'selgr_rq_pt'],
                        
    "TARGETS":          ['soc_gradient'],   #['hv_bat_soc_cval_bms1', 'soc_gradient'],
    "PRIORS":           ['emot_pwr_pred'],  

    # MODEL: -----------------------------------------------------------------------
    "HIDDEN_SIZE":      400,    # features in the hidden state h
    "NUM_LAYERS":       2,      # recurrent layers for stacked LSTMs. Default: 1
    "DROPOUT":          0.5,
    
    # TRAINING & OPTIMIZER: --------------------------------------------------------
    "NUM_EPOCHS":       30,
    "BATCH_SIZE":       16,   # [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    "LEARNING_RATE":    1e-3,   # 0.001 lr
    "OPTIMIZER":        "torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-3)",      
                        # weight_decay = 1e-4     # weight decay coefficient (default: 1e-2)
                        # betas = (0.9, 0.95),    # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
                        # eps = 1e-8,             # term added to the denominator to improve numerical stability (default: 1e-8)
    "LRSCHEDULER":      "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 2, factor = 0.5, min_lr = 1e-7)",

    # LOSS FUNCTION: ---------------------------------------------------------------
    "CRITERION":        "nn.SmoothL1Loss()", # ['nn.MSELoss()', 'nn.L1Loss()', 'nn.SmoothL1Loss()', 'nn.HuberLoss()', 'MASE()']
    "LOSS_FN":          "F.mse_loss(output, target)", # ['F.mse_loss(output, target)', 'F.l1_loss(output, target)', 'F.smooth_l1_loss(output, target)', 'F.huber_loss(output, target)', 'F.mase_loss(output, target)']

    # SAVE & LOAD: -----------------------------------------------------------------
    "MODE":             "train_mode", # ['train_mode', 'test_mode']
    "TRAIN_LOG":        "test1.txt",
}
# -

# ___
# LOCATE DEVICES & SYSTEM FOLDERS

# +
# LOCATE REPOSITORY/DATASTORAGE IN CURRENT SYSTEM ENVIRONMENT  ---------------------------
# ----------------------------------------------------------------------------------------
import sys, os
for key in CONFIG: globals()[key] = CONFIG[key]
if 'ROOT' not in globals(): ROOT = Path('../..').resolve()
sys.path.append(os.path.abspath(ROOT))

# INTERNAL MODULE IMPORTS ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------
from src.__init__ import *
from src.utils.data_utils import *
from src.utils.preprocess_utils import *
from src.utils.eval_utils import *
from src.utils.Trainers import *
from src.LSTM.lstm_models import *

# SETUP ENVIRONMENT ---------------------------------------------------------------------
DATA_PATH, IS_NOTEBOOK, DEVICE = setup_environment(CONFIG, ROOT, SEED, GPU_SELECT)
# -

# ___
# DATA PREPROCESSING

# +
# FILE SOURCES ---------------------------------------------------------------
input_folder = Path(DATA_PATH, INPUT_LOCATION) # Trip parquet files
pth_folder = Path(ROOT, OUTPUT_LOCATION)

files, trip_lengths, indices_by_length, sorted_trip_lengths, all_signals \
    = prepare_data(input_folder, pth_folder, MAX_FILES, MIN_SEQ_LENGTH, ROOT)

# +
# INPUT & TARGET SPECIFICATION ---------------------------------------------------
# these signals are required for the physical Model calculation:
base_signals = ["signal_time", "hirestotalvehdist_cval_icuc", "vehspd_cval_cpc", "altitude_cval_ippc", "airtempoutsd_cval_cpc", 
                "hv_batpwr_cval_bms1", "emot_pwr_cval","bs_roadincln_cval", "roadgrad_cval_pt"]

# these signals have to be dropped (from Features) in order for appropriate training:
columns_to_drop = ["signal_time",                       # works as index
                    "hirestotalvehdist_cval_icuc",      # starts from 0, obtained by speed integration
                    "latitude_cval_ippc",               # only GPS 
                    "longitude_cval_ippc",              # only GPS
                    "hv_batpwr_cval_bms1",              # directly related to target (soc_gradient)
                    "hv_batmomavldischrgen_cval_1",     # indirect target 1 in kWh
                    "hv_bat_soc_cval_bms1",              # indirect target 2 in %SoC
                    "soc_gradient",                     # actual target signal   
                    "emot_pwr_cval",                    # replaced as physical prior for PINN
                    "emot_pwr_pred",                    # actual physical prior for PINN
                    ]

# Ensure no element of "columns_to_drop" is included in "FEATURES"
assert not any(col in FEATURES for col in columns_to_drop), "Some columns to drop are still in FEATURES"

# ---------------------------------------------------
selection_1 = ["vehspd_cval_cpc", "altitude_cval_ippc", "airtempoutsd_cval_cpc", 'roadgrad_cval_pt', "vehweight_cval_pt", "accelpdlposn_cval", "bs_brk_cval", "elcomp_pwrcons_cval",
               "epto_pwr_cval", "motortemperature_pti1", "powerstagetemperature_pti1", 'airtempinsd_cval_hvac', 'brktempra_cval', 'selgr_rq_pt']
selection_2 = ["hirestotalvehdist_cval_icuc", "vehspd_cval_cpc", "altitude_cval_ippc", "airtempoutsd_cval_cpc", "hv_batpwr_cval_bms1", "emot_pwr_cval", "roadgrad_cval_pt"]
selection_3 = ["vehspd_cval_cpc", "altitude_cval_ippc", "airtempoutsd_cval_cpc", "vehweight_cval_pt"]

#FEATURES = list(set(all_signals) - set(columns_to_drop) - set(TARGETS) - set(PRIORS))
print(f"{'-'*60}\nInput Signals:\t{len(FEATURES)}\nTarget Signals:\t{len(TARGETS)}\nPhysical Prior Signals:\t{len(PRIORS)}")


# +
# FEATURE SELECTION  ----------------------------------------------------------------------------
INPUT_COLUMNS = FEATURES
TARGET_COLUMN = TARGETS
PRIOR_COLUMN = PRIORS

# FEATURE NORMALIZATION/SCALING -----------------------------------------------------------------
scaler = eval(SCALERS['feature_scaler'])
target_scaler = eval(SCALERS['target_scaler'])
prior_scaler = eval(SCALERS['prior_scaler'])
# -

INPUT_COLUMNS

# INSTANTIATE MODEL --------------------
model = LSTM1(len(INPUT_COLUMNS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DEVICE).to(DEVICE)
print_info(model)

# +
# GENERATE DATALOADERS ---------------------------------------------------------------

# DATA SET SPLITTING AND SORTING ----------------------------------------------------------------
train_subset, val_subset, test_subset = random_split(files, TRAIN_VAL_TEST)

# DATALOADER SETTINGS ------------------------------------------------------------------
dataloader_settings = {
    'batch_size': 1,                    # see *Note above
    'shuffle': True,                    # shuffle the batches before each epoch
    'collate_fn': collate_fn_PINN,      # include optional arguments
    'num_workers': 4,                   # number of workers
    'pin_memory': False if DEVICE.type == 'cpu' else True}

# PREPARE TRAIN, VAL & TEST DATALOADERS  ------------------------------------------------------------
train_subset, train_dataset, train_dataset_batches, train_loader = prepare_dataloader_PINN(train_subset, indices_by_length, \
    BATCH_SIZE, INPUT_COLUMNS, TARGET_COLUMN, PRIOR_COLUMN, scaler, target_scaler, prior_scaler, dataloader_settings, fit=True, drop_last=False)

val_subset, val_dataset, val_dataset_batches, val_loader = prepare_dataloader_PINN(val_subset, indices_by_length, \
    BATCH_SIZE, INPUT_COLUMNS, TARGET_COLUMN, PRIOR_COLUMN, scaler, target_scaler, prior_scaler, dataloader_settings)

test_subset, test_dataset, test_dataset_batches, test_loader = prepare_dataloader_PINN(test_subset, indices_by_length, \
    BATCH_SIZE, INPUT_COLUMNS, TARGET_COLUMN, PRIOR_COLUMN, scaler, target_scaler, prior_scaler, dataloader_settings)

# print dataset info
subset_files = print_dataset_sizes(train_dataset, val_dataset, test_dataset, train_subset, val_subset, test_subset, files)
# -

if IS_NOTEBOOK and False: 
    check_batch_PINN(train_loader)
    visualize_padding(BATCH_SIZE, trip_lengths, sorted_trip_lengths, train_loader, val_loader, test_loader)

# +
# TRAINING CONFIGURATION -----------------------------------------------------------------------

# OPTIMIZER --------------------------------------------------------------------------------
# common optimizers: ['torch.optim.Adam', 'torch.optim.SGD', 'torch.optim.RMSprop']
if 'OPTIMIZER' in globals(): optimizer = eval(OPTIMIZER)
else: optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE,
        weight_decay = 1e-4      # weight decay coefficient (default: 1e-2)
        #betas = (0.9, 0.95),    # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        #eps = 1e-8,             # term added to the denominator to improve numerical stability (default: 1e-8)
)
print(f"{'-'*60}\n{optimizer}\n{'-'*60}")

# LR SCHEDULER ----------------------------------------------------------------------------
if 'LRSCHEDULER' in globals(): scheduler = eval(LRSCHEDULER)
else: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 2, factor = 0.5, min_lr = 1e-6)
print(f"LRScheduler: {scheduler.__class__}\n{'-'*60}")


# +
# LOSS FUNCTION ----------------------------------------------------------------
def loss_fn(output, target):
    if 'LOSS_FN' in globals(): loss = eval(LOSS_FN)
    else: loss = F.mse_loss(output, target) # mean-squared error for regression
    return loss

def loss_fn_PINN_1(output, target, prior):
    loss = F.mse_loss(output, target) # mean-squared error for regression
    return loss

# or define criterion function:
criterion_list = [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss(), nn.HuberLoss(), MASE()]

if 'CRITERION' in globals(): criterion = eval(CRITERION)
else: criterion = nn.SmoothL1Loss()
print(f"Loss_Fn: {LOSS_FN}\n{'-'*60}\n", f"Criterion: {criterion.__class__}\n{'-'*60}\n")


# +
def loss_fn_PINN_2(output, target, prior):
    y_pred = output
    y_true = target
    y_phys = prior

    mse_loss = F.mse_loss(y_pred, y_true)
    phys_loss = F.mse_loss(y_pred, y_phys)

    total_loss = mse_loss + phys_loss
    return total_loss

def loss_fn_PINN_3(output, target, prior):
    l_p = 0.5

    y_pred = output
    y_true = target
    y_phys = prior

    total_loss = F.mse_loss(y_true, (l_p * y_phys + (1 - l_p) * y_pred))
    
    return total_loss


# -

# SET NETWORK TRAINER -----------------------------------------------------------------
TRAINER = PTrainer_PINN(
    model = model, 
    optimizer = optimizer,
    scheduler = scheduler, 
    loss_fn = loss_fn_PINN_3, 
    train_loader = train_loader,
    val_loader = val_loader,
    test_loader = test_loader,
    num_epochs = NUM_EPOCHS, 
    device = DEVICE, 
    is_notebook = IS_NOTEBOOK,
    use_mixed_precision = False,
    log_file = TRAIN_LOG)

# START TRAINING -----------------------------------------------------------------
if MODE == 'train_mode': CHECKPOINT = TRAINER.train_model()

# Example usage
CHECKPOINT, model_destination_path = save_checkpoint(TRAINER, train_loader, val_loader, test_loader, CHECKPOINT, CONFIG, subset_files, pth_folder)

# ___

# +
# #%%skip
# LOAD MODEL -----------------------------------------------------------------
#model_destination_path = Path(pth_folder, "LSTM1_241211_200821.pth")
CHECKPOINT = load_checkpoint(model_destination_path, DEVICE)

# get model type:
for key in CHECKPOINT.keys(): globals()[key] = CHECKPOINT[key]

# load model and optimizer states
model.load_state_dict(model_state_dict)
optimizer.load_state_dict(optimizer_state_dict)

model.eval()  # set model to evaluation mode for inference
print(f"Model loaded from:\t{model_destination_path}\n{'-'*60}\nModel: {model.__class__.__name__}\tParameters on device: {next(model.parameters()).device}"
        f"\n{'-'*60}\nTrain/Batch size:\t{len(train_loader.dataset)} / {train_loader.batch_size}\n"
        f"Loss:\t\t\t{loss_fn}\nOptimizer:\t\t{optimizer.__class__.__name__}\nLR:\t\t\t"
        f"{optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{optimizer.param_groups[0]['weight_decay']}\n{'-'*60}\n", model)


# +
# PLOT TRAINING PERFORMANCE -----------------------------------------------------------------
# get DataFrame of training metrics:
keys = ['training_table', 'train_losses_per_iter', 'train_losses', 'val_losses', 'lr_history', 'train_batches']
training_table, train_losses_per_iter, train_losses, val_losses, lr_history, train_batches = (CHECKPOINT[key] for key in keys)
training_df = pd.DataFrame(training_table, columns=["Epoch", "Iteration", "Batch Loss", "Train Loss"])

# -------------------------------------
NUM_EPOCHS = CONFIG['NUM_EPOCHS']
plot_training_performance(training_df, train_losses_per_iter, train_losses, val_losses, lr_history, train_batches, NUM_EPOCHS)


# -

def concat_outputs_targets_priors(outputs, targets, priors, original_lengths) -> tuple:
    all_outputs, all_targets, all_priors, all_original_lengths = [], [], [], []
    for batch_outputs, batch_targets, batch_priors, batch_lengths in zip(outputs, targets, priors, original_lengths):
        all_outputs.extend(batch_outputs)
        all_targets.extend(batch_targets)
        all_priors.extend(batch_priors)
        all_original_lengths.extend(batch_lengths)
        return all_outputs, all_targets, all_priors, all_original_lengths


# +
# EVALUATION -----------------------------------------------------------------

# get file list of test subset
test_files = CHECKPOINT["test_files"]
# -------------------------------------
# evaluate model on test set
test_loss, outputs, targets, priors, original_lengths = TRAINER.evaluate_model()
# -------------------------------------
all_outputs, all_targets, all_priors, all_original_lengths = concat_outputs_targets_priors(outputs, targets, priors, original_lengths)

# Inverse-transform on all outputs and targets for evaluation
scaled_outputs = [target_scaler.inverse_transform(output_sequence.reshape(1, -1)).squeeze() for output_sequence in all_outputs]
scaled_targets = [target_scaler.inverse_transform(target_sequence.reshape(1, -1)).squeeze() for target_sequence in all_targets]

print(f"Test Loss:  {test_loss:.6f}")
print(f"RMSE: {root_mean_squared_error(np.concatenate(scaled_targets), np.concatenate(scaled_outputs)):.4f}")
print(f"Standard Deviation: {np.std(np.concatenate(scaled_targets) - np.concatenate(scaled_outputs)):.4f}")

# +
# PLOT RESULTS -----------------------------------------------------------------
# get random sample sequence from test set
# -------------------------------------
sample_int = random.randint(1, len(scaled_outputs))
y_pred = scaled_outputs[sample_int]
y_true = scaled_targets[sample_int]

# -------------------------------------
def plot_prediction(y_true, y_pred, plot_active=True):
     if plot_active:
          plt.figure(figsize=(18,4))
          plt.xlabel('Time in s')
          plt.ylabel('SOC Change Rate in %/s')
          plt.title('Battery State of Charge: Prediction vs. Actual Data')
          plt.plot(y_true, label='Actual Data')  # actual plot
          plt.plot(np.arange(0, len(y_true), 1), y_pred, label='Predicted Data')  # predicted plot
          plt.legend()
          plt.text(0.01, 0.02, f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}\nStd Dev: {np.std(y_true - y_pred):.4f}",
          transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

          plt.figure(figsize=(18,4))
          plt.xlabel('Time in s')
          plt.ylabel('SOC in %')
          plt.plot(savgol_filter(y_true.flatten(), window_length=60, polyorder=3), label='Actual Data (Smoothed)')  # actual plot
          plt.plot(np.arange(0, len(y_true), 1), savgol_filter(y_pred.flatten(), window_length=60, polyorder=3), label='Predicted Data (Smoothed)')  # predicted plot
          plt.legend()

          
plot_prediction(y_true, y_pred, PLOT_ACTIVE)
# -

plt.figure(figsize=(18,4))
plt.plot(np.cumsum(y_pred), label='SOC Predicted')
plt.plot(np.cumsum(y_true), label='Actual SOC')

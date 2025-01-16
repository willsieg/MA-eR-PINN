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

# #%reset -f -s
# %matplotlib inline
'''------------------------------------------------------------------
MA-eR-PINN: eRange Prediction using Physics-Informed Neural Networks
---------------------------------------------------------------------
Version: V2.0      Modified: 12.01.2025        William Siegle
---------------------------------------------------------------------
PTRAIN - Standard Pipeline Framework for Training the PINN
+ OPTUNA - Hyperparameter Optimization using Optuna
------------------------------------------------------------------''';

# MA-eR-PINN: CONFIGURATION FILE -------------------------------------------------
from pathlib import Path
CONFIG = {
    # SYSTEM: ---------------------------------------------------------------------
    "GPU_SELECT":       0,
    "ROOT":             Path('../..').resolve(),
    "INPUT_LOCATION":   Path("TripSequences", "trips_processed_pinn_2"), 
    "OUTPUT_LOCATION":  Path("src", "models", "pth"),
    "SEED"  :           19,
    "MIXED_PRECISION":  True,

    # DATA PREPROCESSING: ---------------------------------------------------------
    "TRAIN_VAL_TEST":   [0.8, 0.1, 0.1], # [train, val, test splits]
    "MAX_FILES":        None, # None: all files
    "MIN_SEQ_LENGTH":   1800, # minimum sequence length in s to be included in DataSets
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
    "HIDDEN_SIZE":      12,    # features in the hidden state h
    "NUM_LAYERS":       3,      # recurrent layers for stacked LSTMs. Default: 1
    "DROPOUT":          0.35,   # usually: [0.2 - 0.5]
    
    # TRAINING & OPTIMIZER: --------------------------------------------------------
    "NUM_EPOCHS":       10,
    "BATCH_SIZE":       128,        # [2, 4, 8, 16, 32, 64, 128, 256]
    "LEARNING_RATE":    1e-3,       # 0.001 lr
    "WEIGHT_DECAY":     1e-3,       # weight decay coefficient (default: 1e-2)
    "MOMENTUM_SGD":     0.1,        # (default: 0.0)
    "OPTIMIZER":        'adamw',    # ('adam', 'sgd', 'adamw')
    "WEIGHT_INIT_TYPE": 'he',  # ('he', 'normal', 'default')
    "CLIP_GRAD":        2.0,        # default: None
    "LRSCHEDULER":      "torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)",  # constant LR for 1.0 as multiplicative factor
                        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 3, factor = 0.5, min_lr = 1e-7)

    # LOSS FUNCTION: ---------------------------------------------------------------
    "CRITERION":        "nn.SmoothL1Loss()", # ['nn.MSELoss()', 'nn.L1Loss()', 'nn.SmoothL1Loss()', 'nn.HuberLoss()', 'MASE()']
    "P_LOSS_FACTOR":    0.0, # Physics loss factor
}

# +
# OPTUNA: SEARCH SPACE ---------------------------------------------------
N_TRIALS = 100

global search_space, search_space_NewData
search_space = {
    # MODEL: -----------------------------------------------------------------------
    'HIDDEN_SIZE': ('int', 10, 100, 10),
    'NUM_LAYERS': ('int', 1, 10, 1),
    'DROPOUT': ('float', 0.0, 0.6, 0.05),
    'CLIP_GRAD': ('categorical', (None, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0)),
    'WEIGHT_INIT_TYPE': ('categorical', ('he', 'normal', 'default')),

    # TRAINING & OPTIMIZER: --------------------------------------------------------
    'OPTIMIZER': ('categorical', ('adam', 'adamw','sgd')),
    'NUM_EPOCHS': ('int', 5, 20, 1),
    'LEARNING_RATE': ('categorical', (5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 3e-3, 5e-3, 8e-3, 1e-2, 2e-2, 5e-2)),
    'WEIGHT_DECAY': ('categorical', (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
    'MOMENTUM_SGD': ('float', 0.0, 0.9, 0.1),

    #'P_LOSS_FACTOR': ('float', 0.05, 1.0, 0.05)
}

search_space_NewData = {
    # DATA PREPROCESSING: ---------------------------------------------------------
    "MIN_SEQ_LENGTH": ('int', 300, 3600, 300),
    'BATCH_SIZE': ('categorical', (4, 8, 16, 32, 64, 128)),
}


# +
# LOSS FUNCTION MODULES ----------------------------------------------------------------   

def loss_fn_PINN_3(output, target, prior):
    l_p = P_LOSS_FACTOR
    y_pred = output; y_true = target; y_phys = prior
    total_loss = F.mse_loss(y_true, (l_p * y_phys + (1 - l_p) * y_pred), reduction='mean')
    return total_loss

global LOSS_FN
LOSS_FN = loss_fn_PINN_3

# -

# ___
# SETUP: Locate devices & system folders

# +
# LOCATE REPOSITORY/DATASTORAGE IN CURRENT SYSTEM ENVIRONMENT  ---------------------------
import sys, os
for key in CONFIG: globals()[key] = CONFIG[key]
print(ROOT)
if 'ROOT' not in globals(): ROOT = Path('../..').resolve()
sys.path.append(os.path.abspath(ROOT))

# INTERNAL MODULE IMPORTS ----------------------------------------------------------------
from src.__init__ import *
from src.utils.data_utils import *
from src.utils.preprocess_utils import *
from src.utils.eval_utils import *
from src.utils.Trainers import *
from src.models.lstm_models import *

# SETUP ENVIRONMENT ---------------------------------------------------------------------
DATA_PATH, IS_NOTEBOOK, DEVICE, LOG_FILE_NAME, TS = setup_environment(CONFIG, ROOT, SEED, GPU_SELECT)
if not IS_NOTEBOOK: output_file = open(f"{LOG_FILE_NAME}", "w"); sys.stdout = Tee(sys.stdout, output_file); sys.stderr = Tee(sys.stderr, output_file)

# FILE SOURCES ---------------------------------------------------------------
input_folder = Path(DATA_PATH, INPUT_LOCATION) # Trip parquet files
pth_folder = Path(ROOT, OUTPUT_LOCATION, f"{TS}")
pth_folder.mkdir(parents=True, exist_ok=True)
files, trip_lengths, indices_by_length, sorted_trip_lengths, all_signals = prepare_data(input_folder, pth_folder, MAX_FILES, MIN_SEQ_LENGTH, ROOT)
# -

# ___
# DATA SELECTION & PREPROCESSING

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

        self.input_size = input_size    # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers    # number of layers
        self.dropout = dropout

        # LSTM CELL --------------------------------
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True,dropout=self.dropout,device=device)

        # LAYERS -----------------------------------
        self.dropout_layer = nn.Dropout(self.dropout)
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

    # Define the weight initialization function for LSTM
    def initialize_weights_lstm(self, init_type):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                if init_type == 'he': nn.init.kaiming_uniform_(param.data, nonlinearity='relu')     # HE INIT
                elif init_type == 'normal': nn.init.normal_(param.data, mean=0.0, std=0.02)         # NORMAL INIT
                elif init_type == 'default': continue                                               # TORCH DEFAULT INIT
            elif 'bias' in name and init_type != 'default': nn.init.constant_(param.data, 0)


# -

# ___
# OPTUNA: Hyperparameter Optimization

# OPTUNA: OBJECTIVE ---------------------------------------------------
def objective(trial, LOSS_FN):

    # OPTUNA: CREATE TRIAL OBJECTS ---------------------------------------------------
    optuna_params = {}
    for param, (suggest_type, *args) in search_space.items():
        if suggest_type == 'int': optuna_params[param.lower()] = trial.suggest_int(param, *args[:2], step=args[2])
        elif suggest_type == 'float': optuna_params[param.lower()] = trial.suggest_float(param, *args[:2], step=args[2])
        elif suggest_type == 'categorical': optuna_params[param.lower()] = trial.suggest_categorical(param, *args)
    
    # Update CONFIG with suggested hyperparameters
    for param in optuna_params.keys(): CONFIG[param.upper()] = optuna_params[param]
    for key in CONFIG: globals()[key] = CONFIG[key]
    # -----------------------------------------------------------------------------------

    # TRAINING_CODE: -----------------------------------------------------------------
    # INSTANTIATE MODEL AND APPLY WEIGHT INITIALIZATION --------------------
    model = LSTM1_packed_old_version(len(INPUT_COLUMNS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model.initialize_weights_lstm(WEIGHT_INIT_TYPE)
    #print_info(model)
    
    # SET OPTIMIIZER, SCHEDULER AND LOSS MODULES ---------------------------
    if OPTIMIZER=='adam': optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER=='adamw': optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    elif OPTIMIZER=='sgd': optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch): return 1.0
    scheduler = eval(LRSCHEDULER); criterion = eval(CRITERION)

    # TRAIN -----------------------------------------------------------------
    TRAINER = PTrainer_PINN(
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler,
        loss_fn = LOSS_FN, 
        train_loader = train_loader, 
        val_loader = val_loader, 
        test_loader = test_loader, 
        num_epochs = NUM_EPOCHS, 
        device = DEVICE, 
        is_notebook = IS_NOTEBOOK, 
        use_mixed_precision = MIXED_PRECISION, 
        clip_value = CLIP_GRAD, 
        log_file = Path(LOG_FILE_NAME).with_name(f"{TS}_Trial{trial.number}.txt"))

    RESULTS = TRAINER.train_model()

    # PLOT RESULTS AND SAVE OPERATIONS ----------------------------------------------
    print(f"Trial {trial.number}")
    plot_training_performance(RESULTS)
    
    # TRAIN/VAL Loss:
    val_loss = RESULTS['val_losses'][-1]
    train_loss = RESULTS['train_losses'][-1]

    # EVALUATE -----------------------------------------------------------------
    test_loss, all_outputs, all_targets, all_priors, all_original_lengths = TRAINER.evaluate_model()
    # back-transform:
    scaled_outputs = [target_scaler.inverse_transform(output_sequence.reshape(1, -1)).squeeze() for output_sequence in all_outputs]
    scaled_targets = [target_scaler.inverse_transform(target_sequence.reshape(1, -1)).squeeze() for target_sequence in all_targets]
    scaled_priors = [prior_scaler.inverse_transform(prior_sequence.reshape(1, -1)).squeeze() for prior_sequence in all_priors]
    # concatenate:
    all_y_true, all_y_pred, all_y_phys = np.concatenate(scaled_targets), np.concatenate(scaled_outputs), np.concatenate(scaled_priors)
    # calculate evaluation metrics
    print(f"Test Loss:\t\t{test_loss:.6f}")
    metrics = calculate_metrics(all_y_true, all_y_pred) # [rmse, mae, std_dev, mape, r2, max_error]

    rmse = metrics["rmse"]; trial.set_user_attr("rmse", rmse)
    mae = metrics["mae"]; trial.set_user_attr("mae", metrics["mae"])
    std = metrics["std_dev"]; trial.set_user_attr("std_dev", metrics["std_dev"])
    mape = metrics["mape"]; trial.set_user_attr("mape", metrics["mape"])

    return rmse


# OPTUNA: STUDY -------------------------------------------------------------------
study = optuna.create_study(direction='minimize', sampler = optuna.samplers.RandomSampler())    # TPESampler, RandomSampler, GridSampler, CmaEsSampler, NSGAIISampler
study.optimize(lambda trial: objective(trial, LOSS_FN), n_trials=N_TRIALS)

for trial in study.trials:
    print(f"Trial {trial.number}: Value: {trial.value}, RMSE: {trial.user_attrs['rmse']}, MAE: {trial.user_attrs['mae']}, STD_dev: {trial.user_attrs['std_dev']}, MAPE: {trial.user_attrs['mape']}")

# ___
# SAVE CHECKPOINT

# Save the best trial config to a text file
best_trial = study.best_trial
best_trial_dict = {'params': best_trial.params, 'value': best_trial.value}
with open(Path(LOG_FILE_NAME).with_name(f"{TS}_BEST_IS_{best_trial.number}.txt"), 'w') as f: json.dump(best_trial_dict, f, indent=4)
print("Best Trial: ", best_trial.number); print("Best hyperparameters: ", study.best_params)
trials_df = study.trials_dataframe()
print(tabulate(trials_df, headers='keys', tablefmt='psql'))
trials_df.to_csv(Path(LOG_FILE_NAME).with_name(f"trials_overview.csv"), index=False)

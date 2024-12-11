from pathlib import Path, WindowsPath, PosixPath

# GENERAL MODULE IMPORTS -----------------------------------------------------------------
import math, time, random, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import pyarrow.parquet as pq
from copy import deepcopy
from datetime import datetime
from tabulate import tabulate

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from scipy.signal import savgol_filter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
torch.set_default_dtype(torch.float32); torch.set_printoptions(precision=6, sci_mode=True)

import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

from torchmetrics.functional import mean_squared_error
from pytorch_forecasting.metrics import MASE

# SETUP ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def setup_environment(CONFIG, ROOT, SEED, GPU_SELECT):
    global DATA_PATH, IS_NOTEBOOK, DEVICE

    import sys, os
    print(f"{'-'*60}\nDirectories:\n  {ROOT}:\t\t\t{', '.join([_.name for _ in ROOT.glob('*/')])}")
    sys.path.append(os.path.abspath(ROOT))

    # Set DATA_PATH
    if 'DATA_PATH' not in globals():
        from data import get_data_path  # paths set in "data/__init__.py"
        DATA_PATH = get_data_path()
    print(f"  {DATA_PATH}:\t\t\t{', '.join([_.name for _ in DATA_PATH.glob('*/')])}")

    # NOTEBOOK / SCRIPT SETTINGS -------------------------------------------------------------
    IS_NOTEBOOK = False
    try:    # if running in IPython
        shell = get_ipython().__class__.__name__ # type: ignore 
        from IPython.display import display, HTML, Javascript, clear_output
        from IPython.core.magic import register_cell_magic
        @register_cell_magic    # cells can be skipped by using '%%skip' in the first line
        def skip(line, cell): return
        from tqdm.notebook import tqdm as tqdm_nb
        IS_NOTEBOOK = True
        print(f"{'-'*60}\nRunning in notebook mode")
    except (NameError, ImportError):    # if running in script
        from tqdm import tqdm as tqdm
        print(f"{'-'*60}\nRunning in script mode")

    config_df = pd.DataFrame(list(CONFIG.items()), columns=['Parameter', 'Value'])
    config_df['Value'] = config_df['Value'].apply(lambda x: str(x).replace(',', ',\n') if len(str(x)) > 120 else str(x))
    print(f"CONFIG Dictionary:\n{'-'*129}\n", tabulate(config_df, headers='keys', colalign=("left", "left"), \
        maxcolwidths=[30, 120]), f"\n{'-'*129}\n")

    # DEVICE SELECTION ---------------------------------------------------------------------
    print(f"Torch version: ", torch.__version__)
    if not torch.cuda.is_available() or GPU_SELECT is None:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{GPU_SELECT}")
    print(f"Using: -->  {str(DEVICE).upper()}")

    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

    return DATA_PATH, IS_NOTEBOOK, DEVICE
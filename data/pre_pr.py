import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pickle
from datetime import datetime
from pathlib import Path
from scipy.signal import savgol_filter
#from Physics_Model.VehModel import CreateVehicle

'''
################################################################################################
# Specify Data Locations:
parquet_folder = '/home/sieglew/data/processed'                 # Volts Database
new_parquet_folder = '/home/sieglew/data/processed_2'           # same, but with modified time series data

volts_stats = '/home/sieglew/data/Volts.pickle'                 # list of volts Data files (from Parquet_Stats.ipynb)
pickle_destination_folder = '/home/sieglew/data/TripFiles'      # Trip pickles for Vehicle Model

y_true_folder = '/home/sieglew/data/y_true'                     # Energy Consumption Time Series Data
################################################################################################
'''
print(Path('.').resolve())
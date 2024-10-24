import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pickle
from datetime import datetime
from pathlib import Path
from scipy.signal import savgol_filter

# ------------ LOCATE REPOSITORY/DATASTORAGE IN CURRENT SYSTEM ENVIRONMENT  --------------
import sys, os; from pathlib import Path                                                #|
global ROOT, DATA_PATH, IS_NOTEBOOK; IS_NOTEBOOK = False                                #|
ROOT = Path('..', '..').resolve() if IS_NOTEBOOK else Path('..').resolve()    
sys.path.append(os.path.abspath(ROOT))                                                  #|
from data import get_data_path  # paths set in "data/__init__.py"                       #|
DATA_PATH = get_data_path()                                                             #|
print(f"{'-'*60}\n{DATA_PATH}:\t\t{', '.join([_.name for _ in DATA_PATH.glob('*/')])}") #|
print(f"{ROOT}:\t{', '.join([_.name for _ in ROOT.glob('*/')])}")   	                #|
# ----------------------------------------------------------------------------------------

# relative Imports: ---------------------------------------------------------------------------------------------------
from src.physics_model.VehModel import CreateVehicle
# -----------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# Specify Data Locations:
# FILE SOURCES ---------------------------------------------------------------
parquet_folder = Path(DATA_PATH, "processed") # Trip parquet files
parquet_folder_resampled = Path(DATA_PATH, "processed_resampled") # Trip parquet filesn for resampled time series
save_model_folder = Path(ROOT, "src", "models", "pth")

# ------------------------------------------------------------------------------------------------------------------------------
# OUTPUT LOCATIONS ---------------------------------------------------------------
new_parquet_folder = Path(DATA_PATH, "processed_new") # Trip parquet files
pickle_destination_folder = Path(DATA_PATH, "TripFiles") # Trip parquet files

'''
# ------------------------------------------------------------------------------------------------------------------------------
volts_stats = Path(ROOT, "data", "Volts.pickle")
# import database statistics and complete list of files:
try:
    with open(volts_stats, 'rb') as handle:
        _, all_trips_soc, trips_sizes, trip_by_vehicle = pickle.load(handle)
except:
    pass
'''

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
all_files = [os.path.basename(Path(parquet_folder, f)) for f in os.listdir(parquet_folder) if f.endswith(".parquet")]

# loop through every file:
for f in all_files:
    print(f"Reading File: {f}")

    vehicle_id = f[8:10].strip("_t")
    file_code = f[7:-8]

    df = pd.read_parquet(Path(parquet_folder, f), engine='fastparquet')
    df.sort_index(axis=1, inplace=True)

    ############################################################################################
    #pd.DataFrame(df.hv_batmomavldischrgen_cval_1).to_parquet(f'{y_true_folder}/{file_code}.parquet')

    signal_only_nans = list(df.columns[np.array(df.isnull().all())])
    signal_with_nans = list(df.columns[np.array(df.isnull().any())])

    # Remove GPS interruptions
    ###############################################
    # identify simultaneous 0 values in all GPS related Signals (Lat/Long and Altitude)
    gps_drop = np.where((df[['latitude_cval_ippc','longitude_cval_ippc']].values < 0.1).all(1))[0]

    # also add neighboring time steps:
    gps_drop = list(set(list(gps_drop + 1) + list(gps_drop - 1) + list(gps_drop)))
    gps_drop = [i for i in gps_drop if (i>0 and i<len(df))]

    if len(gps_drop) > 0:
        df.loc[gps_drop,['latitude_cval_ippc','longitude_cval_ippc','altitude_cval_ippc']]  = np.nan

    # Forward/Backward filling of signal gaps
    ###############################################
    for sig in signal_with_nans:
    # consider only signals that have less than 100 % NaNs:
        if df[sig].isnull().mean() < 1:
            try:
                df[sig] = df[sig].ffill().bfill()
            except:
                pass

    # extract constant values:
    constants = {}
    for col in df.columns:
        if len({x for x in df[col] if x==x}) == 1:      # number of distinct non-NaN-values
            constants[col] = df[col].iloc[df[col].first_valid_index()] 

    # TIME:
    ###############################################
    # correct timestamp if necessary:
    if max(df.signal_time).year < 2000:
        df.signal_time = pd.to_datetime(df.signal_ts * (10**3))

    time = pd.DataFrame(np.array((df.signal_time - df.signal_time[0]).dt.seconds))    # TIME since trip Start [s] --> corresponding to time series indices
    time_unix = df.signal_time  # UNIX TIME (starting 1970) [s]

    # RESAMPLING: --> separate directory
    ###############################################
    df.set_index('signal_time', inplace=True)
    df_resampled = df.resample('10s').mean() # RESAMPLE AT 10 seconds intervals
    df_resampled.reset_index(inplace=True)  # Reset the index if you want 'signal_time' to be a column again
    #df_resampled.fillna(df.median(), inplace=True)  # Basic Median Filling

    # VEHICLE MOTION:
    ###############################################
    dist = ((df.hirestotalvehdist_cval_icuc - df.hirestotalvehdist_cval_icuc.iloc[0]) * 1000).round(3)   # MILEAGE since trip Start [m]
    speed = df.vehspd_cval_cpc/3.6                                  # VEHICLE SPEED [m/s]
    accel = pd.DataFrame(np.diff(speed, prepend = speed.iloc[0]))        # LONGITUDINAL VEHICLE ACCELERATION [m/s^2] 

    # ROUTE: 
    ###############################################
    gps_pos = pd.DataFrame(tuple(zip(df['latitude_cval_ippc'], df['longitude_cval_ippc'])))  
    alt = df.altitude_cval_ippc                                           # ALTITUDE [m]  
    road_grad = pd.DataFrame(df.bs_roadincln_cval)                        # ROAD GRADIENT [%] = [tan(alpha) * 100]   
    amb_temp = pd.DataFrame(savgol_filter(df.ambtemp_cval_pt, 100, 3))    # ambient temperature [°C] [smoothed]

    # BATTERY:
    ###############################################
    soc = df.hv_bat_soc_cval_bms1/100                  # Battery State of Charge [-]

    if "hv_bat_soh_cval_bms1" in constants.keys():
        soh = constants["hv_bat_soh_cval_bms1"]/100
    else:
        soh = np.mean(df.hv_bat_soh_cval_bms1)/100          # Battery State of Health [-] (constant) 

    bat_pwr = df.hv_batpwr_cval_bms1                # Battery Power [kW]  = df.hv_bat_dc_momvolt_cval_bms1  *  df.hv_batcurr_cval_bms1  = U*I
    bat_mom_en = df.hv_batmomavldischrgen_cval_1    # Momentary Available discharge energy [kWh]
    bat_cap_total = np.mean(bat_mom_en + df.hv_batmomavlchrgen_cval_bms1)  # Total battery capacity (constant) [kWh]
    # calculated by mean of sum of momentary available charge and discharge energies

    # MOTOR:
    ###############################################
    mot_1_speed = df.actualspeed_pti1               # Motor 1 Speed [rpm]
    mot_1_torque = df.actualtorque_pti1             # Motor 1 Torque [Nm] (or [%] ?)

    mot_2_speed = df.actualspeed_pti1               # Motor 2 Speed [rpm]
    mot_2_torque = df.actualtorque_pti1             # Motor 2 Torque [Nm] (or [%] ?)

    mot_pwr = df.emot_pwr_cval                      # Electrical power of motors (combined) [kW]

    # VEHICLE:
    ###############################################
    V = CreateVehicle(vehicle_id)        # import vehicle parameters

    vehweight = df.vehweight_cval_pt*1000  # weight (t) (PT) [kg]
    grocmb = df.grocmbvehweight_cval       # gross combination weight (t) [kg]

    try:
        weight_est_mean = sum(np.multiply(grocmb,dist))/sum(dist)       # approximated vehicle weight [kg]
    # if 'grocombvehweight' is empty:
    except:
        weight_est_mean = np.mean(vehweight)

    # COLLECTED TIME SERIES DATA 
    ############################################################################################
    T = pd.concat([time, time_unix, dist, speed, accel, gps_pos, alt, road_grad, amb_temp,soc,bat_pwr,bat_mom_en, 
                mot_1_speed,mot_1_torque,mot_2_speed,mot_2_torque,mot_pwr, vehweight, grocmb], axis = 1)

    T.columns = ['t',                      # Time since Trip Start ,[s]
                'date',                    # Date-time stamp
                'dist',                    # Distance since Start, [m]
                'speed',                   # Speed, [m/s]
                'accel',                   # Acceleration, [m/s²]
                'lat','long',              # GPS-Coordinates (Latitude, Longitude)
                'alt',                     # Altitude, [m]
                'road_grad',               # Road Slope, [%]
                'amb_temp',                # Ambient Air Temperature, [°C]
                'soc',                     # Battery SOC, [-]
                'bat_pwr',                 # Battery Power, [kW]
                'bat_mom_en',              # Momentary Available discharge energy [kWh]
                'Mot1_speed',              # Motor 1 Speed, [rpm]
                'Mot1_torque',             # Motor 1 Torque, [Nm]
                'Mot2_speed',              # Motor 2 Speed, [rpm]
                'Mot1_torque',             # Motor 2 Torque, [Nm]
                'Mot_pwr',                 # Total Motor Power, [kW]
                'vehweight',               # Vehicle Weight (PT), [kg]
                'grocmb'                   # Vehicle Weight (GroCmb), [kg]
                ]      

    # PARAMETERS EXTRACTED FROM DATA 
    ############################################################################################
    C = {}
    C = dict((k,eval(k)) for k in ["weight_est_mean","soh","bat_cap_total"])
    C = {**C,**constants}         

    # ORIGINAL TIME SERIES DATA (Preprocessed and reduced)
    ############################################################################################
    #df.drop(columns = signal_only_nans + ['diff','vehicle_id'], inplace=True)

    # Save as pickle file in destination folder
    ############################################################################################
    #with open(f'{pickle_destination_folder}/{file_code}.pickle', 'wb') as handle:
    #    pickle.dump([T,C,V.prm], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #    print(f'{file_code}.pickle saved')

    ############################################################################################
    #df.to_parquet(f'{new_parquet_folder}/{file_code}.parquet')
    #df_resampled.to_parquet(f'{parquet_folder_resampled}/{file_code}.parquet')

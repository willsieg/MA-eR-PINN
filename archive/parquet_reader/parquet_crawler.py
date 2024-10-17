# %%
import pandas as pd

df = pd.read_parquet("./data/processed/v_id983V4_trip14.parquet")
df.sort_index(axis=1, inplace=True)
df.info()

# %%
signal_names = list(df.columns)
signal_names.sort()
for c in signal_names:
    print(c)


# %%
selected_columns = [
    "hv_bat_dc_volt_cval_bms1",
    "hv_curr_cval_dcl1",
    "airtempoutsd_cval_cpc",
    "accelpdlposn_cval",
    "hv_ptc_cabin1_pwr_cval",
    "hv_batcurr_cval_bms1",
    "bs_brk_cval",
    "brktempra_cval",
    "hv_batmaxdischrgpwrlim_cval_1",
    "hv_batmaxchrgpwrlim_cval_1",
    "hv_batmomavlchrgen_cval_bms1",
    "hv_batmomavldischrgen_cval_1",
    "hv_chrgpwr_ecpc_cval",
    "hv_bat_dc_momvolt_cval_bms1",
    "hv_pwr_cval_dcl1",
    "txoiltemp_cval_tcm",
    "oiltemp_ra_cval",
    "hv_bat_soh_cval_bms1",
    "meanmoduletemperature_bms01",
    "brktempfa_cval",
    "hv_bat_soc_cval_bms1",
    "hv_batpwr_cval_bms1",
    "actualspeed_pti1",
    "actualspeed_pti2",
    "vehspd_cval_cpc",
    "rmsmotorcurrent_pti2",
    "rmsmotorcurrent_pti1",
    "actualdcvoltage_pti1",
    "powerstagetemperature_pti1",
    "edrvspd_cval",
    "motortemperature_pti1",
    "actualtorque_pti1",
    "actdrvtrnpwrprc_cval",
    "epto_pwr_cval",
    "cc_setspd_cval",
    "maxrecuppwrprc_cval"
]
selected_columns.sort()
for x in selected_columns:
    print(x)


# %%
weights = df[["grocmbvehweight_cval" ,"vehweight_cval_pt","vehweight_stat_pt"]]
# %%

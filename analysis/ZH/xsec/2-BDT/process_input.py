import importlib, time
import pandas as pd

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, train_vars, mode_names, final_state, miss_BDT

from tools.utils import get_procDict, update_keys, get_data_paths, BDT_input_numbers
from tools.utils import counts_and_efficiencies, save_to_pickle
from tools.utils import additional_info, df_split_data


procFile = "FCCee_procDict_winter2023_training_IDEA.json"
proc_dict = get_procDict(procFile)
procDict = update_keys(proc_dict, mode_names)

xsec = {}
for key, value in procDict.items(): 
    if key in mode_names:      
        xsec[key] = value["crossSection"]

sig = f"{final_state}H"
data_path = loc.MVA_INPUTS
pkl_path  = loc.MVA_PROCESSED

files, df, eff = {}, {}, {}
N_events, vars_list = {}, train_vars.copy()

frac = {
    f"{final_state}H": 1.0, f"WW{final_state}": 1.0,
    "ZZ": 1.0, "Zll": 1.0, "egamma": 1.0, "gammae": 1.0,
    f"gaga_{final_state}": 1.0
}

for cur_mode in mode_names:
    files[cur_mode] = get_data_paths(cur_mode, data_path, mode_names)
    N_events[cur_mode], df[cur_mode], eff[cur_mode] = counts_and_efficiencies(cur_mode, files[cur_mode], vars_list)
    print(f"Number of events in {cur_mode} = {N_events[cur_mode]}")
    print(f"Efficiency of {cur_mode} = {eff[cur_mode]*100:.3f}%")
    df[cur_mode] = additional_info(df[cur_mode], cur_mode, sig)

N_BDT_inputs = BDT_input_numbers(mode_names, sig, df, eff, xsec, frac)
for cur_mode in mode_names:
    print(f"Number of BDT inputs for {cur_mode} = {N_BDT_inputs[cur_mode]}")

for cur_mode in mode_names:
    df[cur_mode] = df_split_data(df[cur_mode], N_BDT_inputs, xsec, N_events, cur_mode)

dfsum = pd.concat([df[cur_mode] for cur_mode in mode_names])

save_to_pickle(dfsum, pkl_path)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
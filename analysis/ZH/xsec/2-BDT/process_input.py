import importlib, time, argparse
import pandas as pd

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)

from tools.utils import get_procDict, update_keys, get_data_paths, BDT_input_numbers
from tools.utils import counts_and_efficiencies, save_to_pickle
from tools.utils import additional_info, df_split_data

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, train_vars

cat, ecm = arg.cat, arg.ecm

# Decay modes used in first stage training and their respective file names
ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if cat=='ee' else f"wzp6_ee_mumu_ecm{ecm}"
modes = {f"{cat}H":       f"wzp6_ee_{cat}H_ecm{ecm}",
         f"ZZ":                   f"p8_ee_ZZ_ecm{ecm}", 
         f'Z{cat}':       ee_ll,
         f"WW{cat}":      f"p8_ee_WW_{cat}_ecm{ecm}",
         f"gammae_{cat}": f"wzp6_gammae_eZ_Z{cat}_ecm{ecm}",
         f"egamma_{cat}": f"wzp6_egamma_eZ_Z{cat}_ecm{ecm}",
         f"gaga_{cat}":   f"wzp6_gaga_{cat}_60_ecm{ecm}"
}

proc_dict = get_procDict("FCCee_procDict_winter2023_training_IDEA.json")
procDict = update_keys(proc_dict, modes)

xsec = {}
for key, value in procDict.items(): 
    if key in modes: xsec[key] = value["crossSection"]

sig = f"{cat}H"
frac = {
    f"{cat}H": 1.0, f"WW{cat}": 1.0, "ZZ": 1.0, f"Z{cat}": 1.0, 
    f"egamma_{cat}": 1.0, f"gammae_{cat}": 1.0, f"gaga_{cat}": 1.0
}

selections = [
    'sel0'
]

for sel in selections:

    data_path = get_loc(loc.HIST_MVA,   cat, ecm, '')
    pkl_path  = get_loc(loc.MVA_INPUTS, cat, ecm, sel)

    files, df, eff = {}, {}, {}
    N_events, vars_list = {}, train_vars.copy()

    for cur_mode in modes:
        files[cur_mode] = get_data_paths(cur_mode, data_path, modes, f'_{sel}')
        N_events[cur_mode], df[cur_mode], eff[cur_mode] = counts_and_efficiencies(files[cur_mode], vars_list)
        print(f"Number of events in {cur_mode} = {N_events[cur_mode]}")
        print(f"Efficiency of {cur_mode} = {eff[cur_mode]*100:.3f}%")
        df[cur_mode] = additional_info(df[cur_mode], cur_mode, sig)

    N_BDT_inputs = BDT_input_numbers(modes, sig, df, eff, xsec, frac)
    for cur_mode in modes:
        print(f"Number of BDT inputs for {cur_mode:{' '}{'<'}{10}} = {N_BDT_inputs[cur_mode]}")

    for cur_mode in modes:
        df[cur_mode] = df_split_data(df[cur_mode], N_BDT_inputs, eff, xsec, N_events, cur_mode)

    dfsum = pd.concat([df[cur_mode] for cur_mode in modes])

    save_to_pickle(dfsum, pkl_path)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')

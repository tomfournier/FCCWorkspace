import importlib, time, argparse
import pandas as pd

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)

parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument('--leading', help='Add the p_leading and p_subleading cuts', action='store_true')
parser.add_argument('--vis', help='Add E_vis > 10 GeV cut', action='store_true')
parser.add_argument('--visbdt', help='Add E_vis in the training variables for the BDT', action='store_true')
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
from userConfig import loc, get_loc, select, train_vars

final_state, ecm = arg.cat, arg.ecm
sel = select(arg.recoil120, arg.miss, arg.bdt, arg.leading, arg.vis, arg.visbdt)

# Decay modes used in first stage training and their respective file names
ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if final_state=='ee' else f"wzp6_ee_mumu_ecm{ecm}"
modes = {f"{final_state}H":       f"wzp6_ee_{final_state}H_ecm{ecm}",
         f"ZZ":                   f"p8_ee_ZZ_ecm{ecm}", 
         f'Z{final_state}':       ee_ll,
         f"WW{final_state}":      f"p8_ee_WW_{final_state}_ecm{ecm}",
         f"gammae_{final_state}": f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
         f"egamma_{final_state}": f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
         f"gaga_{final_state}":   f"wzp6_gaga_{final_state}_60_ecm{ecm}"
}

procFile = "FCCee_procDict_winter2023_training_IDEA.json"
proc_dict = get_procDict(procFile)
procDict = update_keys(proc_dict, modes)

xsec = {}
for key, value in procDict.items(): 
    if key in modes: xsec[key] = value["crossSection"]

sig = f"{final_state}H"
data_path = get_loc(loc.MVA_INPUTS,    final_state, ecm, sel)
pkl_path  = get_loc(loc.MVA_PROCESSED, final_state, ecm, sel)

files, df, eff = {}, {}, {}
N_events, vars_list = {}, train_vars.copy()
if arg.bdt: vars_list.append("cosTheta_miss")
if arg.visbdt: vars_list.append("visibleEnergy")

if not arg.miss and not arg.vis:
    frac = {
        f"{final_state}H": 1.0, f"WW{final_state}": 1.0, "ZZ": 1.0, f"Z{final_state}": 1.0, 
        f"egamma_{final_state}": 1.0, f"gammae_{final_state}": 1.0, f"gaga_{final_state}": 1.0
    }
elif arg.vis and final_state=='mumu':
    frac = {
        f"{final_state}H": 1.0, f"WW{final_state}": 0.95, "ZZ": 1.0, f"Z{final_state}": 1.0, 
        f"egamma_{final_state}": 1.0, f"gammae_{final_state}": 1.0, f"gaga_{final_state}": 0.95
    }
elif arg.miss and arg.leading and not arg.vis and not arg.recoil120:
        frac = {
        f"{final_state}H": 1.0, f"WW{final_state}": 0.34, "ZZ": 0.34, f"Z{final_state}": 0.34, 
        f"egamma_{final_state}": 0.34, f"gammae_{final_state}": 0.34, f"gaga_{final_state}": 0.34
    }
elif arg.miss and arg.leading and arg.recoil120:
        frac = {
        f"{final_state}H": 1.0, f"WW{final_state}": 0.18, "ZZ": 0.18, f"Z{final_state}": 0.18, 
        f"egamma_{final_state}": 0.18, f"gammae_{final_state}": 0.18, f"gaga_{final_state}": 0.18
    }
else:
    frac = {
        f"{final_state}H": 1.0, f"WW{final_state}": 0.4, "ZZ": 0.4, f"Z{final_state}": 0.4, 
        f"egamma_{final_state}": 0.4, f"gammae_{final_state}": 0.4, f"gaga_{final_state}": 0.4
    }

for cur_mode in modes:
    files[cur_mode] = get_data_paths(cur_mode, data_path, modes)
    N_events[cur_mode], df[cur_mode], eff[cur_mode] = counts_and_efficiencies(cur_mode, files[cur_mode], vars_list)
    print(f"Number of events in {cur_mode} = {N_events[cur_mode]}")
    print(f"Efficiency of {cur_mode} = {eff[cur_mode]*100:.3f}%")
    df[cur_mode] = additional_info(df[cur_mode], cur_mode, sig)

N_BDT_inputs = BDT_input_numbers(modes, sig, df, eff, xsec, frac)
for cur_mode in modes:
    print(f"Number of BDT inputs for {cur_mode:{' '}{'<'}{10}} = {N_BDT_inputs[cur_mode]}")

for cur_mode in modes:
    df[cur_mode] = df_split_data(df[cur_mode], N_BDT_inputs, xsec, N_events, cur_mode)

dfsum = pd.concat([df[cur_mode] for cur_mode in modes])

save_to_pickle(dfsum, pkl_path)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')

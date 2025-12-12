import time, argparse, importlib
import pandas as pd

t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, add_package_path, get_loc
add_package_path(loc.PACKAGE)

from package.config import timer, warning, vars

if arg.cat=='':
    warning(log_msg='Final state was not selected, please select one to run this script')

from package.tools.utils import get_paths, to_pkl
from package.tools.utils import get_procDict, update_keys

from package.func.bdt import counts_and_effs, additional_info
from package.func.bdt import BDT_input_numbers, df_split_data



cat, ecm = arg.cat, arg.ecm
inDir = get_loc(loc.HIST_MVA, cat, ecm, '')

sels = [
    'Baseline'
]

# Decay modes used in first stage training and their respective file names
modes = {
    f'Z{cat}H':       f'wzp6_ee_{cat}H_ecm{ecm}',
    f'ZZ':           f'p8_ee_ZZ_ecm{ecm}', 
    f'Z{cat}':       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' 
                     else f'wzp6_ee_mumu_ecm{ecm}',
    f'WW{cat}':      f'p8_ee_WW_{cat}_ecm{ecm}',
    f'gammae_{cat}': f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'egamma_{cat}': f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
    f'gaga_{cat}':   f'wzp6_gaga_{cat}_60_ecm{ecm}'
}

# Name of the dictionary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict_name = 'FCCee_procDict_winter2023_training_IDEA.json'



def run(inDir: str, 
        sels:  list[str], 
        modes: list[str], 
        vars:  list[str], 
        sig: str, 
        procDict_name: str
        ) -> None:

    proc_dict = get_procDict(procDict_name)
    procDict = update_keys(proc_dict, modes)

    xsec = {}
    for key, value in procDict.items(): 
        if key in modes: xsec[key] = value['crossSection']

    frac ={mode: 1.0 for mode in modes}

    for sel in sels:
        pkl_path  = get_loc(loc.MVA_INPUTS, cat, ecm, sel)

        files, df, eff, N_events = {}, {}, {}, {}

        for mode in modes:
            files[mode] = get_paths(mode, inDir, modes, f'_{sel}')
            df[mode], eff[mode], N_events[mode] = counts_and_effs(files[mode], vars, only_eff=False)
            print(f'Number of events in {mode} = {N_events[mode]}')
            print(f'Efficiency of {mode} = {eff[mode]*100:.3f}%')
            df[mode] = additional_info(df[mode], mode, sig)

        N_BDT_inputs = BDT_input_numbers(df, modes, sig, eff, xsec, frac)

        print('\n')
        for mode in modes:
            print(f'Number of BDT inputs for {mode:{' '}{'<'}{15}} = {N_BDT_inputs[mode]}')
            df[mode] = df_split_data(df[mode], N_BDT_inputs, eff, xsec, N_events, mode)

        dfsum = pd.concat([df[mode] for mode in modes])
        to_pkl(dfsum, pkl_path)



if __name__=='__main__':
    run(inDir, sels, modes, vars, f'Z{cat}H', procDict_name)
    timer(t)

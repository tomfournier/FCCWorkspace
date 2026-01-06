##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Standard library imports for timing and command-line arguments
from time import time
from argparse import ArgumentParser

# Data manipulation
import pandas as pd

# Start execution timer
t = time()

# Import configuration and utilities
from package.userConfig import loc, get_loc
from package.config import (
    timer, warning, 
    input_vars
)
# Utility functions for file handling and process dictionaries
from package.tools.utils import (
    get_paths, 
    to_pkl, 
    get_procDict, 
    update_keys
)
# BDT-specific functions for data processing
from package.func.bdt import (
    counts_and_effs, 
    additional_info, 
    BDT_input_numbers, 
    df_split_data
)



########################
### ARGUMENT PARSING ###
########################

# Command-line argument parsing
parser = ArgumentParser()
# Define final state: ee or mumu
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

# Validate that final state was selected
if arg.cat=='':
    warning(log_msg='Final state was not selected, please select one to run this script')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Set analysis category and energy from arguments
cat, ecm = arg.cat, arg.ecm
inDir = get_loc(loc.HIST_MVA, cat, ecm, '')

# Selection strategies to process
sels = [
    'Baseline'
]

# Decay modes used in first stage training and their respective file names
modes = {
    f'Z{cat}H':       f'wzp6_ee_{cat}H_ecm{ecm}',                # Signal: ZH production
    f'ZZ':           f'p8_ee_ZZ_ecm{ecm}',                       # Background: diboson ZZ
    f'Z{cat}':       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee'  # Background: Z+jets
                     else f'wzp6_ee_mumu_ecm{ecm}',
    f'WW{cat}':      f'p8_ee_WW_{cat}_ecm{ecm}',              # Background: diboson WW
    f'gammae_{cat}': f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',       # Background: radiative Z
    f'egamma_{cat}': f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',       # Background: radiative Z
    f'gaga_{cat}':   f'wzp6_gaga_{cat}_60_ecm{ecm}'           # Background: diphoton
}

# Name of the dictionary that contains all the cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict_name = 'FCCee_procDict_winter2023_training_IDEA.json'



##########################
### EXECUTION FUNCTION ###
##########################

def run(inDir: str, 
        sels:  list[str], 
        modes: list[str], 
        vars:  list[str], 
        sig: str, 
        procDict_name: str
        ) -> None:
    """Process MVA input histograms and prepare BDT training data."""

    # Load process dictionary and update keys with mode mapping
    proc_dict = get_procDict(procDict_name)
    procDict = update_keys(proc_dict, modes)

    # Extract cross sections for each mode
    xsec = {}
    for key, value in procDict.items(): 
        if key in modes: xsec[key] = value['crossSection']

    # Set uniform fraction for all modes
    frac = {mode: 1.0 for mode in modes}

    for sel in sels:
        # Define output path for preprocessed data
        pkl_path = get_loc(loc.MVA_INPUTS, cat, ecm, sel)

        # Initialize storage for each mode
        files, df, eff, N_events = {}, {}, {}, {}

        # Process each decay mode
        for mode in modes:
            # Get input files
            files[mode] = get_paths(mode, inDir, modes, f'_{sel}')

            # Load data and calculate efficiencies
            df[mode], eff[mode], N_events[mode] = counts_and_effs(files[mode], vars, only_eff=False)
            print(f'Number of events in {mode} = {N_events[mode]}')
            print(f'Efficiency of {mode} = {eff[mode]*100:.3f}%')

            # Add signal/background labels and weights
            df[mode] = additional_info(df[mode], mode, sig)

        # Calculate optimal number of BDT inputs per mode
        N_BDT_inputs = BDT_input_numbers(df, modes, sig, eff, xsec, frac)

        print('\n')
        # Split data for each mode into training and validation sets
        for mode in modes:
            print(f'Number of BDT inputs for {mode:<{15}} = {N_BDT_inputs[mode]}')
            df[mode] = df_split_data(df[mode], N_BDT_inputs, eff, xsec, N_events, mode)

        # Combine all modes and save to pickle file
        dfsum = pd.concat([df[mode] for mode in modes])
        to_pkl(dfsum, pkl_path)



######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run preprocessing pipeline and prepare BDT inputs
    run(inDir, sels, modes, input_vars, f'Z{cat}H', procDict_name)

    # Print execution time
    timer(t)

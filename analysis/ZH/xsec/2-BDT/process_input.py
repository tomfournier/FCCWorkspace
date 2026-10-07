#################################
### IMPORT STANDARD LIBRARIES ###
#################################

# Standard library imports for timing and command-line arguments
import time

# Data manipulation
import pandas as pd

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from zh_xsec.parsing import create_parser, parse_args, set_log
from zh_xsec.logger import get_logger
parser = create_parser('2-BDT', 'process_input')
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

LOGGER.debug('Loading custom modules')

# Configuration and directory management
from zh_xsec.userConfig import loc
from zh_xsec.config import (
    timer,                      # Performance timing utility
    get_bdt_modes,              # Build BDT signal and background samples
    input_vars_ll,              # List of variables for BDT training (hadronic channel)
    input_vars_qq,              # List of variables for BDT training (hadronic channel)
)

# File I/O and process dictionary utilities
from zh_xsec.tools.utils import (
    get_paths,                  # Find histogram files for each process
    to_pkl,                     # Save dataframes to pickle format
    get_procDict,               # Load process metadata
)

# BDT data preparation functions
from zh_xsec.func.bdt import (
    counts_and_effs,            # Calculate event counts and efficiencies
    additional_info,            # Add signal/background labels and weights
    BDT_input_numbers,          # Determine optimal training set sizes
    sample_df_by_xsec,          # Sample process dataframes in proportion to cross-sections
    df_split_data,              # Split data into training/validation sets
    apply_balanced_training_weights
)

LOGGER.debug('Modules loaded')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Analysis parameters from command-line arguments
cat, ecm, sels = arg.cat, arg.ecm, arg.sels.split('-')
lumi = 10.8 if ecm == 240 else (3.12 if ecm == 365 else -1)

inputdir = loc.get('HIST_MVA', cat, ecm)  # Input directory with MVA histograms
input_vars = input_vars_ll if cat in ['ee', 'mumu'] else input_vars_qq

# Process modes for BDT training (signal and all major background processes)
modes = get_bdt_modes(cat, ecm)

# Signal mode
sig = f'Z{cat}H'

# Process dictionary with cross-sections and normalization info
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict_name = 'FCCee_procDict_winter2023_training_IDEA.json'

# Set uniform reweighting fraction for all processes
# (can be adjusted to emphasize specific backgrounds)
# Choose frac = 1 if not provided
frac = {}



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    """Process MVA input histograms and prepare balanced BDT training data.

    This function loads histograms produced by final-selection.py, calculates
    event efficiencies, applies signal/background labels, and creates balanced
    training/validation datasets for BDT training.
    """

    # Load process dictionary and map sample names
    procDict = get_procDict(procDict_name)

    # Extract cross sections for each process (used for normalization)
    xsec: dict[str, float] = {}
    proc_xsec: dict[str, float] = {}
    for mode, procs in modes.items():
        xsec[mode] = sum(procDict[proc]['crossSection'] for proc in procs)
        for proc in procs:
            proc_xsec[proc] = procDict[proc]['crossSection']

    for sel in sels:
        # Output directory for preprocessed pickle files
        outputdir = loc.get('MVA_INPUTS', cat, ecm, sel)

        # Initialize storage containers for each process
        eff, eff_proc, N_procs, N_events = {}, {}, {}, {m:0 for m in modes}
        df: dict[str, pd.DataFrame] = {}

        # Formatting for aligned console output
        lenght = max(len(m) for m in modes)
        modes_list = list(modes.keys())
        LOGGER.info(f'Modes used: {", ".join(modes_list)}\n')

        # Process each decay mode: load dataframe, compute weights
        for mode in modes_list:
            procs = modes[mode]
            # Locate histogram files for this process and selection
            df_mode: dict[str, pd.DataFrame] = {}
            selected_events = 0
            for proc in procs:
                files = get_paths(proc, inputdir, f'_{sel}')

                # Load data from TTrees and calculate survival efficiency
                df_proc, eff_proc[proc], N_procs[proc] = counts_and_effs(files, input_vars, only_eff=False)
                N_events[mode]  += N_procs[proc]
                selected_events += df_proc.shape[0]

                # Add signal/background classification and event weights
                df_proc = additional_info(df_proc, proc_xsec, eff_proc, mode, proc, sig)
                df_mode[proc] = df_proc

            if selected_events == 0:
                df[mode], eff[mode] = pd.DataFrame(), 0.0
                LOGGER.info(f'Number of events in {mode:<{lenght}} = {N_events[mode]:,}\n'
                            f'      Efficiency of {mode:<{lenght}} = {eff[mode]*100:.3}%')
                continue

            # Sample the mode dataframe in proportion to process cross-sections.
            df[mode]  = sample_df_by_xsec(df_mode, proc_xsec, eff_proc, selected_events, mode,
                                          1, arg.all_inputs, arg.n_max)
            eff[mode] = selected_events / N_events[mode] if N_events[mode] > 0 else 0.0

            LOGGER.info(f'Number of events in {mode:<{lenght}} = {N_events[mode]:,}\n'
                        f'      Efficiency of {mode:<{lenght}} = {eff[mode]*100:.3}%')


        # Calculate how many events to use from each process for balanced training
        N_BDT_inputs = BDT_input_numbers(df, modes, sig, eff, xsec, frac, arg.all_inputs, arg.n_max)

        LOGGER.debug('Printing BDT inputs number for the different modes')
        # Split data into training/validation sets per process
        for mode in modes:
            LOGGER.info(f'Number of BDT inputs for {mode:<{lenght}} = {N_BDT_inputs[mode]:,}')
            if df[mode].shape[0] == 0: continue
            df[mode] = df_split_data(df[mode], N_BDT_inputs, mode, lumi, arg.test_size)

        good_modes = apply_balanced_training_weights(df, modes, sig)

        # Merge all processes and save to single pickle file for BDT training
        dfsum = pd.concat([df[mode] for mode in good_modes])
        to_pkl(dfsum, input_vars, outputdir)


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run preprocessing pipeline and prepare BDT inputs
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

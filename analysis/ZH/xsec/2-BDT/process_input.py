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

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sels=True,
    allow_qq=True,
    description='BDT Input Processing Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

LOGGER.debug('Loading custom modules')

# Configuration and directory management
from package.userConfig import loc
from package.config import (
    timer,                      # Performance timing utility
    input_vars_ll,              # List of variables for BDT training (hadronic channel)
    input_vars_qq               # List of variables for BDT training (hadronic channel)
)

# File I/O and process dictionary utilities
from package.tools.utils import (
    get_paths,                  # Find histogram files for each process
    to_pkl,                     # Save dataframes to pickle format
    get_procDict,               # Load process metadata
)

# BDT data preparation functions
from package.func.bdt import (
    counts_and_effs,            # Calculate event counts and efficiencies
    additional_info,            # Add signal/background labels and weights
    BDT_input_numbers,          # Determine optimal training set sizes
    df_split_data               # Split data into training/validation sets
)

LOGGER.debug('Modules loaded')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Analysis parameters from command-line arguments
cat, ecm = arg.cat, arg.ecm  # Decay category and center-of-mass energy
inDir = loc.get('HIST_MVA', cat, ecm)  # Input directory with MVA histograms
input_vars = input_vars_ll if cat in ['ee', 'mumu'] else input_vars_qq

# Selection strategies to process (from command-line or defaults)
if arg.sels=='':
    sels = ['Baseline']          # Default selections if not specified
else:
    sels = arg.sels.split('-')   # Parse selection names from command-line

# Process modes for BDT training (signal and all major background processes)
modes = {
    f'Z{cat}H':      [f'wzp6_ee_{cat}H_ecm{ecm}'],                 # Signal: ZH production
    'ZZ':            [f'p8_ee_ZZ_ecm{ecm}'],                       # Background: diboson ZZ
    f'Z{cat}':       [f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee'  # Background: Z+jets
                      else f'wzp6_ee_{cat}_ecm{ecm}'],
    f'WW{cat}':      [f'p8_ee_WW_ecm{ecm}' if cat == 'qq'          # Background: diboson WW
                      else f'p8_ee_WW_{cat}_ecm{ecm}'],
    f'gammae_{cat}': [f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}'],          # Background: radiative Z
    f'egamma_{cat}': [f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}'],          # Background: radiative Z
}
if (cat != 'qq') and not ((cat == 'ee') and ecm == 365):
    modes[f'gaga_{cat}'] = [f'wzp6_gaga_{cat}_60_ecm{ecm}']        # Background: diphoton

# Process dictionary with cross-sections and normalization info
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict_name = 'FCCee_procDict_winter2023_training_IDEA.json'



##########################
### EXECUTION FUNCTION ###
##########################

def run(
    inDir: str,
    sels: list[str],
    modes: dict[str, list[str]],
    vars: list[str],
    sig: str,
    procDict_name: str,
     ) -> None:
    """Process MVA input histograms and prepare balanced BDT training data.

    This function loads histograms produced by final-selection.py, calculates
    event efficiencies, applies signal/background labels, and creates balanced
    training/validation datasets for BDT training.

    Args:
        inDir: Directory containing input histograms from final-selection.py
        sels: List of selection strategies to process
        modes: Dictionary mapping each mode to its list of process identifiers
        vars: List of input variables for BDT training
        sig: Signal process name (e.g., 'ZeeH' or 'ZmumuH')
        procDict_name: Path to process dictionary with cross-sections

    Returns:
        None (writes preprocessed dataframes to pickle files in MVA_INPUTS directories)
    """

    # Load process dictionary and map sample names
    procDict = get_procDict(procDict_name)

    # Extract cross sections for each process (used for normalization)
    xsec: dict[str, float] = {}
    for mode, procs in modes.items():
        xsec[mode] = sum(procDict[proc]['crossSection'] for proc in procs)

    # Set uniform reweighting fraction for all processes
    # (can be adjusted to emphasize specific backgrounds)
    frac = {mode: 0.5 if mode==sig else 1.0 for mode in modes}

    for sel in sels:
        # Output directory for preprocessed pickle files
        outDir = loc.get('MVA_INPUTS', cat, ecm, sel)

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
            df_mode = []
            for proc in procs:
                files = get_paths(proc, inDir, f'_{sel}')

                # Load data from TTrees and calculate survival efficiency
                df_proc, eff_proc[proc], N_procs[proc] = counts_and_effs(files, vars, only_eff=False)
                N_events[mode] += N_procs[proc]

                # Add signal/background classification and event weights
                df_proc = additional_info(df_proc, mode, proc, sig)
                df_mode.append(df_proc)

            df[mode]  = pd.concat(df_mode)
            eff[mode] = df[mode].shape[0] / N_events[mode] if N_events[mode] > 0 else 0.0

            LOGGER.info(f'Number of events in {mode:<{lenght}} = {N_events[mode]:,}\n'
                        f'      Efficiency of {mode:<{lenght}} = {eff[mode]*100:.3}%')


        # Calculate how many events to use from each process for balanced training
        N_BDT_inputs = BDT_input_numbers(df, modes, sig, eff, xsec, frac)

        LOGGER.debug('Printing BDT inputs number for the different modes')
        # Split data into training (50%) and validation (50%) sets per process
        good_modes = []
        for mode in modes:
            LOGGER.info(f'Number of BDT inputs for {mode:<{lenght}} = {N_BDT_inputs[mode]:,}')
            if df[mode].shape[0] == 0:
                continue
            df[mode] = df_split_data(
                df[mode], N_BDT_inputs,
                eff, xsec, N_events, mode
            )
            good_modes.append(mode)

        # Merge all processes and save to single pickle file for BDT training
        dfsum = pd.concat([df[mode] for mode in good_modes])
        to_pkl(dfsum, outDir)


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run preprocessing pipeline and prepare BDT inputs
        run(inDir, sels, modes, input_vars, f'Z{cat}H', procDict_name)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

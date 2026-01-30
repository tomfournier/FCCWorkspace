##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

print('----->[Info] Loading modules')

# Standard library and scientific computing imports
from time import time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xgboost as xgb

from argparse import ArgumentParser

# Start execution timer
t = time()

print('----->[Info] Loading custom modules')

# Import configuration paths and plot settings
from package.userConfig import loc, plot_file
# Import utilities and plotting configurations
from package.config import (
    timer, warning, 
    input_vars, 
    modes_label, 
    modes_color, 
    vars_label, 
    vars_xlabel
)
# Utility functions for data loading
from package.tools.utils import mkdir, load_data
# BDT model utilities
from package.func.bdt import (
    load_model, 
    get_metrics, 
    print_stats, 
    evaluate_bdt
)



########################
### ARGUMENT PARSING ###
########################

# Command-line argument parsing
parser = ArgumentParser()
# Define final state: ee or mumu
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)

parser.add_argument('--metric', help='Do not plot the metrics plots',        action='store_true')
parser.add_argument('--tree',   help='Plot the Decision Trees from the BDT', action='store_true')
parser.add_argument('--check',  help='Plot the variables distribution',      action='store_true')
parser.add_argument('--hl',     help='Plot the variables distribution for high and low score region', action='store_true')
arg = parser.parse_args()

# Validate that final state was selected
if arg.cat=='':
    warning(log_msg='Final state was not selected, please select one to run this script')
    exit(1)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Set analysis category and energy from arguments
cat, ecm = arg.cat, arg.ecm

# Selection strategies to evaluate
sels = [
    'Baseline',
    # 'test'
]

# Decay modes used in first stage training and their respective file names
modes = {
  f'Z{cat}H':      f'wzp6_ee_{cat}H_ecm{ecm}',                 # Signal: ZH production
  f'WW{cat}':      f'p8_ee_WW_{cat}_ecm{ecm}',                 # Background: diboson WW
  f'ZZ':           f'p8_ee_ZZ_ecm{ecm}',                       # Background: diboson ZZ
  f'Z{cat}':       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee'  # Background: Z+jets
              else f'wzp6_ee_mumu_ecm{ecm}',
  f'egamma_{cat}': f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',       # Background: radiative Z
  f'gammae_{cat}': f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',       # Background: radiative Z
  f'gaga_{cat}':   f'wzp6_gaga_{cat}_60_ecm{ecm}'           # Background: diphoton
}



##########################
### EXECUTION FUNCTION ###
##########################

def plot_metrics(df: 'pd.DataFrame', 
                 bdt: 'xgb.XGBClassifier', 
                 vars: list[str], 
                 results: dict[str, 
                               dict[str, 
                                    list[float]]], 
                 x_axis: 'np.ndarray', 
                 modes: list[str], 
                 cat: str, 
                 outDir: str) -> None:
    """Generate comprehensive BDT evaluation plots."""

    # Set LaTeX label based on final state
    if cat=='mumu': label = r'$Z(\mu^+\mu^-)H$'
    elif cat=='ee': label = r'$Z(e^+e^-)H$'
    else: warning('Invalid final state') 

    # Create output directory
    mkdir(outDir)
    
    if not arg.metric:
        from package.plots.eval import (
            log_loss, classification_error, 
            AUC, roc, bdt_score, mva_score, 
            importance, significance, efficiency
        )

        # Generate training performance plots
        log_loss(results, x_axis, label, outDir, best_iteration, format=plot_file)
        classification_error(results, x_axis, label, outDir, best_iteration, format=plot_file)
        AUC(results, x_axis, label, outDir, best_iteration, format=plot_file)
        
        # Generate BDT response plots
        roc(df, label, outDir, format=plot_file)
        bdt_score(df, label, outDir, format=plot_file, unity=False, nbins=200)
        mva_score(df, label, outDir, modes, modes_label, modes_color, format=plot_file, unity=False, nbins=200)
        
        # Generate feature and performance analysis plots
        importance(bdt, vars, vars_label, label, outDir, format=plot_file)
        significance(df, label, outDir, inBDT, format=plot_file, weight='norm_weight') # weights
        efficiency(df, modes, modes_label, modes_color, label, outDir, incr=1e-3, format=plot_file)

    if arg.tree:
        from package.plots.eval import tree_plot
        tree_plot(bdt, inBDT, outDir, epochs, 20, format=plot_file)

    # Generate input variable distribution checks
    if arg.check:
        from package.plots.eval import hist_check
        for var in vars:
            print(f'------>Plotting histogram for {var}')
            hist_check(
                df, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                yscale='linear', suffix='_lin', format=plot_file, strict=True
            )
            hist_check(
                df, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                yscale='log', suffix='_log', format=plot_file, strict=True
            )
    if arg.hl:
        import numpy as np
        from package.plots.eval import hist_check
        print('\n------>PLotting histogram in high/low BDT score region')
        bdt_cut = np.loadtxt(f'{inBDT}/BDT_cut.txt')
        df_high = df.query(f'BDTscore > {bdt_cut}')
        df_low  = df.query(f'BDTscore < {bdt_cut}')
        for var in vars:
            print(f'------>Plotting histogram for {var}')
            hist_check(
                df_high, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                yscale='linear', suff='high', suffix='_lin', format=plot_file, strict=True
            )
            hist_check(
                df_high, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                yscale='log', suff='high', suffix='_log', format=plot_file, strict=True
            )
            
            hist_check(
                df_low, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                yscale='linear', suff='low', suffix='_lin', format=plot_file, strict=True
            )
            hist_check(
                df_low, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                yscale='log', suff='low', suffix='_log', format=plot_file, strict=True
            )


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Evaluate BDT performance for each selection strategy
    for sel in sels:
        # Define input/output directories
        inDir  = loc.get('MVA_INPUTS',  cat, ecm, sel)
        outDir = loc.get('PLOTS_BDT',   cat, ecm, sel)
        inBDT  = loc.get('BDT',         cat, ecm, sel)
        data_path = loc.get('HIST_MVA', cat, ecm, sel)

        if 'Baseline' in sel and cat=='ee' and ecm==365:
            Modes = {m:proc for m, proc in modes.items() if m not in 'gaga_ee'}
        else:
            Modes = modes.copy()

        # Load preprocessed data and print statistics
        print(f'----->[Info] Getting DataFrame from {sel}\n')
        df = load_data(inDir)
        print_stats(df, Modes)
        
        # Load trained BDT model
        print('\n----->[Info] Loading BDT')
        bdt = load_model(inBDT)
        
        # Apply BDT to data and calculate scores
        print('----->[Info] Evaluating BDT')
        df = evaluate_bdt(df, bdt, input_vars)
        
        # Extract training metrics
        print('----->[Info] Extracting metrics from BDT')
        results, epochs, x_axis, best_iteration = get_metrics(bdt)
        
        # Generate all evaluation plots
        print('\n----->[Info] Plotting the metrics for the BDT')
        plot_metrics(df, bdt, input_vars, results, x_axis, Modes, cat, outDir)

    # Print execution time
    timer(t)

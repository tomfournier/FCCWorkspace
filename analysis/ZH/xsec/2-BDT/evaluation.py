#################################
### IMPORT STANDARD LIBRARIES ###
#################################

# Standard library and scientific computing imports
from time import time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xgboost as xgb

# Start execution timer
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sels=True,
    bdt_eval=True,
    allow_qq=False,
    description='BDT Evaluation Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Import plot configuration and directory paths
from package.userConfig import loc, plot_file

# Import configuration utilities and labels paremeters
from package.config import (
    timer, warning,                # Utility functions
    input_vars,                    # Training variable list
    modes_label, modes_color,      # Plot styling for processes
    vars_label, vars_xlabel        # Variable naming for plots
)

# Import data handling utilities
from package.tools.utils import mkdir, load_data

# Import BDT model utilities
from package.func.bdt import (
    load_model,                    # Load trained XGBoost model
    get_metrics,                   # Extract training curves from model
    print_stats,                   # Display event statistics
    evaluate_bdt                   # Apply BDT to data and compute scores
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Analysis parameters from command-line arguments
cat, ecm = arg.cat, arg.ecm  # Decay category and center-of-mass energy

# Selection strategies to evaluate (from command-line or defaults)
if arg.sels=='':
    sels = ['Baseline', 'test']  # Default selections if not specified
else:
    sels = arg.sels.split('-')   # Parse selection names from command-line

# Process modes: maps display names to file naming convention
# Used for loading data and labeling plots
modes = {
    f'Z{cat}H':      f'wzp6_ee_{cat}H_ecm{ecm}',                 # Signal: ZH production
    f'WW{cat}':      f'p8_ee_WW_{cat}_ecm{ecm}',                 # Background: diboson WW
    'ZZ':            f'p8_ee_ZZ_ecm{ecm}',                       # Background: diboson ZZ
    f'Z{cat}':       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee'  # Background: Z+jets
                     else f'wzp6_ee_mumu_ecm{ecm}',
    f'egamma_{cat}': f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',       # Background: radiative Z
    f'gammae_{cat}': f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',       # Background: radiative Z
    f'gaga_{cat}':   f'wzp6_gaga_{cat}_60_ecm{ecm}'           # Background: diphoton
}



#########################
### PLOTTING FUNCTION ###
#########################

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
    """Generate comprehensive BDT evaluation plots and performance metrics.

    Creates multiple categories of plots:
    - Training curves: Loss, classification error, AUC vs boosting rounds
    - Model response: ROC curves, BDT score distributions
    - Feature analysis: Feature importance, signal significance
    - Event distributions: Histograms of training variables (optionally binned by BDT score)
    - Tree visualization: Visual representation of individual decision trees (optional)

    Args:
        df: Evaluated dataframe with BDT scores and event weights
        bdt: Trained XGBoost model
        vars: List of input variables used for training
        results: Dictionary with training curves (loss, error, AUC) per round
        x_axis: Array of boosting rounds for plotting curves
        modes: List of process names for legends
        cat: Decay category (for plot labeling)
        outDir: Output directory for plots

    Returns:
        None (writes plots to outDir)
    """

    # Set LaTeX labels for final state particles
    if cat=='mumu': label = r'$Z(\mu^+\mu^-)H$'
    elif cat=='ee': label = r'$Z(e^+e^-)H$'
    else: warning('Invalid final state')

    # Create output directory
    mkdir(outDir)

    if arg.metric:
        # Lazily import plotting functions for model performance
        from package.plots.eval import (
            log_loss,                  # Training/validation loss curves
            classification_error,      # Error rate vs boosting rounds
            AUC,                       # ROC AUC vs boosting rounds
            roc,                       # ROC curve (sensitivity vs false positive rate)
            bdt_score,                 # BDT score distribution
            mva_score,                 # BDT score per process
            importance,                # Feature importance ranking
            significance,              # Signal significance vs BDT cut
            efficiency                 # Selection efficiency curves
        )

        LOGGER.info('Plotting the metrics for the BDT\n')

        # Generate training performance plots
        # These show how well the BDT is learning over iterations
        log_loss(results, x_axis, label, outDir, best_iteration, format=plot_file)
        classification_error(results, x_axis, label, outDir, best_iteration, format=plot_file)
        AUC(results, x_axis, label, outDir, best_iteration, format=plot_file)

        # Generate model response plots
        # These show the BDT discrimination power
        roc(df, label, outDir, format=plot_file)
        bdt_score(df, label, outDir, format=plot_file, unity=False, nbins=200)
        mva_score(df, label, outDir, modes, modes_label, modes_color, format=plot_file, unity=False, nbins=200)

        # Generate feature and performance analysis plots
        # These show which variables are most important and signal purity
        importance(bdt, vars, vars_label, label, outDir, format=plot_file)
        significance(df, label, outDir, inBDT, format=plot_file, weight='weights')
        efficiency(df, modes, modes_label, modes_color, label, outDir, incr=1e-3, format=plot_file)

    if arg.tree:
        # Generate visualizations of individual decision trees in the BDT
        from package.plots.eval import tree_plot
        LOGGER.info('Plotting the different decision trees in the BDT')
        tree_plot(bdt, inBDT, outDir, epochs, 20, format=plot_file)

    # Check input variable distributions for anomalies
    if arg.check:
        from package.plots.eval import hist_check
        LOGGER.info('Plotting histograms for input variables')
        for var in vars:
            LOGGER.info(f'Plotting histogram for {var}')
            # Create plots with both linear and logarithmic y-axes
            for yscale, suffix in [('linear', '_lin'), ('log', '_log')]:
                hist_check(
                    df, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var],
                    yscale=yscale, suffix=suffix, format=plot_file, strict=True
                )

    # Optionally generate distributions in high/low BDT score regions
    if arg.hl:
        import numpy as np
        from package.plots.eval import hist_check
        LOGGER.info('Plotting histograms for input variables in high/low BDT score regions')
        bdt_cut = np.loadtxt(f'{inBDT}/BDT_cut.txt')
        df_high = df.query(f'BDTscore > {bdt_cut}')  # Signal-enriched region
        df_low  = df.query(f'BDTscore < {bdt_cut}')  # Background-enriched region
        for var in vars:
            LOGGER.info(f'Plotting histogram for {var}')
            for yscale, suffix in [('linear', '_lin'), ('log', '_log')]:
                hist_check(
                    df_high, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var],
                    yscale=yscale, suff='high', suffix=suffix, format=plot_file, strict=True
                )
                hist_check(
                    df_low, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var],
                    yscale=yscale, suff='low', suffix=suffix, format=plot_file, strict=True
                )


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Evaluate trained BDT models for each selection strategy
        for sel in sels:
            # Input: Preprocessed data and trained BDT model
            inDir  = loc.get('MVA_INPUTS',  cat, ecm, sel)
            inBDT  = loc.get('BDT',         cat, ecm, sel)
            # Output: Evaluation plots
            outDir = loc.get('PLOTS_BDT',   cat, ecm, sel)
            # Histograms for reference
            data_path = loc.get('HIST_MVA', cat, ecm, sel)

            # Skip diphoton at 365 GeV (efficiency too low, not enough events)
            if cat=='ee' and ecm==365:
                Modes = {m:proc for m, proc in modes.items() if m not in 'gaga_ee'}
            else:
                Modes = modes.copy()

            # Load preprocessed evaluation data
            LOGGER.info(f'Getting DataFrame from {sel}')
            df = load_data(inDir)
            print_stats(df, Modes)

            # Load trained XGBoost model
            LOGGER.debug('Loading trained BDT model')
            bdt = load_model(inBDT)

            # Apply BDT to data to compute classification scores
            LOGGER.debug('Evaluating BDT on data')
            df = evaluate_bdt(df, bdt, input_vars)

            # Extract training metrics from model object
            # (loss, error, AUC curves, best iteration, etc.)
            LOGGER.debug('Extracting metrics from BDT')
            results, epochs, x_axis, best_iteration = get_metrics(bdt)

            # Generate all evaluation plots and performance metrics
            plot_metrics(df, bdt, input_vars, results, x_axis, Modes, cat, outDir)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

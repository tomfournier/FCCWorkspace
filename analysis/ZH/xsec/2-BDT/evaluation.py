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
parser = create_parser('2-BDT', 'evaluation')
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Import plot configuration and directory paths
from package.userConfig import loc, PathObj, plot_file
loc.set_default_type(PathObj)

# Import configuration utilities and labels paremeters
from package.config import (
    timer,                           # Utility function
    modes_label, modes_color,        # Plot styling for processes
    vars_label_ll, vars_xlabel_ll,   # Variable naming for plots (leptonic channel)
    vars_label_qq, vars_xlabel_qq,   # Variable naming for plots (hadronic channel)
    get_bdt_modes                    # Build BDT signal and background samples
)

# Import data handling utilities
from package.tools.utils import load_data

# Import BDT model utilities
from package.func.bdt import (
    load_model,    # Load trained XGBoost model
    get_metrics,   # Extract training curves from model
    print_stats,   # Display event statistics
    evaluate_bdt   # Apply BDT to data and compute scores
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Analysis parameters from command-line arguments
cat, ecm, sels = arg.cat, arg.ecm, arg.sels.split('-')
vars_label     = vars_label_ll  if cat in ['ee', 'mumu'] else vars_label_qq
vars_xlabel    = vars_xlabel_ll if cat in ['ee', 'mumu'] else vars_xlabel_qq

# Process modes for BDT training (signal and all major background processes)
modes = get_bdt_modes(cat, ecm)



#########################
### PLOTTING FUNCTION ###
#########################

def plot_metrics(
        df: 'pd.DataFrame',
        bdt: 'xgb.XGBClassifier',
        results: dict[str, dict[str, list[float]]],
        x_axis: 'np.ndarray',
        modes: list[str],
        cat: str,
        outputdir: PathObj) -> None:
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
        outputdir: Output directory for plots

    Returns:
        None (writes plots to outputdir)
    """

    # Set LaTeX labels for final state particles
    if cat == 'mumu': label = r'$Z(\to \mu^+\mu^-)H$'
    elif cat == 'ee': label = r'$Z(\to e^+e^-)H$'
    elif cat == 'qq': label = r'$Z(\to q\bar{q})H$'
    else: raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')

    # Create output directory
    outputdir.mkdir(exist_ok=True, parents=True)

    if arg.metric:
        # Lazily import plotting functions for model performance
        from package.plots.eval import (
            log_loss,       # Training/validation loss curves
            error,          # Error rate vs boosting rounds
            AUC,            # ROC AUC vs boosting rounds
            roc_curve,      # ROC curve (sensitivity vs false positive rate)
            bdt_score,      # BDT score distribution
            mva_score,      # BDT score per process
            importance,     # Feature importance ranking
            significance,   # Signal significance vs BDT cut
            efficiency      # Selection efficiency curves
        )

        LOGGER.info('Plotting the metrics for the BDT\n')

        # Generate training performance plots
        # These show how well the BDT is learning over iterations
        log_loss(results, x_axis, label, outputdir, best_iteration, format=plot_file)
        error(results, x_axis, label, outputdir, best_iteration, format=plot_file)
        AUC(results, x_axis, label, outputdir, best_iteration, format=plot_file)

        # Generate model response plots
        # These show the BDT discrimination power
        roc_curve(df, label, outputdir, format=plot_file)
        bdt_score(df, label, outputdir, format=plot_file, unity=True, nbins=200, yscale='linear', suffix='_lin')
        bdt_score(df, label, outputdir, format=plot_file, unity=True, nbins=200, yscale='log',    suffix='_log')
        mva_score(df, label, outputdir, modes, modes_label, modes_color, format=plot_file, unity=False, nbins=200)

        # Generate feature and performance analysis plots
        # These show which variables are most important and signal purity
        importance(bdt, input_vars, vars_label, label, outputdir, format=plot_file)
        significance(df, label, outputdir, loc_BDT, format=plot_file, weight='weights',       suffix='_weights')
        significance(df, label, outputdir, loc_BDT, format=plot_file, weight='train_weights', suffix='_train_weights')
        significance(df, label, outputdir, loc_BDT, format=plot_file, weight='norm_weight',   suffix='_norm_weight')
        efficiency(df, modes, modes_label, modes_color, label, outputdir, incr=1e-3, format=plot_file)

    if arg.tree:
        # Generate visualizations of individual decision trees in the BDT
        from package.plots.eval import tree_plot
        LOGGER.info('Plotting the different decision trees in the BDT')
        tree_plot(bdt, loc_BDT, outputdir, epochs, 20, format=plot_file)

    # Check input variable distributions for anomalies
    if arg.check:
        from package.plots.eval import hist_check
        LOGGER.info('Plotting histograms for input variables')
        for var in input_vars:
            LOGGER.info(f'Plotting histogram for {var}')
            # Create plots with both linear and logarithmic y-axes
            for yscale, suffix in [('linear', '_lin'), ('log', '_log')]:
                hist_check(df, label, outputdir, modes, modes_label, modes_color, var, vars_xlabel[var],
                           yscale=yscale, suffix=suffix, format=plot_file)

    # Optionally generate distributions in high/low BDT score regions
    if arg.hl:
        import numpy as np
        from package.plots.eval import hist_check
        LOGGER.info('Plotting histograms for input variables in high/low BDT score regions')
        bdt_cut = np.loadtxt(f'{loc_BDT}/BDT_cut_weights.txt')
        df_high = df.query(f'BDTscore > {bdt_cut}')  # Signal-enriched region
        df_low  = df.query(f'BDTscore < {bdt_cut}')  # Background-enriched region
        for var in input_vars:
            LOGGER.info(f'Plotting histogram for {var}')
            for yscale, suffix in [('linear', '_lin'), ('log', '_log')]:
                hist_check(df_high, label, outputdir, modes, modes_label, modes_color, var, vars_xlabel[var],
                           yscale=yscale, suff='high', suffix=suffix, format=plot_file)
                hist_check(df_low, label, outputdir, modes, modes_label, modes_color, var, vars_xlabel[var],
                           yscale=yscale, suff='low', suffix=suffix, format=plot_file)


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Evaluate trained BDT models for each selection strategy
        for sel in sels:
            inputdir  = loc.get('MVA_INPUTS',  cat, ecm, sel)
            outputdir = loc.get('PLOTS_BDT',   cat, ecm, sel)
            loc_BDT   = loc.get('BDT',         cat, ecm, sel)

            # Load preprocessed evaluation data
            LOGGER.info(f'Getting DataFrame from {sel}')
            df, input_vars = load_data(inputdir)

            LOGGER.info(f'Using {", ".join(input_vars)}')
            print_stats(df, modes)

            # Load trained XGBoost model
            LOGGER.debug('Loading trained BDT model')
            bdt = load_model(loc_BDT)

            # Apply BDT to data to compute classification scores
            LOGGER.debug('Evaluating BDT on data')
            df = evaluate_bdt(df, bdt, input_vars)

            # Extract training metrics from model object
            # (loss, error, AUC curves, best iteration, etc.)
            LOGGER.debug('Extracting metrics from BDT')
            results, epochs, x_axis, best_iteration = get_metrics(bdt)

            # Generate all evaluation plots and performance metrics
            plot_metrics(df, bdt, results, x_axis, modes, cat, outputdir)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

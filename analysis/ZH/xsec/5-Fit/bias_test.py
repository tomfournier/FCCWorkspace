################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, time, subprocess

import numpy as np
import pandas as pd

# Start timer for performance tracking
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,       # Support single decay category
    allow_empty=True,      # Allow empty category
    include_sel=True,      # Include selection strategy options
    fit=True,              # Include fit-specific options
    bias=True,             # Include bias test options
    bias_extra=True,       # Include extra bias test parameters
    polarization=True,     # Include polarization/scale options
    default_target='bb',   # Default Higgs decay mode
    default_pert=1.05,     # Default perturbation for bias test
    do_bias=True           # Remove fit exclusive argument
)
arg = parse_args(parser, comb=True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory configuration and analysis utilities
from package.userConfig import loc
from package.config import (
    timer,              # Timing utility
    mk_processes,       # Build process definitions
    z_decays,           # Z boson decay modes
    h_decays,           # Higgs decay modes (visible)
    H_decays,           # Higgs decay modes (all)
)
from package.plots.plotting import Bias, PseudoRatio          # Plotting utilities
from package.tools.utils import mkdir                         # Directory creation
from package.tools.process import (                           # Process utilities
    preload_histograms,                                       # Cache histograms
    clear_histogram_cache,
    getMetaInfo
)
from package.func.bias import pseudo_datacard                 # Bias test utilities



###############################
### CONFIGURATION AND SETUP ###
###############################

# Bundle parsed arguments for configuration
cat, ecm, sel, pert = arg.cat, arg.ecm, arg.sel, arg.pert  # Perturbation factor for bias test
lumi = 10.8 if ecm==240 else (3.1 if ecm==365 else -1)     # Integrated luminosity [ab^-1]
if (arg.cat and arg.lep) or (arg.cat and arg.combine):
    raise ValueError("Cannot use '--cat' and '--lep' or '--combine' at the same time")
if arg.lep and arg.combine:
    raise ValueError("Cannot use '--lep' and '--combine' at the same time")
cat = 'leptonic' if arg.lep     else cat                   # Use 'leptonic' for ee and mumu channel combination
cat = 'combined' if arg.combine else cat                   # Use 'combined' for all channels combination

# Build process definitions for analysis
processes = mk_processes(ecm=ecm)

# Build subprocess command arguments for calling make_pseudo.py and fit.py
cmd_args = []
if arg.cat:
    cmd_args.extend(['--cat', f'{arg.cat}'])  # Category argument
if arg.lep:
    cmd_args.append('--lep')
if arg.combine:
    cmd_args.append('--combine')                # Combine channels flag
cmd_args.extend(['--sel', f'{sel}'])          # Selection strategy

# Add extra fit options from command-line if provided
if arg.extra:
    cmd_args.extend(f'--{ext}' for ext in arg.extra)

# Select cross-section scale: polarization or ILC configuration
scale = 'polL' if arg.polL else ('polR' if arg.polR else ('ILC' if arg.ILC else ''))

# Resolve directory paths for bias test results and input
inputdir   = loc.get('BIAS_FIT_RESULT', cat, ecm, sel)
loc_result = loc.get('BIAS_RESULT',     cat, ecm, sel)
nomDir     = loc.get('NOMINAL_RESULT',  cat, ecm, sel)
inDir      = loc.get('BIAS_DATACARD',   cat, ecm, sel)
h_inDir    = loc.get('HIST_PROCESSED',  cat, ecm, sel)



########################
### HELPER FUNCTIONS ###
########################

def _setup_cache() -> None:
    # Preload histograms and xsec caches once to minimize repeated I/O
    # Determine processed histogram directory

    # Process labels used across plots and pseudo-data
    procs_labels = ['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare']
    processes = mk_processes(procs_labels, ecm=ecm)

    # Flatten actual sample names for caching
    samples = []
    for p in procs_labels:
        samples.extend(processes[p])

    # Preload the most used histograms
    hNames = ('zqq_fit',) if cat=='qq' else ('zll_recoil_m',)
    LOGGER.debug('Preloading histograms and cross-section before bias loop')
    preload_histograms(samples, h_inDir, hNames=hNames, rmww=True)

    # Warm up xsec cache for both rmww variants to avoid repeated computations in downstream calls
    for s in samples:
        _ = getMetaInfo(s, rmww=False)
        _ = getMetaInfo(s, rmww=True)



#################
### FUNCTIONS ###
#################

def run_fit(target: str,
            pert: float,
            cmd_args: list[str],
            cat: str,
            ecm: int,
            sel: str,
            ) -> float:
    """Generate pseudodata and run pseudodata fit for a Higgs decay mode.

    Calls pseudodata generation directly (shares cached histograms),
    then runs fit via subprocess.
    """
    # Generate pseudodata and datacard using cached histograms (same process)
    tot = 'tot' not in arg.extra
    decays = h_decays if target!='inv' else H_decays
    if not arg.combine:
        pseudo_datacard(
            h_inDir, inDir,
            cat, ecm, target, pert,
            z_decays, decays,
            processes,
            tot=tot, scales=scale,
            freeze=arg.freeze, float_bkg=arg.float
        )

    # Now run the fit via subprocess (fit.py only, datacard already exists)
    cmd = ['python', '5-Fit/fit.py', '--bias', '--target', target, '--no-timer',
           '--pert', str(pert), '--ecm', str(ecm), '--no-print'] + cmd_args

    result = subprocess.run(cmd, check=False, capture_output=False, text=True, env=os.environ.copy())
    if result.returncode != 0:
        LOGGER.error(f'Fit subprocess failed (exit {result.returncode}) while running {cmd}')
        sys.exit(result.returncode)

    # Extract and return the fitted signal strength
    inputdir = loc.get('BIAS_FIT_RESULT', cat, ecm, sel)
    mu = np.loadtxt(f'{inputdir}/results_{target}.txt')[0]
    return mu

def get_bias(inDir: str,
             outDir: str,
             h_decays: list[str],
             cat: str,
             sel: str,
             pert: float,
             cmd_args: list[str],
             ecm: int = 240,
             lumi: float = 10.8
             ) -> tuple[pd.DataFrame,
                        list[float]]:
    """Run bias test for all Higgs decay modes and generate pseudodata plots."""

    # Initialize bias list for each decay mode
    bias = [0.0] * len(h_decays)

    if not arg.combine: _setup_cache()

    for idx, h_decay in enumerate(h_decays):
        LOGGER.info(f'Running fit for {h_decay} channel')

        # Run fit and compute bias as percentage difference from prior
        # Pass cat, ecm, sel, proc_scales to run_fit for cached histogram access
        mu = run_fit(h_decay, pert, cmd_args, cat, ecm, sel)
        bias[idx] = 100 * (mu - pert)
        LOGGER.info(f'Bias obtained: {bias[idx]:.3f}\n')

        # Generate pseudodata ratio plots (only for single channel, not combined)
        if not arg.combine:
            LOGGER.info('Making plots for pseudo-signal')
            args = {
                'inDir': inDir, 'outDir': outDir,
                'cat': cat, 'target': h_decay,
                'procs': ['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare'],
                'ecm': ecm, 'lumi': lumi, 'pert': pert
            }
            # Create plots for high and low selection regions
            for sel_suffix in ['_high', '_low']:
                sel_full = sel + sel_suffix
                PseudoRatio(**args, sel=sel_full)

    # Save bias results to CSV file
    LOGGER.info('Saving bias in a .csv file')
    df = pd.DataFrame({'mode':h_decays, 'bias':bias})

    result_csv = f'{loc_result}/bias_results.csv'
    df.to_csv(result_csv, index=False)
    LOGGER.info(f'Bias saved at {result_csv}')

    return df, bias


def bias_to_txt(outDir: str,
                bias: list[float],
                h_decays: list[str]
                ) -> None:
    """Format and save bias results to a text file with fixed-width columns."""

    LOGGER.info('Saving bias in a .txt file')

    out = f'{outDir}/bias_results.txt'
    ndecays, col_w = len(h_decays), 15

    # Create header with decay mode names
    header = f"{'Decay modes':<{col_w}}" + ''.join(f'{decay:<{col_w}}' for decay in h_decays)
    sep = '-' * col_w * (ndecays + 1)

    # Format bias values with different precisions
    bias_3dec = [f'{b:.3f}' for b in bias]
    bias_2dec = [f'{b:.2f}' for b in bias]

    # Create rows with formatted values
    row_3dec = f"{'Bias':<{col_w}}" + ''.join(f'{val:<{col_w}}' for val in bias_3dec)
    row_2dec = f"{'Bias':<{col_w}}" + ''.join(f'{val:<{col_w}}' for val in bias_2dec)

    # Write formatted table to file
    with open(out, 'w') as f:
        for row in [header, sep, row_3dec, row_2dec]:
            f.write(row + '\n')

    LOGGER.info(f'Bias saved at {out}')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Create output directory
        mkdir(loc_result)

        # Run bias test for all decay modes
        df, bias = get_bias(
            inDir, loc_result, H_decays,
            cat, sel, pert, cmd_args,
            ecm=ecm, lumi=lumi
        )

        # Generate bias summary plots
        LOGGER.info('Making plot of the bias')
        Bias(df, nomDir, loc_result, H_decays, ecm=ecm, lumi=lumi)

        # Save bias results to formatted text file
        bias_to_txt(loc_result, bias, H_decays)

        clear_histogram_cache()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)

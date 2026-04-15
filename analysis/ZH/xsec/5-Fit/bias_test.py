#################################
### IMPORT STANDARD LIBRARIES ###
#################################

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
    cat_single=True,
    allow_empty=True,
    include_sel=True,
    fit=True,
    bias=True,
    bias_extra=True,
    polarization=True,
    target='bb',
    pert=1.05
)
arg = parse_args(parser, comb=True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
from package.config import (
    timer, mk_processes,
    z_decays, h_decays, H_decays,
)
from package.plots.plotting import Bias, PseudoRatio
from package.tools.utils import mkdir
from package.tools.process import preload_histograms, clear_histogram_cache, getMetaInfo
from package.func.bias import pseudo_datacard



###############################
### CONFIGURATION AND SETUP ###
###############################

# Bundle parsed arguments for configuration
cat, ecm, sel, pert = arg.cat, arg.ecm, arg.sel, arg.pert
lumi = 10.8 if ecm==240 else (3.1 if ecm==365 else -1)
cat = 'combined' if arg.combine else cat
processes = mk_processes(ecm=ecm)

# Build command arguments for subprocess calls
cmd_args = []
if arg.cat:
    cmd_args.extend(['--cat', f'{arg.cat}'])
if arg.combine:
    cmd_args.append('--combine')
cmd_args.extend(['--sel', f'{sel}'])

# Add extra fit options if provided
if arg.extra:
    cmd_args.extend(f'--{ext}' for ext in arg.extra)

scale = 'polL' if arg.polL else ('polR' if arg.polR else ('ILC' if arg.ILC else ''))

# Resolve directory paths for input/output
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
    sig_label = ('ZH' if 'tot' not in arg.extra else f'Z{cat}H')
    procs_labels = [sig_label, 'WW', 'ZZ', 'Zgamma', 'Rare']
    processes = mk_processes(procs_labels, ecm=ecm)

    # Flatten actual sample names for caching
    samples = []
    for p in procs_labels:
        samples.extend(processes[p])

    # Preload the most used histograms
    hNames = ('zll_recoil_m',)
    LOGGER.info('Preloading histograms and cross-section before bias loop')
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
    pseudo_datacard(
        h_inDir, inDir,
        cat, ecm, target, pert,
        z_decays, decays,
        processes,
        tot=tot, scales=scale,
        freeze=arg.freeze, float_bkg=arg.float, plot_dc=arg.plot_dc
    )

    # Now run the fit via subprocess (fit.py only, datacard already exists)
    cmd = ['python3', '5-Fit/fit.py', '--bias', '--target', target,
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

    _setup_cache()

    for idx, h_decay in enumerate(h_decays):
        LOGGER.info(f'Running fit for {h_decay} channel')

        # Run fit and compute bias as percentage difference from prior
        # Pass cat, ecm, sel, proc_scales to run_fit for cached histogram access
        mu = run_fit(h_decay, pert, cmd_args, cat, ecm, sel)
        bias[idx] = 100 * (mu - pert)
        LOGGER.info(f'Bias obtained: {bias[idx]:.3f}')

        # Generate pseudodata ratio plots (only for single channel, not combined)
        if not arg.combine:
            LOGGER.info('Making plots for pseudo-signal')
            args = {
                'inDir': inDir, 'outDir': outDir,
                'cat': cat, 'target': h_decay,
                'procs': ['ZH' if 'tot' not in arg.extra else f'Z{cat}H',
                          'WW', 'ZZ', 'Zgamma', 'Rare'],
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
    # Create output directory
    mkdir(loc_result)

    # Run bias test for all decay modes
    df, bias = get_bias(
        inDir, h_inDir, loc_result, H_decays,
        cat, sel, pert, cmd_args,
        ecm=ecm, lumi=lumi
    )

    # Generate bias summary plots
    LOGGER.info('Making plot of the bias')
    Bias(df, nomDir, loc_result, H_decays, ecm=ecm, lumi=lumi)

    # Save bias results to formatted text file
    bias_to_txt(loc_result, bias, H_decays)

    clear_histogram_cache()

    # Print execution time
    timer(t)

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys, subprocess

import numpy as np
import pandas as pd

from time import time
from argparse import ArgumentParser

# Start timer for performance tracking
t = time()

from package.userConfig import loc, get_loc

from package.config import (
    timer, warning, 
    z_decays, h_decays, H_decays,
    mk_processes,
)
from package.plots.plotting import Bias, PseudoRatio
from package.tools.utils import mkdir
from package.tools.process import preload_histograms, clear_histogram_cache, getMetaInfo
from package.func.bias import pseudo_datacard



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
# Define final state: ee or mumu
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
# Define selection strategy
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

# Bias test parameters
parser.add_argument('--pert',    help='Prior uncertainty on ZH cross-section used for the bias test', 
                    type=float, default=1.05)

# Fit execution flags
parser.add_argument('--freeze',   help='Freeze backgrounds', action='store_true')
parser.add_argument('--float',    help='Float backgrounds',  action='store_true')
parser.add_argument('--plot_dc',  help='Plot datacard',      action='store_true')

# Polarization and luminosity scaling
parser.add_argument('--polL', help='Scale to left polarization',  action='store_true')
parser.add_argument('--polR', help='Scale to right polarization', action='store_true')
parser.add_argument('--ILC',  help='Scale to ILC luminosity',     action='store_true')

# Additional fit options (freeze, float, plot_dc, polarization, etc.)
parser.add_argument('--extra',   help='Extra argument for the fit', 
                    choices=['tot', 'onlyrun', 't'], 
                    nargs='*',  default=[])
# Combine channels option
parser.add_argument('--combine', '--comb', help='Combine the channel to do the fit', action='store_true')
arg = parser.parse_args()

# Validate input: require either a final state or combine option
if arg.cat=='' and not arg.combine:
    msg = 'Final state or combine were not selected, please select one to run this code'
    warning(msg)



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
inputdir   = get_loc(loc.BIAS_FIT_RESULT, cat, ecm, sel)
loc_result = get_loc(loc.BIAS_RESULT,     cat, ecm, sel)
nomDir     = get_loc(loc.NOMINAL_RESULT,  cat, ecm, sel)
inDir      = get_loc(loc.BIAS_DATACARD,   cat, ecm, sel)
h_inDir    = get_loc(loc.HIST_PROCESSED,  cat, ecm, sel)



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
    print('----->[Info] Preloading histograms and xsecs before bias loop')
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
           '--pert', str(pert), '--ecm', str(ecm), '--noprint'] + cmd_args
    
    result = subprocess.run(cmd, check=False, capture_output=False, text=True, env=os.environ.copy())
    if result.returncode != 0:
        print(f"----->[Error] Fit subprocess failed (exit {result.returncode}) while running {cmd}")
        sys.exit(result.returncode)

    # Extract and return the fitted signal strength
    inputdir = get_loc(loc.BIAS_FIT_RESULT, cat, ecm, sel)
    mu = np.loadtxt(f'{inputdir}/results_{target}.txt')[0]
    return mu

def get_bias(inDir: str, 
             h_inDir: str,
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
        print(f'----->[Info] Running fit for {h_decay} channel')

        # Run fit and compute bias as percentage difference from prior
        # Pass cat, ecm, sel, proc_scales to run_fit for cached histogram access
        mu = run_fit(h_decay, pert, cmd_args, cat, ecm, sel)
        bias[idx] = 100 * (mu - pert)

        print(f'\n----->[Info] Bias obtained: {bias[idx]:.3f}\n')

        # Generate pseudodata ratio plots (only for single channel, not combined)
        if not arg.combine:
            print('----->[Info] Making plots for pseudo-signal')
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
    print(f'----->[Info] Saving bias in a .csv file')
    df = pd.DataFrame({'mode':h_decays, 'bias':bias})

    result_csv = f'{loc_result}/bias_results.csv'
    df.to_csv(result_csv, index=False)
    print(f'----->[Info] Bias saved at {result_csv}')

    return df, bias


def bias_to_txt(outDir: str, 
                bias: list[float], 
                h_decays: list[str]
                ) -> None:
    """Format and save bias results to a text file with fixed-width columns."""
    
    print(f'----->[Info] Saving bias in a .txt file')

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
    
    print(f'----->[Info] Bias saved at {out}')



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
    print(f'----->[Info] Making plot of the bias')
    Bias(df, nomDir, loc_result, H_decays, ecm=ecm, lumi=lumi)

    # Save bias results to formatted text file
    bias_to_txt(loc_result, bias, H_decays)

    clear_histogram_cache()
    
    # Print execution time
    timer(t)

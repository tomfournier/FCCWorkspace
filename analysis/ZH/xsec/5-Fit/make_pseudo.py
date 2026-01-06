##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, subprocess, sys

from time import time
from ROOT import TFile
from argparse import ArgumentParser

# Start timing for performance tracking
t = time()

from package.userConfig import loc, get_loc
from package.config import (
    timer, warning, 
    mk_processes, 
    z_decays, 
    h_decays, 
    H_decays
)
from package.tools.utils import mkdir
from package.tools.process import getHist
from package.func.bias import make_pseudodata, make_datacard



########################
### ARGUMENT PARSING ###
########################

# Initialize argument parser
parser = ArgumentParser()

# Final state selection
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
# Collision energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
# Selection criteria for histogram fitting
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

# Combine channels for joint fit
parser.add_argument('--combine', help='Combine the channel to do the fit', action='store_true')

# Target Higgs decay mode for pseudodata
parser.add_argument('--target',  help='Target pseudodata', 
                    type=str, default='bb')
# Scaling factor for pseudodata
parser.add_argument('--pert',    help='Target pseudodata size', 
                    type=float, default=1.0)
# Use all Z decays for cross-section calculation
parser.add_argument('--tot',     help='Do not consider all Z decays for making cross-section',
                    action='store_true')

# Fit execution flags
parser.add_argument('--onlyrun', help='Only run the fit',   action='store_true')
parser.add_argument('--run',      help='Run combine',        action='store_true')
parser.add_argument('--freeze',   help='Freeze backgrounds', action='store_true')
parser.add_argument('--float',    help='Float backgrounds',  action='store_true')
parser.add_argument('--plot_dc',  help='Plot datacard',      action='store_true')

# Polarization and luminosity scaling
parser.add_argument('--polL', help='Scale to left polarization',  action='store_true')
parser.add_argument('--polR', help='Scale to right polarization', action='store_true')
parser.add_argument('--ILC',  help='Scale to ILC luminosity',     action='store_true')

# Performance timing
parser.add_argument('--t', help='Compute the elapsed time to run the code', action='store_true')
arg = parser.parse_args()

# Validate that either a category or combine mode is selected
if arg.cat=='' and not arg.combine:
    msg = 'Final state or combine were not selected, please select one to run this code'
    warning(msg)



###############################
### CONFIGURATION AND SETUP ###
###############################

# Parse configuration parameters
cat, ecm, sel, tot = arg.cat, arg.ecm, arg.sel, not arg.tot

# Set process-wise scale factors based on polarization or luminosity
if arg.ILC: ## change fit to ASIMOV -t -1 !!!
    proc_scales  = {'ZH': 1.048, 'WW': 0.971, 'ZZ': 0.939, 'Zgamma': 0.919}
elif arg.polL:
    proc_scales  = {'ZH': 1.554, 'WW': 2.166, 'ZZ': 1.330, 'Zgamma': 1.263}
elif arg.polR:
    procs_scales = {'ZH': 1.047, 'WW': 0.219, 'ZZ': 1.011, 'Zgamma': 1.018}
else:
    proc_scales  = {}

# Set input and output directories
inDir  = get_loc(loc.HIST_PROCESSED, cat, ecm, sel)
outDir = get_loc(loc.BIAS_DATACARD,  cat, ecm, sel)

# Define histogram names and categories
hNames, categories = (f'zll_recoil_m',), (f'z_{cat}',)

# List of processes (signal first, then backgrounds)
procs = ['ZH' if tot else f'Z{cat}H', 'WW', 'ZZ', 'Zgamma', 'Rare']
processes = mk_processes(procs, ecm=ecm)

# Select Higgs decay modes (invisible or visible)
decays = H_decays if arg.target=='inv' else h_decays



######################
### MAIN EXECUTION ###
######################

# Process histograms and create pseudodata 
# (unless only running fit or combining)
if not arg.combine and not arg.onlyrun:
    hists = []

    # Loop over categories and processes to collect histograms
    for i, categorie in enumerate(categories):
        for proc in procs:
            h = getHist(hNames[i], processes[proc], inDir)
            h.SetName(f'{categorie}_{proc}')
            hists.append(h)

        # Generate pseudodata histogram
        args = [inDir, procs, processes, hNames[i], arg.target, cat, z_decays]
        args.append(decays) if arg.target=='inv' else args.append(decays)

        hist_pseudo = make_pseudodata(
            hNames[i], inDir, procs, processes, cat, z_decays, decays,
            arg.target, ecm=ecm, variation=arg.pert, tot=tot,
            proc_scales=proc_scales
        )
        hist_pseudo.SetName(f'{categorie}_data_{arg.target}')
        hists.append(hist_pseudo)

    # Create output directory and save histograms
    mkdir(outDir)

    print('----->[Info] Saving pseudo histograms')
    fOut = f'{outDir}/datacard_{arg.target}.root'

    with TFile(fOut, 'RECREATE') as f:
        for hist in hists:
            hist.Write()

    print(f'----->[Info] Histograms saved in {fOut}')

    # Generate datacard for combine fit
    print('----->[Info] Making datacard')
    make_datacard(outDir, procs, arg.target, 1.01, categories,
                  freezeBkgs=arg.freeze, floatBkgs=arg.float, 
                  plot_dc=arg.plot_dc)



#####################
### FIT EXECUTION ###
#####################

# Run combine fit if requested
if arg.run:
    # Build command to execute fit.py
    cmd = ['python3', '5-Fit/fit.py']

    # Add category argument if specified
    if arg.cat:
        cmd.extend(['--cat', arg.cat])
    
    # Add common fit arguments
    cmd.extend([
        '--bias',
        '--target', arg.target,
        '--pert', str(arg.pert),
        '--sel', arg.sel,
        '--ecm', str(ecm),
        '--noprint'
    ])

    # Add combine flag if channels are combined
    if arg.combine:
        cmd.append('--combine')

    # Execute fit command with error handling
    try:
        show_cmd = ' '.join(cmd)
        print(f"----->[Info] Running fit command: {show_cmd}")
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=False,
            text=True,
            env=os.environ.copy()
        )
    except FileNotFoundError:
        print(f'----->[Error] Could not find python or 5-Fit/fit.py')
        sys.exit(1)

# Print elapsed time if requested
if __name__=='__main__' and arg.t:
    timer(t)

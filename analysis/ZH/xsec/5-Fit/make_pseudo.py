#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import os, sys, time, subprocess

# Start timing for performance tracking
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import ArgumentParser, create_parser, parse_args, set_log
from package.logger import get_logger
parser: ArgumentParser = create_parser(
    cat_single=True,
    allow_empty=True,
    include_sel=True,
    fit=True,
    bias=True,
    polarization=True,
    target='bb',
    description='Pseudo-data Script'
)
# Use all Z decays for cross-section calculation
parser.add_argument('--tot', help='Do not consider all Z decays for making cross-section', action='store_true')

# To know if make_pseudo.py is ran from bias_test.py
parser.add_argument('--nobias', help='Do not run from bias_test.py', action='store_true')

# Fit execution flags
parser.add_argument('--onlyrun', help='Only run the fit', action='store_true')
parser.add_argument('--run',     help='Run the fit',      action='store_true')
arg = parse_args(parser, comb=True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
from package.config import (
    timer, mk_processes,
    z_decays,
    h_decays,
    H_decays
)
from package.func.bias import pseudo_datacard



###############################
### CONFIGURATION AND SETUP ###
###############################

# Parse configuration parameters
cat, ecm, sel, tot = arg.cat, arg.ecm, arg.sel, not arg.tot

# Set input and output directories
inDir  = loc.get('HIST_PROCESSED', cat, ecm, sel)
outDir = loc.get('BIAS_DATACARD',  cat, ecm, sel)

# Define histogram names and categories
hNames, categories = ('zll_recoil_m',), (f'z_{cat}',)

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
if not arg.combine and not arg.onlyrun and not arg.nobias:
    scales = 'ILC' if arg.ILC else \
                ('polL' if arg.polL else
                    ('polR' if arg.polR else ''))
    pseudo_datacard(
        inDir, outDir,
        cat=cat, ecm=ecm,
        target=arg.target, pert=arg.pert,
        z_decays=z_decays, h_decays=decays,
        processes=processes,
        tot=tot,
        freeze=arg.freeze,
        float_bkg=arg.float,
        plot_dc=arg.plot_dc
    )



#####################
### FIT EXECUTION ###
#####################

# Run combine fit if requested
if arg.run or arg.onlyrun:
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
        LOGGER.info(f'Running fit command: {show_cmd}')
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=False,
            text=True,
            env=os.environ.copy()
        )
    except Exception as e:
        LOGGER.error(f"Error during fit execution: {e}")
        sys.exit(1)

# Print elapsed time if requested
if __name__=='__main__' and arg.t:
    timer(t)

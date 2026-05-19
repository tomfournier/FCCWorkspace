################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, time, subprocess

# Start timing for performance tracking
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import ArgumentParser, create_parser, parse_args, set_log
from package.logger import get_logger
parser: ArgumentParser = create_parser(
    cat_single=True,       # Support single decay category
    allow_empty=True,      # Allow empty category
    include_sel=True,      # Include selection strategy options
    fit=True,              # Include fit-specific options
    bias=True,             # Include bias test options
    polarization=True,     # Include polarization/scale options
    target='bb',           # Default Higgs decay mode
    description='Pseudo-data Script'
)
# Use all Z decays for cross-section calculation in pseudo-data generation
parser.add_argument('--tot', help='Do not consider all Z decays for cross-section', action='store_true')

# Flag to indicate if called from bias_test.py (affects working directory)
parser.add_argument('--nobias', help='Do not run from bias_test.py', action='store_true')

# Fit execution flags
parser.add_argument('--onlyrun', help='Only run the fit without pre-processing', action='store_true')
parser.add_argument('--run',     help='Run the fit after setup',            action='store_true')

arg = parse_args(parser, comb=True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory configuration and process definitions
from package.userConfig import loc
from package.config import (
    timer,              # Timing utility
    mk_processes,       # Build process definitions
    z_decays,           # Z boson decay modes
    h_decays,           # Higgs decay modes (visible only)
    H_decays            # Higgs decay modes (all including invisible)
)
from package.func.bias import pseudo_datacard  # Pseudo-datacard generation utilities



###############################
### CONFIGURATION AND SETUP ###
###############################

# Parse and extract configuration parameters
cat, ecm, sel, tot = arg.cat, arg.ecm, arg.sel, not arg.tot  # Include all Z decays if --tot not specified

# Define input/output directories
inDir  = loc.get('HIST_PROCESSED', cat, ecm, sel)  # Input: processed histograms
outDir = loc.get('BIAS_DATACARD',  cat, ecm, sel)  # Output: pseudo-data datacards

# Define observable for pseudo-data generation
hNames, categories = ('zll_recoil_m',), (f'z_{cat}',)  # Recoil mass histogram

# List of physics processes (signal first, then backgrounds)
# Signal: ZH production with all Higgs decay modes
# Backgrounds: WW, ZZ, Drell-Yan, and rare processes
procs = ['ZH' if tot else f'Z{cat}H', 'WW', 'ZZ', 'Zgamma', 'Rare']
processes = mk_processes(procs, ecm=ecm)

# Select Higgs decay modes: all modes for invisible target, visible only for other targets
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
if __name__=='__main__' and (arg.run or arg.onlyrun):
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
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
        sys.exit(1)
    finally:
        # Print elapsed time if requested
        if arg.timer: timer(t)

'''Wrapper to run the BDT pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu/qq), energy (240/365 GeV), and pipeline stages.
- Batch execution of BDT stages across energies and channels while streaming output.
- Optional toggles to skip metrics plots, draw trees, or check variable distributions.

Conventions:
- Scripts are executed in nested loops: ecm -> cat -> stage-specific script.
- Paths are built from loc.ROOT/2-BDT to match the repository layout.

Usage:
    python 2-run.py                               # Default: all channels, all ecms, stages 1-2-3
    python 2-run.py --cat ee --ecm 365 --run 1-2  # Process + train for ee at 365
    python 2-run.py --cat ee-mumu --ecm 240-365   # Multiple channels and energies
'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, time, subprocess

# Start execution timer
t = time.time()



#######################
## ARGUMENT PARSING ###
#######################

from parsing import set_log
from zh_xsec.parsing import create_parser
from logger import get_logger               # Logging setup
arg = create_parser('2-BDT').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and timing utility
from zh_xsec.userConfig import loc  # Directory path configuration
from tools.utils import timer    # Execution timing utility
from run import log_msg, update_namespace, get_extra_args



################################
### SCRIPT MAP CONFIGURATION ###
################################

# Map pipeline stage numbers to analysis script names
script_map = {
    '1': 'process_input',  # Stage 1: Load histograms, balance samples, prepare for BDT training
    '2': 'train_bdt',      # Stage 2: Train XGBoost classifier with early stopping
    '3': 'evaluation'      # Stage 3: Evaluate BDT performance, generate plots
}

# Map to associate script to command (must match script_map values)
cmds = {v: 'python' for v in script_map.values()}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Expand dash-separated channel and energy values into lists.
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for BDT analysis scripts
path = f'{loc.ROOT}/2-BDT'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def main(cat: str, ecm: int, script: str) -> None:
    '''Execute one BDT stage with streaming output.

    Builds the downstream command from the selected parser configuration,
    overrides the current channel and energy, and forwards the resulting
    arguments to the subprocess while piping stdout and stderr to the terminal.

    Args:
        cat (str): Lepton channel identifier ('ee', 'mumu' or 'qq').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        script (str): Stage script name ('pre-selection', 'final-selection', or 'plots').

    Returns:
        int: Return code from the subprocess.
    '''

    # Log the stage context before launching the subprocess.
    log_msg('▶ STARTING', script, cat=cat, ecm=ecm)

    stage_args = update_namespace(arg, cat=cat, ecm=ecm)
    parser = {'directory': '2-BDT', 'script': script}
    extra_args = get_extra_args(stage_args, parser, create_parser)
    result = subprocess.run(cmds[script].split() + [f'{path}/{script}.py'] + extra_args,
                            env=ENV, stdout=sys.stdout, stderr=sys.stderr)

    # Log completion status without changing the subprocess return code.
    status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
    log_msg(status, script, cat=cat, ecm=ecm)

    return result.returncode


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    try:
        # Nested loops: iterate over energies, channels, and pipeline stages
        for ecm in ecms:
            for cat in cats:
                for script in scripts:
                    result = main(cat, ecm, script)
                    if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

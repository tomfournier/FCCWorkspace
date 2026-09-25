'''Wrapper to run the selection/plotting pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Automatic forwarding of arguments supported by each downstream script.
- Batch execution across energies and channels while streaming child output.

Conventions:
- Each stage receives its channel and energy through command-line arguments.
- Stage-specific arguments are selected from the downstream parser configuration.
- Scripts are executed in nested loops: energy -> channel -> stage.
- Paths are rooted at loc.ROOT/1-MVAInputs to match the repository layout.

Usage:
    python 1-run.py                               # Default: all channels, all ecms, stages 2-3
    python 1-run.py --cat ee --ecm 365 --run 1-2  # Pre + final selection for ee at 365
    python 1-run.py --cat ee-mumu --ecm 240-365   # Multiple channels and energies
'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, time, subprocess

# Start the total execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger  # Logging setup
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories (--cat ee-mumu)
    ecm_multi=True,        # Support multiple energies (--ecm 240-365)
    include_sels=True,     # Include selection strategy options
    sel_default='all',
    run_stages=3,          # MVA pipeline has 3 stages: pre + final + plots
    add_test=True,         # Add test sample option
    batch=True,            # Include batch execution options
    presel=True,
    training=True,
    is_final=True,
    is_presel_plot=True,
    description='Run MVA Inputs pipeline'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and timing utility
from package.userConfig import loc         # Directory path configuration
from package.config import timer           # Execution timing utility
from package.run import log_msg, update_namespace, get_extra_args



################################
### SCRIPT MAP CONFIGURATION ###
################################

# Map to associate script to command
cmds = {'pre-selection':   'fccanalysis run',
        'final-selection': 'fccanalysis final',
        'plots':           'fccanalysis plots'}

# Parser configurations for the downstream scripts. These dictionaries contain
# the create_parser options needed to describe each stage's arguments.
script_parsers = {
    'pre-selection':   {'cat_single': True, 'presel': True, 'batch': True, 'training': True},
    'final-selection': {'cat_single': True, 'presel': True, 'include_sels': True, 'sel_default': 'all', 'is_final': True},
    'plots':           {'cat_single': True, 'presel': True, 'include_sels': True, 'sel_default': 'all', 'is_presel_plot': True},
}

# Map pipeline stage numbers to analysis script names.
script_map = {
    '1': 'pre-selection',      # Stage 1: Apply pre-selection cuts, compute kinematic variables
    '2': 'final-selection',    # Stage 2: Fill histograms with BDT variables
    '3': 'plots'               # Stage 3: Generate distribution plots
}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Expand dash-separated channel and energy values into lists.
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for MVA analysis scripts
path = f'{loc.ROOT}/1-MVAInputs'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def main(cat: str, ecm: int, path: str, script: str) -> int:
    '''Execute one pipeline stage and stream its output.

    Builds the downstream command from the selected parser configuration,
    overrides the current channel and energy, and forwards the resulting
    arguments to the fccanalysis subprocess while piping stdout and stderr
    to the terminal.

    Args:
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('pre-selection', 'final-selection', or 'plots').

    Returns:
        int: Return code from the subprocess.
    '''

    # Log the stage context before launching the subprocess.
    log_msg('▶ STARTING', script, cat=cat, ecm=ecm)

    # Forward only arguments supported by the selected downstream parser.
    stage_args = update_namespace(arg, cat=cat, ecm=ecm)
    extra_args = get_extra_args(stage_args, script_parsers[script])
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
                    result = main(cat, ecm, path, script)
                    if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

'''Wrapper to run the FSR (Final State Radiation) optimization pipeline with automated parameters.

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
    python a-run.py                                 # Default: all channels, all ecms, stages 1-2
    python a-run.py --cat ee --ecm 365 --run 1      # Pre-selection only
    python a-run.py --cat ee-mumu --ecm 240-365     # Multiple channels and energies
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

from parsing import set_log
from zh_others.parsing import create_parser
from logger import get_logger
arg = create_parser('a-FSR').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and timing utility
from zh_others.userConfig import loc
from tools.utils import timer
from run import log_msg, update_namespace, get_extra_args



###############################
### SCRIT MAP CONFIGURATION ###
###############################

# Map pipeline stage numbers to analysis script names.
script_map = {
    '1': 'pre-selection',  # Stage 1: Apply pre-selection cuts, compute kinematic variables
    '2': 'plots'           # Stage 2: Generate FSR-related distribution plots
}

# Map to associate script to command (must match script_map values)
cmds = {'pre-selection': 'fccanalysis run', 'plots': 'python'}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for FSR analysis scripts
path = f'{loc.ROOT}/a-FSR'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def main(cat: str, ecm: int, script: str) -> None:
    '''Execute one pipeline stage and stream its output.

    Args:
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        script (str): Stage script name ('pre-selection', 'plots').

    Returns:
        int: Return code from the subprocess.
    '''

    # Log the stage context before launching the subprocess.
    log_msg('▶ STARTING', script, cat=cat, ecm=ecm)

    # forward only the arguments supported by the selected downstream parser.
    stage_args = update_namespace(arg, cat=cat, ecm=ecm)
    parser = {'directory': 'a-FSR', 'script': script}
    extra_args = get_extra_args(stage_args, parser, create_parser)
    result = subprocess.run(cmds[script].split() + [f'{path}/{script}.py'] + extra_args,
                            env=ENV, stdout=sys.stdout, stderr=sys.stderr)

    # Log completion status.
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

'''Wrapper to run the measurement pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Temporary config JSON per job to pass cat/ecm/sel to downstream scripts.
- Batch execution across energies and channels while streaming child output.

Conventions:
- Temporary configuration files are created in loc.RUN and removed after each stage.
- Environment variable RUN='1' flags automated mode for the analysis scripts.
- Scripts are executed in nested loops: ecm -> optional process_histogram -> cat/sel -> combine.

Usage:
    python 4-run.py                              # Default: all channels, all ecms, combine only
    python 4-run.py --cat ee --ecm 365 --run 1-2 # process_histogram then combine for ee at 365
    python 4-run.py --cat ee-mumu --ecm 240-365  # Multiple channels and energies
'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, time, subprocess

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_xsec.parsing import create_parser
from logger import get_logger               # Logging setup
arg = create_parser('4-Combine').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and utilities
from zh_xsec.userConfig import loc           # Directory path configuration
from utilities import timer             # Execution timing utility
from zh_xsec.run import log_msg, update_namespace, get_extra_args



################################
### SCRIPT MAP CONFIGURATION ###
################################

# Map pipeline stage numbers to analysis script names.
script_map = {
    '1': 'process_histogram',  # Stage 1: Split histograms into high/low BDT score regions
    '2': 'combine'             # Stage 2: Create combine datacards from processed histograms
}

# Map to associate script to command (must match script_map values)
cmds = {v:'python' for v in script_map.values()}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Expand dash-separated channel and energy values into lists.
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for Combine analysis scripts
path = f'{loc.ROOT}/4-Combine'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def main(cat: str, ecm: int, script: str) -> None:
    '''Execute one measurement stage and stream its output.

    Builds the downstream command from the selected parser configuration,
    overrides the current channel and energy, and forwards the resulting
    arguments to the fccanalysis subprocess while piping stdout and stderr
    to the terminal.

    Args:
        cat (str): Lepton channel identifier ('ee', 'mumu' or 'qq').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        script (str): Stage script name ('pre-selection', 'final-selection', or 'plots').

    Returns:
        int: Return code from the subprocess.
    '''

    # Log the stage context before launching the subprocess.
    log_msg('▶ STARTING', script, cat=cat, ecm=ecm)

    # Forward only arguments supported by the selected downstream parser.
    stage_args = update_namespace(arg, ecm=ecm)
    extra_args = get_extra_args(stage_args, {'directory': '4-Combine', 'script': script})
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
        for ecm in ecms:
            for script in scripts:
                result = main(arg.cat, ecm, script)
                if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

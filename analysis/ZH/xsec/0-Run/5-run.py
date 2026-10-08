'''Wrapper to run the fit + bias-test pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), pipeline stages
- Sequential execution across energies, channels, and selections
- Combined fits, timing, quiet mode, and bias-test options

Conventions:
- Nested loops order: ecm -> cat -> selection -> stage script
- Paths built from loc.ROOT/5-Fit matching repository layout

Usage:
    python 5-run.py                               # All channels/ecms, bias_test
    python 5-run.py --cat ee --ecm 365 --run 1-2  # Fit + bias_test, ee @ 365
    python 5-run.py --cat ee-mumu --ecm 240-365   # Multi channel/energy
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

from zh_xsec.parsing import create_parser, set_log  # Argument parsing utilities
from logger import get_logger               # Logging setup
arg = create_parser('5-Fit').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and timing utility
from zh_xsec.userConfig import loc  # Directory path configuration
from utilities import timer    # Execution timing utility
from zh_xsec.run import log_msg, update_namespace, get_extra_args



################################
### SCRIPT MAP CONFIGURATION ###
################################

# Map pipeline stage number to analysis script names
script_map = {
    '1': 'fit',       # Stage 1: Run nominal fit
    '2': 'bias_test'  # Stage 2: Run bias test with pseudo-data
}

# Map to associate script to command (must match script_map values)
cmds = {v:'python' for v in script_map.values()}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Expand dash-separated channel and energy values into lists.
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]
sels = arg.sels.split('-')                     # Selections: ['Baseline'] or ['Baseline', 'test']

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for Fit analysis scripts
path = f'{loc.ROOT}/5-Fit'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def main(cat: str, ecm: int, sel: str, script: str) -> int:
    '''Execute one fit stage and stream its output.

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
    log_msg('▶ STARTING', script, cat=cat, ecm=ecm, sels=sel)

    # Forward only arguments supported by the selected downstream parser.
    stage_args = update_namespace(arg, cat=cat, ecm=ecm, sel=sel)
    extra_args = get_extra_args(stage_args, {'directory': '5-Fit', 'script': script})
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
            for cat in cats:
                for sel in sels:
                    for script in scripts:
                        result = main(cat, ecm, sel, script)
                        if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

'''Wrapper to run the measurement pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Temporary config JSON per job to pass cat/ecm/lumi to downstream scripts.
- Batch execution across energies and channels while streaming child output.

Conventions:
- Temporary configuration files are created in loc.RUN and removed after each stage.
- Environment variable RUN='1' flags automated mode for the analysis scripts.
- Scripts are executed in nested loops: ecm -> cat -> stage-specific script, then plots/cutflow.

Usage:
    python 3-run.py                                  # Default: all channels, all ecms, stages 2-3
    python 3-run.py --cat ee --ecm 365 --run 1-2-3-4 # All stages including cutflow
    python 3-run.py --cat ee-mumu --ecm 240-365      # Multiple channels and energies
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
from zh_xsec.logger import get_logger               # Logging setup
arg = create_parser('3-Measurement').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and timing utility
from zh_xsec.userConfig import loc  # Directory path configuration
from zh_xsec.config import timer    # Execution timing utility
from zh_xsec.run import log_msg, update_namespace, get_extra_args



################################
### SCRIPT MAP CONFIGURATION ###
################################

# Map pipeline stage number to analysis script names.
script_map = {
    '1': 'pre-selection',    # Stage 1: Apply pre-selection cuts, compute kinematic variables
    '2': 'final-selection',  # Stage 2: Fill measurement histograms with BDT scores
    '3': 'plots',            # Stage 3: Generate distribution plots
    '4': 'cutflow'           # Stage 4: Analyze event yield per cut stage
}

cmds = {'pre-selection':   'fccanalysis run',
        'final-selection': 'fccanalysis final',
        'plots':           'python',
        'cutflow':         'python'}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Expand dash-separated channel and energy values into lists.
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for Measurement analysis scripts
path = f'{loc.ROOT}/3-Measurement'

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
    stage_args = update_namespace(arg, cat=cat, ecm=ecm)
    extra_args = get_extra_args(stage_args, {'directory': '3-Measurement', 'script': script})
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
        is_there_plots   = 'plots' in scripts
        is_there_cutflow = 'cutflow' in scripts

        if is_there_plots:   scripts.remove('plots')
        if is_there_cutflow: scripts.remove('cutflow')

        # Nested loops: iterate over energies, channels, and pipeline stages
        for ecm in ecms:
            # BATCH info for pre/final-selection
            if ('pre-selection' in scripts) or ('final-selection' in scripts):
                for cat in cats:
                    for script in scripts:
                        result = main(cat, ecm, script)
                        if result != 0: sys.exit(result)

            # BATCH info for plots
            if is_there_plots:
                result = main(arg.cat, ecm, 'plots')
                if result != 0: sys.exit(result)
            # BATCH info for cutflow
            if is_there_cutflow:
                result = main(arg.cat, ecm, 'cutflow')
                if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

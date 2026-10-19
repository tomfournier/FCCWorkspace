'''Wrapper to run the measurement pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Temporary config JSON per job to pass cat/ecm/lumi to downstream scripts.
- Batch execution across energies and channels while streaming child output.

Conventions:
- Scripts are executed in nested loops: ecm -> cat -> stage-specific script, then plots/cutflow.

Usage:
    python 1-run.py                                  # Default: all channels, all ecms, stages 2-3
    python 1-run.py --cat ee --ecm 365 --run 1-2-3-4 # All stages including cutflow
    python 1-run.py --cat ee-mumu --ecm 240-365      # Multiple channels and energies
'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, time, subprocess

from pathlib import Path

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_mass.parsing import create_parser
from logger import get_logger
arg = create_parser('1-Selection').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_mass.userConfig import loc
from tools.utils import timer
from run import log_msg, update_namespace, get_extra_args



################################
### SCRIPT MAP CONFIGURATION ###
################################

# Map pipeline stage number to anlaysis script names.
script_map = {
    '1': 'pre-selection',    # Stage 1: Apply pre-selection cuts, compute kinematic variables
    '2': 'final-selection',  # Stage 2: Fill measurement histograms with BDT scores
    '3': 'plots',            # Stage 3: Generate distribution plots
    '4': 'cutflow'           # Stage 4: Analyze event yield per cut stage
}

# Map to associate script to command (must match script_map values)
cmds = {'pre-selection':   'fccanalysis run',
        'final-selection': 'fccanalysis final',
        'plots': 'python', 'cutflow': 'python'}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for Selection analysis scripts
path = f'{loc.ROOT}/1-Selection'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def run(cat: str, ecm: int, script: str) -> None:
    '''Execute one Selection pipeline and stream its output.
    
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
    parser = {'directory': '1-Selection', 'script': script}
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
                        result = run(cat, ecm, path, script)
                        if result != 0: sys.exit(result)

            # BATCH info for plots
            if is_there_plots:
                result = run(arg.cat, ecm, path, 'plots')
                if result != 0: sys.exit(result)
            # BATCH info for cutflow
            if is_there_cutflow:
                result = run(arg.cat, ecm, path, 'cutflow')
                if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

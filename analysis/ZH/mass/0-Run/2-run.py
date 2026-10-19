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
    python 2-run.py                              # Default: all channels, all ecms, combine only
    python 2-run.py --cat ee --ecm 365 --run 1-2 # process_histogram then combine for ee at 365
    python 2-run.py --cat ee-mumu --ecm 240-365  # Multiple channels and energies
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
from zh_mass.parsing import create_parser
from logger import get_logger
arg = create_parser('2-Selection').parse_args()
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

# Map pipeline stage numbers to analysis script names.
script_map = {
    '1': 'process_histogram',  # Stage 1: Process histograms before combine
    '2': 'combine',            # Stage 2: Create combine datacards from processed histograms
    '3': 'fit'                 # Stage 3: Run fit
}

# Map to associate script to command (must match script_map values)
cmds = {v:'python' for v in script_map.values()}



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')                      # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]  # Energies: [240] or [240, 365]
sels = arg.sels.split('-')

scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for Fit analysis scripts
path = f'{loc.ROOT}/2-Fit'

ENV = os.environ.copy()



##########################
### EXECUTION FUNCTION ###
##########################

def main(cat: str, ecm: int, sel: str, script: str) -> None:
    '''Execute one measurement stage with a temporary config and streamed output.

    Builds a JSON file with cat/ecm/sel, sets RUN=1, and calls the stage script
    via fccanalysis (or python for non-fccanalysis steps) while piping stdout/stderr
    through to the parent terminal.

    Args:
        cfg_dir (str): Directory where the temporary config file will be stored.
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('process_histogram' or 'combine').

    Returns:
        int: Return code from the subprocess.
    '''
    
    # Log the stage context before launching the subprocess.
    log_msg('▶ STARTING', script, cat=cat, ecm=ecm)

    # Forward only arguments supported by the selected downstream parser.
    if script in 'fit':
        stage_args = update_namespace(arg, cat=cat, ecm=ecm, sel=sel)
    else: stage_args = update_namespace(arg, ecm=ecm)
    parser = {'directory': '4-Combine', 'script': script}
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
                    if script in ['process_histogram', 'combine']:
                        result = main(cat, ecm, arg.sels, script)
                        if result != 0: sys.exit(result)
                    else:
                        for sel in sels:
                            result = main(cat, ecm, sel, script)
                            if result !=0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

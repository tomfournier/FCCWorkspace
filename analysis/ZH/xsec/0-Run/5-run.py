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

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys, time, subprocess

# Load directory path manager and timing utility
from package.userConfig import loc         # Directory path configuration
from package.config import timer           # Execution timing utility

# Start execution timer
t = time.time()

# Reuse environment without copying for each subprocess
ENV = os.environ  # Use same environment for all subprocesses

########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log  # Argument parsing utilities
from package.logger import get_logger               # Logging setup
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories (--cat ee-mumu)
    ecm_multi=True,        # Support multiple energies (--ecm 240-365)
    include_sels=True,     # Include selection strategy options
    run_stages=2,          # Fit pipeline has 2 stages: fit + bias_test
    fit=True,              # Include fit-specific options
    bias=True,             # Include bias test options
    bias_extra=True,       # Include extra bias test parameters
    polarization=True,     # Include polarization/scale options
    description='Run Fit pipeline'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



#############################
### SETUP CONFIG SETTINGS ###
#############################

def parse_channels(cat_arg: str, lep: bool, combine: bool) -> list[str]:
    '''Return channel list from CLI values.

    If `combine` is requested, include the combined fit channel ('comb').
    With an empty category string, only 'comb' is used.

    Args:
        cat_arg: Channel argument string (e.g., 'ee', 'ee-mumu')
        combine: Whether to include combined fit in the list

    Returns:
        List of channel identifiers to process
    '''
    cats_local = cat_arg.split('-') if cat_arg else []
    if lep:     cats_local.append('lep')
    if combine: cats_local.append('comb')
    return cats_local


def parse_ecms(ecm_arg: str) -> list[int]:
    '''Return list of center-of-mass energies as integers.

    Args:
        ecm_arg: Energy argument string (e.g., '240', '240-365')

    Returns:
        List of CoM energies as integers
    '''
    return [int(e) for e in ecm_arg.split('-')]


# Parse channel configuration and center-of-mass energies
cats = parse_channels(arg.cat, arg.lep, arg.combine)  # Includes 'comb' if --combine specified
ecms = parse_ecms(arg.ecm)                                        # Convert to list of integers

# Selection strategies to process (from command-line or defaults)
if arg.sels == '':
    sels = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']  # Default selections
else:
    sels = arg.sels.split('-')  # Parse from command-line


# Map pipeline stage identifiers to script names
SCRIPT_MAP = {
    '1': 'fit',        # Stage 1: Run nominal fit
    '2': 'bias_test'   # Stage 2: Run bias test with pseudo-data
}
scripts = [SCRIPT_MAP[s] for s in arg.run.split('-')]

# Base directory for fit scripts
BASE_PATH = f'{loc.ROOT}/5-Fit'



##########################
### EXECUTION FUNCTION ###
##########################

def run(cat: str, ecm: int, sel: str, script: str) -> int:
    '''Execute one fit stage with streaming output and clear markers.

    Builds command-line arguments for the selected stage and forwards
    execution to the target script while piping stdout/stderr to the
    terminal in real-time.

    Args:
        cat: Channel identifier ('ee', 'mumu', 'comb')
        ecm: Center-of-mass energy in GeV (240, 365)
        sel: Selection name (e.g., 'Baseline_sep')
        script: Stage name ('fit', 'bias_test')

    Returns:
        Return code from the subprocess for error handling.
    '''
    script_path = f'{BASE_PATH}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = } | {sel = }'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    # Build base arguments common to all stages
    cmd = [sys.executable, script_path, '--ecm', str(ecm), '--sel', sel]

    # Add channel or combine flag
    if cat == 'lep':
        cmd.append('--lep')
    elif cat == 'comb':
        cmd.append('--comb')
    else:
        cmd.extend(['--cat', cat])
    if arg.toy!=0:
        cmd.extend(['--toy', arg.toy])

    # Append stage-specific arguments
    if script == 'fit':
        cmd.append('--no-timer')
        if arg.print: cmd.append('--print')
    elif script == 'bias_test':
        cmd.extend(['--pert', str(arg.pert)])
        if arg.extra:
            cmd.extend(['--extra'] + arg.extra)

    # Execute with real-time streaming to terminal
    result = subprocess.run(cmd, env=ENV, stdout=sys.stdout,
                            stderr=sys.stderr)

    # Completion status marker
    status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
    msg = f'{status}: [{script}] {cat = } | {ecm = } | {sel = }'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    return result.returncode


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    try:
        for sel in sels:
            for script in scripts:
                for ecm in ecms:
                    for cat in cats:
                        result = run(cat, ecm, sel, script)
                        if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

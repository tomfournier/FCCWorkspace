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

from argparse import ArgumentParser

from package.userConfig import loc
from package.config import timer

t = time.time()

# Reuse environment without copying for each subprocess
ENV = os.environ

########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser(
    description='Run fit pipeline with automated parallel execution'
)
# Lepton final states: empty string only valid with --combine for direct combined fit
parser.add_argument(
    '--cat', type=str, default='ee-mumu',
    choices=['ee', 'mumu', 'ee-mumu', 'mumu-ee', ''],
    help='Final state: ee, mumu, or both (default: ee-mumu)'
)
# Center-of-mass energy: dash-separated for multiple energies
parser.add_argument(
    '--ecm', type=str, default='240-365',
    choices=['240', '365', '240-365', '365-240'],
    help='Center-of-mass energy in GeV (default: 240-365)'
)
# Pipeline stages: 1=fit, 2=bias_test
parser.add_argument(
    '--run', type=str, default='2',
    choices=['1', '2', '1-2'],
    help='Stages: 1=fit, 2=bias_test (default: 2)'
)
# Pseudodata scaling for bias_test
parser.add_argument(
    '--pert', type=float, default=1.05,
    help='Target pseudodata size (default: 1.05)'
)
# Fit options: combine channels, timing, quiet mode
parser.add_argument(
    '--combine', '--comb', action='store_true',
    help='Combine channels for fit'
)
parser.add_argument(
    '--t', action='store_true',
    help='Display elapsed time'
)
parser.add_argument(
    '--noprint', action='store_true',
    help='Suppress uncertainty output'
)
# Additional fit/bias-test arguments
parser.add_argument(
    '--extra', nargs='*', default=[],
    choices=['freeze', 'float', 'plot_dc', 'polL', 'polR', 
             'ILC', 'tot', 'onlyrun', 't'],
    help='Extra fit arguments'
)
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

def parse_channels(cat_arg: str, combine: bool) -> list[str]:
    '''Return channel list from CLI values.

    If `combine` is requested, include the combined fit channel
    (`comb`). With an empty category string, only `comb` is used.
    '''
    cats_local = cat_arg.split('-') if cat_arg else []
    if combine:
        return ['comb'] if not cats_local or cats_local == [''] else (
            cats_local + ['comb']
        )
    return cats_local


def parse_ecms(ecm_arg: str) -> list[int]:
    '''Return list of center-of-mass energies as integers.'''
    return [int(e) for e in ecm_arg.split('-')]


# Parse channel configuration and center-of-mass energies
cats = parse_channels(arg.cat, arg.combine)
ecms = parse_ecms(arg.ecm)

# Active selections for analysis
sels = [
    'Baseline',
    'Baseline_miss',
    'Baseline_sep',
]

# Map stage identifiers to script names
SCRIPT_MAP = {'1': 'fit', '2': 'bias_test'}
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
    print('\n' + '=' * length)
    print(msg.center(length))
    print('=' * length)

    # Build base arguments common to all stages
    cmd = [sys.executable, script_path, '--ecm', str(ecm), '--sel', sel]

    # Add channel or combine flag
    if cat == 'comb':
        cmd.append('--comb')
    else:
        cmd.extend(['--cat', cat])

    # Append stage-specific arguments
    if script == 'fit':
        if arg.t:       cmd.append('--t')
        if arg.noprint: cmd.append('--noprint')
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
    print('=' * length)
    print(msg.center(length))
    print('=' * length + '\n')

    return result.returncode



######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    '''Sequential execution using nested loops with batch markers.'''
    for sel in sels:
        for script in scripts:
            # Display batch execution info without allocating task list
            task_count = len(ecms) * len(cats)
            msg = f'BATCH: Running {task_count} task(s) for {sel = } | {script = }'
            length = len(msg) + 2
            print('\n' + '█' * length)
            print(msg.center(length))
            print('█' * length)

            # Nested loops: ecm -> cat for deterministic ordering
            for ecm in ecms:
                for cat in cats:
                    ret_code = run(cat, ecm, sel, script)
                    if ret_code != 0:
                        msg = (
                            f'Failed: {cat = } | {ecm = } | {sel = } | {script = }'
                        )
                        print(f'*** ERROR: {msg} ***\n')
                        exit(1)

    timer(t)

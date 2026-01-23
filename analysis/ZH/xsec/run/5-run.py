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
    choices=['240', '365', '240-365'],
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

# Parse channel configuration
cats = arg.cat.split('-') if arg.cat else []
if arg.combine:
    # Add 'comb' for combined fit or use exclusively if no channels specified
    cats = ['comb'] if not cats or cats == [''] else cats + ['comb']

# Parse center-of-mass energies as integers
ecms = [int(e) for e in arg.ecm.split('-')]

# Active selections for analysis
sels = [
    'Baseline',
    # 'Baseline_miss',
    'Baseline_sep',
    # 'Jan_sample'
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
    print('\n' + '=' * 70)
    print(f'▶ STARTING: [{script}] {cat = } | {ecm = } | {sel = }')
    print('=' * 70)

    # Build base arguments common to all stages
    cmd = ['python3', script_path, '--ecm', str(ecm), '--sel', sel]

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
    result = subprocess.run(
        cmd,
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Completion status marker
    status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
    print('=' * 70)
    print(f'{status}: [{script}] cat={cat} | ecm={ecm} | sel={sel}')
    print('=' * 70 + '\n')

    return result.returncode



######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    # Sequential execution using nested loops with batch markers
    failed_tasks = []

    for sel in sels:
        for script in scripts:
            # Build task list for current selection/script combination
            tasks = [(cat, ecm, sel, script) for ecm in ecms for cat in cats]

            # Display batch execution info
            print('\n' + '█' * 70)
            print(f'BATCH: Running {len(tasks)} task(s) for '
                  f'{sel = }, {script = }')
            print('█' * 70)

            # Nested loops: ecm -> cat for deterministic ordering
            for ecm in ecms:
                for cat in cats:
                    ret_code = run(cat, ecm, sel, script)
                    if ret_code != 0:
                        # Track failures but continue to collect all errors
                        msg = (f'Failed: cat={cat}, ecm={ecm}, '
                               f'sel={sel}, script={script}')
                        failed_tasks.append(msg)
                        print(f'*** ERROR: {msg} ***\n')

    # Report summary and exit with error if any tasks failed
    if failed_tasks:
        print('\n' + '=' * 60)
        print('EXECUTION SUMMARY: FAILURES DETECTED')
        print('=' * 60)
        for task in failed_tasks:
            print(f'  - {task}')
        sys.exit(1)
    else:
        print('\n' + '=' * 60)
        print('EXECUTION SUMMARY: ALL TASKS COMPLETED SUCCESSFULLY')
        print('=' * 60)

    timer(t)

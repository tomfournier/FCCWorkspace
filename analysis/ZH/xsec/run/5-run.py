'''Wrapper to run the fit + bias-test pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Batch execution of fit stages across energies, channels, and selections while streaming output.
- Optional toggles for combined fits, timing, quiet uncertainty printout, and bias-test extras.

Conventions:
- Scripts are executed in nested loops: ecm -> cat -> selection -> stage-specific script.
- Paths are built from loc.ROOT/5-Fit to match the repository layout.

Usage:
    python 5-run.py                               # Default: all channels, all ecms, bias_test only
    python 5-run.py --cat ee --ecm 365 --run 1-2  # Fit then bias_test for ee at 365
    python 5-run.py --cat ee-mumu --ecm 240-365   # Multiple channels and energies
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

parser = ArgumentParser(description='Run analysis pipeline with automated parameters')
# Select lepton final states; empty string is only valid with --combine to run the combined fit directly
parser.add_argument('--cat', type=str, default='ee-mumu', 
                    choices=['ee', 'mumu', 'ee-mumu', ''],
                    help='Final state (ee, mumu) or both, qq is not available yet (default: ee-mumu)')
# Choose center-of-mass energy; dash-separated values run multiple energies sequentially
parser.add_argument('--ecm', type=str, default='240-365', 
                    choices=['240', '365', '240-365'],
                    help='Center-of-mass energy in GeV (default: 240-365)')

# Select pipeline stages: 1=fit, 2=bias_test; dash-separated runs both in order
parser.add_argument('--run', type=str, default='2', 
                    choices=['1', '2', '1-2'],
                    help='Pipeline stages: 1=fit, 2=bias_test (default: 2)')

# Pseudodata scaling used by bias_test
parser.add_argument('--pert',   help='Target pseudodata size', 
                    type=float, default=1.05)

# Fit mode: combine channels; timer and print options
parser.add_argument('--combine', '--comb', 
                                 help='Combine the channel to do the fit', action='store_true')
parser.add_argument('--t',       help='Compute the elapsed time to run the code', action='store_true')
parser.add_argument('--noprint', help='Do not display the uncertainty', action='store_true')
# Additional fit/bias-test options passed through to the underlying scripts
parser.add_argument('--extra',   help='Extra argument for the fit', 
                    choices=['freeze', 'float', 'plot_dc', 'polL', 'polR', 'ILC', 'tot', 'onlyrun', 't'], 
                    nargs='*',  default=[])
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
if arg.combine and cats!=['']: cats.append('comb')
elif arg.combine and cats==['']: cats = ['comb']
else:
    raise ValueError('Wrong value of cats and combine')
ecms = [int(e) for e in arg.ecm.split('-')]
sels = [
    # 'Baseline',
    # 'Baseline_miss',
    'Baseline_sep',
    'Baseline_sep1', 'Baseline_sep2', 
    # 'Baseline_sep3'
]

# Map stage numbers to BDT script names
script_map = {'1': 'fit', '2': 'bias_test'}
scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for analysis scripts
path = f'{loc.ROOT}/5-Fit'



##########################
### EXECUTION FUNCTION ###
##########################

def run(cat: str, 
        ecm: int,
        sel: str,
        path: str,
        script: str,
        ) -> None:
    '''Execute one fit stage with streaming output.

    Builds command-line arguments for the selected stage, appends optional
    flags when applicable, and forwards execution to the target script while
    piping stdout/stderr to the terminal.

    Args:
        cat (str): Lepton channel identifier ('ee', 'mumu', or 'comb').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('fit' or 'bias_test').

    Returns:
        int: Return code from the subprocess.
    '''
    
    # Preserve the current environment; the called scripts interpret args directly
    env = os.environ.copy()

    script_path = f'{path}/{script}.py'
    
    # Display execution information for traceability
    print('=' * 60)
    print(f'Running: {cat = }, {ecm = }, {sel = }')
    print('=' * 60)

    # Build per-stage arguments and append optional evaluation flags when relevant
    extra_args = ['--ecm', str(ecm), 
                  '--sel', sel]
    if arg.combine and 'comb' in cat: 
        extra_args.append('--comb')
    else:
        extra_args.extend(['--cat', cat])
    if 'fit' in script:
        if arg.t:       extra_args.append('--t')
        if arg.noprint: extra_args.append('--noprint')
    elif 'bias_test' in script:
        extra_args.extend(['--pert', str(arg.pert)])
        if arg.extra:   extra_args.extend(['--extra']+arg.extra)
    
    # Execute stage script; pipe outputs through to this terminal
    result = subprocess.run(
        ['python3', script_path] + extra_args,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return result.returncode



######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    # Nested loops: iterate over energies, channels, selections and pipeline stages
    for ecm in ecms:
        for cat in cats:
            for sel in sels:
                for script in scripts:
                    result = run(cat, ecm, sel, path, script)
                    if result != 0: sys.exit(result)

    timer(t)

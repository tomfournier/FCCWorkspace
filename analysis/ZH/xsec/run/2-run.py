'''Wrapper to run the BDT pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Batch execution of BDT stages across energies and channels while streaming output.
- Optional toggles to skip metrics plots, draw trees, or check variable distributions.

Conventions:
- Scripts are executed in nested loops: ecm -> cat -> stage-specific script.
- Paths are built from loc.ROOT/2-BDT to match the repository layout.

Usage:
    python 2-run.py                               # Default: all channels, all ecms, stages 1-2-3
    python 2-run.py --cat ee --ecm 365 --run 1-2  # Process + train for ee at 365
    python 2-run.py --cat ee-mumu --ecm 240-365   # Multiple channels and energies
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
# Select lepton final states; dash-separated values run both channels
parser.add_argument('--cat', type=str, default='ee-mumu', 
                    choices=['ee', 'mumu', 'ee-mumu', 'mumu-ee'],
                    help='Final state (ee, mumu) or both, qq is not available yet (default: ee-mumu)')
# Choose center-of-mass energy; dash-separated values run multiple energies sequentially
parser.add_argument('--ecm', type=str, default='240-365', 
                    choices=['240', '365', '240-365', '365-240'],
                    help='Center-of-mass energy in GeV (default: 240-365)')
# Select pipeline stages: 1=process_input, 2=train_bdt, 3=evaluation; dash-separated runs multiple
parser.add_argument('--run', type=str, default='1-2-3', 
                    choices=['1', '2', '3', '1-2', '2-3', '1-2-3'],
                    help='Pipeline stages: 1=pre-selection, 2=final-selection, 3=plots (default: 1-2-3)')

# Evaluation options (only used during stage 3)
parser.add_argument('--metric', help='Do not plot the metrics plots',        action='store_true')
parser.add_argument('--tree',   help='Plot the Decision Trees from the BDT', action='store_true')
parser.add_argument('--check',  help='Plot the variables distribution',      action='store_true')
parser.add_argument('--hl',     help='Plot the variables distribution for high and low score region', action='store_true')
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
ecms = [int(e) for e in arg.ecm.split('-')]

# Map stage numbers to BDT script names
script_map = {'1': 'process_input', '2': 'train_bdt', '3': 'evaluation'}
scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for analysis scripts
path = f'{loc.ROOT}/2-BDT'



##########################
### EXECUTION FUNCTION ###
##########################

def run(cat: str, 
        ecm: int, 
        path: str,
        script: str,
        ) -> None:
    '''Execute one BDT stage with streaming output.
    
    Builds command-line arguments for the selected stage, appends optional
    evaluation flags when applicable, and forwards execution to the target
    script while piping stdout/stderr to the terminal.
    
    Args:
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('process_input', 'train_bdt', or 'evaluation').
    
    Returns:
        int: Return code from the subprocess.
    '''
    
    # Preserve the current environment; the called scripts interpret args directly
    env = os.environ.copy()

    script_path = f'{path}/{script}.py'
    
    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = }'
    length = len(msg) + 2
    print('\n' + '=' * length)
    print(msg.center(length))
    print('=' * length)

    # Build per-stage arguments and append optional evaluation flags when relevant
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'evaluation' in script:
        if arg.metric: extra_args.append('--metric')
        if arg.tree:   extra_args.append('--tree')
        if arg.check:  extra_args.append('--check')
        if arg.hl:     extra_args.append('--hl')
    
    # Execute stage script; pipe outputs through to this terminal
    result = subprocess.run(
        ['python', script_path] + extra_args,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    # Completion status marker
    status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
    msg = f'{status}: [{script}] {cat = } | {ecm = }'
    length = len(msg) + 2
    print('=' * length)
    print(msg.center(length))
    print('=' * length + '\n')
    return result.returncode



######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    # Nested loops: iterate over energies, channels, and pipeline stages
    for ecm in ecms:
        task_count = len(cats) * len(scripts)
        msg = f'BATCH: Running {task_count} task(s) for {ecm = }'
        length = len(msg) + 2
        print('\n' + '█' * length)
        print(msg.center(length))
        print('█' * length)
        
        for cat in cats:
            for script in scripts:
                result = run(cat, ecm, path, script)
                if result != 0: sys.exit(result)

    timer(t)

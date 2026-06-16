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

# Load directory path manager and timing utility
from package.userConfig import loc         # Directory path configuration
from package.config import timer           # Execution timing utility

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log  # Argument parsing utilities
from package.logger import get_logger               # Logging setup
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories (--cat ee-mumu)
    ecm_multi=True,        # Support multiple energies (--ecm 240-365)
    allow_qq=True,         # Disable hadronic channel (ee/mumu only)
    include_sels=True,     # Include selection strategy options
    bdt_eval=True,         # Include BDT evaluation options (metrics, trees, checks)
    run_stages=3,          # BDT pipeline has 3 stages: process + train + evaluate
    description='Run BDT pipeline'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')                              # Decay categories: ['ee'] or ['ee', 'mumu']
ecms = [int(e) for e in arg.ecm.split('-')]          # Energies: [240] or [240, 365]

# Map pipeline stage numbers to BDT script names
script_map = {
    '1': 'process_input',   # Stage 1: Load histograms, balance samples, prepare for BDT training
    '2': 'train_bdt',       # Stage 2: Train XGBoost classifier with early stopping
    '3': 'evaluation'       # Stage 3: Evaluate BDT performance, generate plots
}
scripts = [script_map[s] for s in arg.run.split('-')]

# Base path for BDT analysis scripts
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
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    # Build per-stage arguments and append optional evaluation flags when relevant
    extra_args = ['--cat', cat, '--ecm', str(ecm)]

    # Pass verbose flag to child process so logging is configured the same way
    if arg.verbose:
        extra_args.append('-v')

    if 'evaluation' in script:
        if not arg.metric: extra_args.append('--no-metric')
        if arg.tree:       extra_args.append('--tree')
        if arg.check:      extra_args.append('--check')
        if arg.hl:         extra_args.append('--hl')
    if arg.sels!='':
        extra_args.extend(['--sels', arg.sels])

    # Execute stage script; pipe outputs through to this terminal
    cmd = ['python', script_path] + extra_args
    LOGGER.info(f'Executing command: {" ".join(cmd)}')
    result = subprocess.run(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    # Completion status marker
    status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
    msg = f'{status}: [{script}] {cat = } | {ecm = }'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)
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
                    result = run(cat, ecm, path, script)
                    if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

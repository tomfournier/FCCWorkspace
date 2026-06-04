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

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys, json, time, subprocess

# Load directory path manager and utilities
from package.userConfig import loc, PathObj  # Directory path configuration
loc.set_default_type('Path')
from package.config import timer             # Execution timing utility

# Start execution timer
t = time.time()

# Reuse environment without copying for each subprocess
ENV = os.environ.copy()
ENV['RUN'] = '1'  # Flag for automated mode


########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log  # Argument parsing utilities
from package.logger import get_logger               # Logging setup
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories (--cat ee-mumu)
    ecm_multi=True,        # Support multiple energies (--ecm 240-365)
    include_sels=True,     # Include selection strategy options
    run_stages=2,          # Combine pipeline has 2 stages: process_histogram + combine
    run_default='1-2',     # Run both stages by default
    polarization=True,     # Include polarization/scale options
    description='Run Combine pipeline'
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

# Parse selection strategies (from command-line or defaults)
if arg.sels == '':
    sels = ['Baseline']  # Default selections
else:
    sels = arg.sels.split('-')  # Parse from command-line

# Map pipeline stage numbers to script names
script_map = {
    '1': 'combine',   # Stage 1: Create combine datacards from histograms
    '2': 'fit'        # Stage 2: Fit the datacards
}
scripts = [script_map[s] for s in arg.run.split('-')]

# Map script names to fccanalysis subcommands
cmds = {'combine': 'combine'}  # Only 'combine' uses fccanalysis subcommand



##########################
### EXECUTION FUNCTION ###
##########################

def run(cfg_dir: PathObj,
        cat: str,
        ecm: int,
        sel: str,
        script: str,
        ) -> None:
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
    # Create configuration directory if it doesn't exist
    cfg_dir.mkdir(exist_ok=True, parents=True)
    cfg_file = cfg_dir / '2-run.json'

    # Write configuration to temporary JSON file
    cfg_file.write_text(json.dumps({'cat': cat, 'ecm': ecm, 'sel': sel}))

    script_path = loc.ROOT / '2-Fit' / f'{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = }'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    # Use fccanalysis subcommands when available; fall back to python for others
    cmd = ['fccanalysis', cmds[script], script_path] if script in cmds \
        else [sys.executable, script_path]


    if script == 'fit':
        # Add channel or combine flag
        if cat == 'comb':
            cmd.append('--comb')
        else:
            cmd.extend(['--cat', cat])
        cmd.extend(['--ecm', str(ecm)])
        cmd.append('--no-timer')
        if arg.print: cmd.append('--print')


    # Execute fccanalysis with modified environment and stream output
    result = subprocess.run(
        cmd,
        env=ENV,
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
                for sel in sels:
                    for script in scripts:
                        result = run(loc.RUN, cat, ecm, sel, script)
                        if result != 0: sys.exit(result)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)

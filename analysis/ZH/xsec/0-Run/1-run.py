'''Wrapper to run the selection/plotting pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Temporary config JSON per job to pass cat/ecm/lumi to the downstream scripts.
- Batch execution across energies and channels while streaming child output.

Conventions:
- Temporary configuration files are created in loc.RUN and removed after each stage.
- Environment variable RUN='1' flags automated mode for the analysis scripts.
- Scripts are executed in nested loops: ecm -> cat -> stage-specific script.
- Paths are rooted at loc.ROOT/1-MVAInputs to match the repository layout.

Usage:
    python 1-run.py                               # Default: all channels, all ecms, stages 2-3
    python 1-run.py --cat ee --ecm 365 --run 1-2  # Pre + final selection for ee at 365
    python 1-run.py --cat ee-mumu --ecm 240-365   # Multiple channels and energies
'''

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys, json, time, subprocess, uuid

from pathlib import Path

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
    include_sels=True,     # Include selection strategy options
    run_stages=3,          # MVA pipeline has 3 stages: pre + final + plots
    add_test=True,         # Add test sample option
    batch=True,            # Include batch execution options
    description='Run MVA Inputs pipeline'
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

# Map pipeline stage numbers to analysis script names
script_map = {
    '1': 'pre-selection',      # Stage 1: Apply pre-selection cuts, compute kinematic variables
    '2': 'final-selection',    # Stage 2: Fill histograms with BDT variables
    '3': 'plots'               # Stage 3: Generate distribution plots
}
scripts = [script_map[s] for s in arg.run.split('-')]

# Map script names to fccanalysis subcommands
cmds = {
    'pre-selection': 'run',      # Use fccanalysis run
    'final-selection': 'final',  # Use fccanalysis final
    'plots': 'plots'             # Use fccanalysis plots
}

# Base path for MVA analysis scripts
path = f'{loc.ROOT}/1-MVAInputs'



##########################
### EXECUTION FUNCTION ###
##########################

def run(cat: str,
        ecm: int,
        path: str,
        script: str,
        ) -> None:
    '''Execute one stage with a temporary config and streamed output.

    Builds a JSON config file with cat/ecm/lumi, sets RUN=1, and calls the proper
    fccanalysis subcommand for the requested stage while piping stdout/stderr
    through to the parent terminal. Uses unique filenames only in batch mode to avoid
    race conditions from concurrent batch submissions.

    Args:
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('pre-selection', 'final-selection', or 'plots').

    Returns:
        int: Return code from the subprocess.
    '''
    # Only use UUID for batch mode to avoid race conditions with concurrent submissions
    # For local execution, the standard 1-run.json is fine since it runs sequentially
    run_uuid = None
    if arg.batch:
        run_uuid = uuid.uuid4().hex[:8]
        config_filename = f'1-run-{run_uuid}.json'
    else:
        config_filename = '1-run.json'

    cfg_path = Path(loc.RUN) / config_filename
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration dictionary
    lumi = 10.8 if ecm == 240 else (3.12 if ecm==365 else -1)
    config = {'cat': cat, 'ecm': ecm, 'lumi': lumi, 'test': arg.test}

    # Write configuration to temporary JSON file
    cfg_path.write_text(json.dumps(config))

    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'
    if arg.batch:
        env['RUN_UUID'] = run_uuid  # Pass UUID only for batch mode
        # Create a userBatchConfig file that exports the UUID for batch jobs
        userBatchConfig = Path(loc.RUN) / 'userBatch.Config'
        userBatchConfig.write_text(f'export RUN_UUID={run_uuid}\n')
        env['RUN_USER_BATCH_CONFIG'] = str(userBatchConfig)
        env['RUN_BATCH'] = '1'

    script_path = f'{path}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = } | {lumi = } | test = {arg.test}'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    # Execute fccanalysis with modified environment and stream output
    result = subprocess.run(
        ['fccanalysis', cmds[script], script_path],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    # Completion status marker
    status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
    msg = f'{status}: [{script}] {cat = } | {ecm = } | test = {arg.test}'
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

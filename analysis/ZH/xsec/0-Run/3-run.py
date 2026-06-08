'''Wrapper to run the measurement pipeline with automated parameters.

Provides:
- Argument parsing for channel (ee/mumu), energy (240/365 GeV), and pipeline stages.
- Temporary config JSON per job to pass cat/ecm/lumi to downstream scripts.
- Batch execution across energies and channels while streaming child output.

Conventions:
- Temporary configuration files are created in loc.RUN and removed after each stage.
- Environment variable RUN='1' flags automated mode for the analysis scripts.
- Scripts are executed in nested loops: ecm -> cat -> stage-specific script, then plots/cutflow.

Usage:
    python 3-run.py                                  # Default: all channels, all ecms, stages 2-3
    python 3-run.py --cat ee --ecm 365 --run 1-2-3-4 # All stages including cutflow
    python 3-run.py --cat ee-mumu --ecm 240-365      # Multiple channels and energies
'''

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys, uuid, json, time, subprocess

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
    plots=True,            # Include plot generation options
    cutflow=True,          # Include cutflow analysis options
    run_stages=4,          # Measurement pipeline has 4 stages: pre + final + plots + cutflow
    add_test=True,         # Add test sample option
    batch=True,            # Include batch execution options
    description='Run Measurement pipeline'
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

# Map pipeline stage numbers to script names
script_map = {
    '1': 'pre-selection',      # Stage 1: Apply pre-selection cuts, compute kinematic variables
    '2': 'final-selection',    # Stage 2: Fill measurement histograms with BDT scores
    '3': 'plots',              # Stage 3: Generate distribution plots
    '4': 'cutflow'             # Stage 4: Analyze event yield per cut stage
}
scripts = [script_map[s] for s in arg.run.split('-')]

# Map script names to fccanalysis subcommands
cmds = {
    'pre-selection': 'run',      # Use fccanalysis run
    'final-selection': 'final'   # Use fccanalysis final
}

# Base path for measurement analysis scripts
path = f'{loc.ROOT}/3-Measurement'



##########################
### EXECUTION FUNCTION ###
##########################

def run(cat: str,
        ecm: int,
        path: str,
        script: str,
        ) -> None:
    '''Execute one measurement stage with a temporary config and streamed output.

    Builds a JSON file with cat/ecm/lumi, sets RUN=1, and calls the stage script
    via fccanalysis (or python for non-fccanalysis steps) while piping stdout/stderr
    through to the parent terminal.

    Args:
        cfg_dir (str): Directory where the temporary config file will be stored.
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('pre-selection', 'final-selection', 'plots', 'cutflow').

    Returns:
        int: Return code from the subprocess.
    '''
    run_uuid = None
    if arg.batch:
        run_uuid = uuid.uuid4().hex[:8]
        config_name = f'3-run-{run_uuid}.json'
    else:
        config_name = '3-run.json'

    # Create configuration directory if it doesn't exist
    cfg_path = Path(loc.RUN) / config_name
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration dictionary
    lumi = 10.8 if ecm == 240 else (3.12 if ecm==365 else -1)
    config = {'cat': cat, 'ecm': ecm, 'lumi': lumi, 'test': arg.test}

    # Write configuration to temporary JSON file
    cfg_path.write_text(json.dumps(config))
    LOGGER.info(f'Wrote config file to {cfg_path}')

    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'
    if arg.batch:
        env['RUN_UUID'] = run_uuid  # Pass UUID only for batch mode
        # Create a userBatchConfig file that exports the UUID for batch mode
        userBatchConfig = Path(loc.RUN) / 'userBatch.Config'
        userBatchConfig.write_text(f'export RUN_UUID={run_uuid}\n')
        env['RUN_USER_BATCH_CONFIG'] = str(userBatchConfig)
        env['RUN_BATCH'] = '1'

    script_path = f'{path}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = } | {lumi = } | test = {arg.test}'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    # Build per-stage arguments and apply plotting cutflow flags
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'plots' in script:
        if not arg.yields: extra_args.append('--no-yields')
        if not arg.decay:  extra_args.append('--no-decay')
        if not arg.make:   extra_args.append('--no-make')
        if arg.scan:       extra_args.append('--scan')
    elif 'cutflow' in script:
        if not arg.tot:  extra_args.append('--no-tot')
        if not arg.test: extra_args.append('--no-test')
        if not arg.kin:  extra_args.append('--no-kin')
    if arg.sels!='':
        extra_args.extend(['--sels', arg.sels])

    # Use fccanalysis subcommands when available; fall back to python for others
    cmd = ['fccanalysis', cmds[script], script_path] if script in cmds \
        else ['python', script_path] + extra_args

    # Execute fccanalysis with modified environment and stream output
    result = subprocess.run(
        cmd,
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

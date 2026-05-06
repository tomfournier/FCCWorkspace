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

import os, sys, json, time, subprocess

from pathlib import Path

from package.userConfig import loc
from package.config import timer

t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,
    ecm_multi=True,
    include_sel=True,
    plots=True,
    cutfflow=True,
    run_stages=4,
    batch=True,
    description='Run Measurement pipeline'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
ecms = [int(e) for e in arg.ecm.split('-')]

# Map stage numbers to script names (cutflow added as stage 4)
script_map = {'1': 'pre-selection', '2': 'final-selection', '3': 'plots', '4': 'cutflow'}
scripts = [script_map[s] for s in arg.run.split('-')]
cmds = {'pre-selection': 'run', 'final-selection': 'final'}

# Base path for analysis scripts
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
    # Create configuration directory if it doesn't exist
    cfg_path = Path(loc.RUN) / '3-run.json'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration dictionary
    lumi = 10.8 if ecm == 240 else (3.12 if ecm==365 else -1)
    config = {'cat': cat, 'ecm': ecm, 'lumi': lumi}

    # Write configuration to temporary JSON file
    cfg_path.write_text(json.dumps(config))
    LOGGER.info(f'Wrote config file to {cfg_path}')

    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'
    if arg.batch:
        env['RUN_BATCH'] = '1'

    script_path = f'{path}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = } | {lumi = }'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)

    # Build per-stage arguments and apply plotting cutflow flags
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'plots' in script:
        if arg.yields: extra_args.append('--yields')
        if arg.decay:  extra_args.append('--decay')
        if arg.make:   extra_args.append('--make')
        if arg.scan:   extra_args.append('--scan')
    elif 'cutflow' in script:
        if arg.tot:    extra_args.append('--tot')
    if arg.sels!='':
        extra_args.extend(['--sels', arg.sels])

    # Use fccanalysis subcommands when available; fall back to python for others
    cmd = ['fccanalysis', cmds[script], script_path] if script in cmds \
        else ['python', script_path] + extra_args

    try:
        # Execute fccanalysis with modified environment and stream output
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
    finally:
        pass


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
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
                    result = run(loc.RUN, cat, ecm, path, script)
                    if result != 0: sys.exit(result)

        # BATCH info for plots
        if is_there_plots:
            result = run(loc.RUN, arg.cat, ecm, path, 'plots')
            if result != 0: sys.exit(result)
        # BATCH info for cutflow
        if is_there_cutflow:
            result = run(loc.RUN, arg.cat, ecm, path, 'cutflow')
            if result != 0: sys.exit(result)

    timer(t)

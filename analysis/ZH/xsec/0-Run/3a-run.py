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
    python 3a-run.py                                  # Default: all channels, all ecms, stages 2-3
    python 3a-run.py --cat ee --ecm 365 --run 1-2-3   # All stages including cutflow
    python 3a-run.py --cat ee-mumu --ecm 240-365      # Multiple channels and energies
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

from package.parsing import create_parser
parser = create_parser(
    cat_multi=True,
    ecm_multi=True,
    include_sels=True,
    run_stages=3,
    batch=True,  # Have to implement it
    optimize=True,
    description='Run Optimisation pipeline'
)
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
ecms = [int(e) for e in arg.ecm.split('-')]

# Map stage numbers to script names
script_map = {'1': 'pre-selection', '2': 'optimize_ll', '3': 'plots'}
scripts = [script_map[s] for s in arg.run.split('-')]
cmds = {'pre-selection': 'run', 'final-selection': 'final'}

# Base path for analysis scripts
path = f'{loc.ROOT}/3a-Optimisation'



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
        script (str): Stage script name ('pre-selection', 'optimisation', 'plots').

    Returns:
        int: Return code from the subprocess.
    '''
    # Create configuration directory if it doesn't exist
    cfg_path = Path(loc.RUN) / '3a-run.json'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration dictionary
    config = {'cat': cat, 'ecm': ecm}

    # Write configuration to temporary JSON file
    cfg_path.write_text(json.dumps(config))
    print(f'----->[Info] Wrote config file to {cfg_path}')

    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'
    if arg.batch:
        env['RUN_BATCH'] = '1'

    script_path = f'{path}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = }'
    length = len(msg) + 2
    print('\n' + '=' * length)
    print(msg.center(length))
    print('=' * length)

    # Build per-stage arguments and apply plotting cutflow flags
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'optimize' in script:
        extra_args.extend(['--procs',   arg.procs])
        extra_args.extend(['--nevents', str(arg.nevents)])
        extra_args.extend(['--incr',    str(arg.incr)])
    elif 'plots' in script:
        extra_args.extend(['--procs', arg.procs])

    # Use fccanalysis subcommands when available; fall back to python for others
    cmd = ['fccanalysis', cmds[script], script_path] if script in cmds \
        else ['python', script_path] + extra_args

    try:
        # Execute fccanalysis with modified environment and stream output
        result = subprocess.run(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        # Completion status marker
        status = '✓ COMPLETED' if result.returncode == 0 else '✗ FAILED'
        msg = f'{status}: [{script}] {cat = } | {ecm = }'
        length = len(msg) + 2
        print('=' * length)
        print(msg.center(length))
        print('=' * length + '\n')
        return result.returncode
    finally:
        pass


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
        pass  # Let finally block run without printing traceback
    except Exception:
        pass
    finally:
        timer(t)

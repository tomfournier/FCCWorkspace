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

import os, sys, json, time, subprocess

from argparse import ArgumentParser
from pathlib import Path

from package.userConfig import loc
from package.config import timer

t = time.time()



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser(description='Run analysis pipeline with automated parameters')
# Select lepton final states; dash-separated values run both channels
parser.add_argument('--cat', type=str, default='ee-mumu',
                    choices=['ee', 'mumu', 'qq', 'ee-mumu', 'mumu-ee'],
                    help='Final state (ee, mumu) or both, qq is not available yet (default: ee-mumu)')
# Choose center-of-mass energy; dash-separated values run multiple energies sequentially
parser.add_argument('--ecm', type=str, default='240-365',
                    choices=['240', '365', '240-365', '365-240'],
                    help='Center-of-mass energy in GeV (default: 240-365)')
# Select pipeline stages: 1=pre-selection, 2=final-selection, 3=plots; dash-separated runs multiple
parser.add_argument('--run', type=str, default='2-3',
                    choices=['1', '2', '3', '1-2', '2-3', '1-2-3'],
                    help='Pipeline stages: 1=pre-selection, 2=final-selection, 3=plots (default: 2-3)')
parser.add_argument('--batch', action='store_true', help='run pre-selection on HTCondor')
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
ecms = [int(e) for e in arg.ecm.split('-')]

# Map stage numbers to script filenames
script_map = {'1': 'pre-selection', '2': 'final-selection', '3': 'plots'}
scripts = [script_map[s] for s in arg.run.split('-')]
# Map script to fccanalysis subcommand (plots handled directly by fccanalysis)
cmds = {'pre-selection': 'run', 'final-selection': 'final', 'plots': 'plots'}

# Base path for analysis scripts
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

    Builds a JSON file with cat/ecm/lumi, sets RUN=1, and calls the proper
    fccanalysis subcommand for the requested stage while piping stdout/stderr
    through to the parent terminal.

    Args:
        cat (str): Lepton channel identifier ('ee' or 'mumu').
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        path (str): Base directory for stage scripts.
        script (str): Stage script name ('pre-selection', 'final-selection', or 'plots').

    Returns:
        int: Return code from the subprocess.
    '''
    cfg_path = loc.RUN.astype(Path) / '1-run.json'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration dictionary
    lumi = 10.8 if ecm == 240 else (3.12 if ecm==365 else -1)
    config = {'cat': cat, 'ecm': ecm, 'lumi': lumi}

    # Write configuration to temporary JSON file(s)
    cfg_path.write_text(json.dumps(config))

    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'
    if arg.batch:
        env['RUN_BATCH'] = '1'

    script_path = f'{path}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = } | {lumi = }'
    length = len(msg) + 2
    print('\n' + '=' * length)
    print(msg.center(length))
    print('=' * length)

    try:
        # Execute fccanalysis with modified environment and stream output
        result = subprocess.run(
            ['fccanalysis', cmds[script], script_path],
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
    finally:
        pass


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

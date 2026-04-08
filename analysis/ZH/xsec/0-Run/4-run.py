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
    python 4-run.py                              # Default: all channels, all ecms, combine only
    python 4-run.py --cat ee --ecm 365 --run 1-2 # process_histogram then combine for ee at 365
    python 4-run.py --cat ee-mumu --ecm 240-365  # Multiple channels and energies
'''

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys, json, time, subprocess

from pathlib import Path

from package.userConfig import loc
from package.tools.utils import mkdir
from package.config import timer

t = time.time()

# Reuse environment without copying for each subprocess
ENV = os.environ.copy()
ENV['RUN'] = '1'


########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser
parser = create_parser(
    cat_multi=True,
    ecm_multi=True,
    include_sels=True,
    run_stages=2,
    run_default='1-2',
    polarization=True,
    description='Run Combine pipeline'
)
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
ecms = [int(e) for e in arg.ecm.split('-')]
if arg.sels == '':
    sels = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']
else:
    sels = arg.sels.split('-')

# Map stage numbers to script names (cutflow added as stage 4)
script_map = {'1': 'process_histogram', '2': 'combine'}
scripts = [script_map[s] for s in arg.run.split('-')]
cmds = {'combine': 'combine'}

# Base path for analysis scripts
path = f'{loc.ROOT}/4-Combine'



##########################
### EXECUTION FUNCTION ###
##########################

def run(cfg_dir: str,
        cat: str,
        ecm: int,
        sel: str,
        path: str,
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
    mkdir(cfg_dir)
    cfg_file = Path(cfg_dir) / '4-run.json'

    # Build configuration dictionary
    config = {'cat': cat, 'ecm': ecm}
    if script=='combine': config['sel'] = sel

    # Write configuration to temporary JSON file
    cfg_file.write_text(json.dumps(config))

    script_path = f'{path}/{script}.py'

    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = }'
    length = len(msg) + 2
    print('\n' + '=' * length)
    print(msg.center(length))
    print('=' * length)

    # Build per-stage arguments and apply plotting cutflow flags
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'process_histogram' in script:
        if arg.polL: extra_args.append('--polL')
        if arg.polR: extra_args.append('--polR')
        if arg.ILC:  extra_args.append('--ILC')
        if arg.sels!='': extra_args.extend(['--sels', arg.sels])

    # Use fccanalysis subcommands when available; fall back to python for others
    cmd = ['fccanalysis', cmds[script], script_path] if script in cmds \
        else [sys.executable, script_path] + extra_args

    try:
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
        print('=' * length)
        print(msg.center(length))
        print('=' * length + '\n')
        return result.returncode
    finally:
        # Cleanup: remove temporary configuration file
        if cfg_file.exists():
            cfg_file.unlink()


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    # Nested loops: iterate over energies, channels, and pipeline stages
    for ecm in ecms:
        if 'process_histogram' in scripts:
            msg = f'BATCH: Running 1 task(s) for {ecm = } | script = process_histogram'
            length = len(msg) + 2
            print('\n' + '█' * length)
            print(msg.center(length))
            print('█' * length)
            result = run(loc.RUN, arg.cat, ecm, '', path, 'process_histogram')
            if result != 0: sys.exit(result)
        if 'combine' in scripts:
            task_count = len(cats) * len(sels)
            msg = f'BATCH: Running {task_count} task(s) for {ecm = } | script = combine'
            length = len(msg) + 2
            print('\n' + '█' * length)
            print(msg.center(length))
            print('█' * length)
            for cat in cats:
                for sel in sels:
                    result = run(loc.RUN, cat, ecm, sel, path, 'combine')
                    if result != 0: sys.exit(result)

    timer(t)

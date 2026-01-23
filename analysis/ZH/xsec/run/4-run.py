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

from argparse import ArgumentParser
from pathlib import Path

from package.userConfig import loc
from package.tools.utils import mkdir
from package.config import timer

t = time.time()


########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser(description='Run analysis pipeline with automated parameters')
# Select lepton final states; dash-separated values run both channels
parser.add_argument('--cat', type=str, default='ee-mumu', 
                    choices=['ee', 'mumu', 'ee-mumu'],
                    help='Final state (ee, mumu) or both, qq is not available yet (default: ee-mumu)')
# Choose center-of-mass energy; dash-separated values run multiple energies sequentially
parser.add_argument('--ecm', type=str, default='240-365', 
                    choices=['240', '365', '240-365'],
                    help='Center-of-mass energy in GeV (default: 240-365)')
# Select pipeline stages: 1=process_histogram, 2=combine; dash-separated runs both in order
parser.add_argument('--run', type=str, default='2', choices=['1', '2', '1-2'],
                    help='Pipeline stages: 1=process-histogram, 2=combine (default: 2)')

# Polarization and luminosity scaling options used by process_histogram
parser.add_argument('--polL', help='Scale to left polarization',  action='store_true')
parser.add_argument('--polR', help='Scale to right polarization', action='store_true')
parser.add_argument('--ILC',  help='Scale to ILC cross-section',  action='store_true')
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Parse comma-separated arguments into lists
cats = arg.cat.split('-')
ecms = [int(e) for e in arg.ecm.split('-')]
sels = [
    'Baseline', 
    'Baseline_miss', 
    'Baseline_sep'
]

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
    
    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'

    script_path = f'{path}/{script}.py'
    
    # Display execution information for traceability
    print('=' * 60)
    print(f'Running: {cat = }, {ecm = } for {script}')
    print('=' * 60)


    # Build per-stage arguments and apply plotting cutflow flags
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'process-histogram' in script:
        if arg.polL: extra_args.append('--polL')
        if arg.polR: extra_args.append('--polR')
        if arg.ILC:  extra_args.append('--ILC')
    
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
            result = run(loc.RUN, arg.cat, ecm, '', path, 'process_histogram')
            if result != 0: sys.exit(result)
        if 'combine' in scripts:
            for cat in cats:
                for sel in sels:
                    result = run(loc.RUN, cat, ecm, sel, path, 'combine')
                    if result != 0: sys.exit(result)

    timer(t)

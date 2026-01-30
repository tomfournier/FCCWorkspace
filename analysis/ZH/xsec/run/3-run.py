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
                    choices=['ee', 'mumu', 'ee-mumu', 'mumu-ee'],
                    help='Final state (ee, mumu) or both, qq is not available yet (default: ee-mumu)')
# Choose center-of-mass energy; dash-separated values run multiple energies sequentially
parser.add_argument('--ecm', type=str, default='240-365', 
                    choices=['240', '365', '240-365', '365-240'],
                    help='Center-of-mass energy in GeV (default: 240-365)')
# Select pipeline stages: 1=pre-selection, 2=final-selection, 3=plots, 4=cutflow; dash-separated runs multiple
parser.add_argument('--run', type=str, default='2-3', 
                    choices=['1', '2', '3', '4', '1-2', '2-3', '3-4', '1-2-3', '2-3-4', '1-2-3-4'],
                    help='Pipeline stages: 1=pre-selection, 2=final-selection, 3=plots, 4=cutflow (default: 2-3)')

# Flags to control which plot types to skip (inverted logic: flag skips the plot except for --scan)
parser.add_argument('--yields', help='Do not make yields plots',            action='store_true')
parser.add_argument('--decay',  help='Do not make Higgs decays only plots', action='store_true')
parser.add_argument('--make',   help='Do not make distribution plots',      action='store_true')
parser.add_argument('--scan',   help='Make significance scan plots',        action='store_true')

# Include all Z decay modes in plots
parser.add_argument('--tot', help='Include all the Z decays in the plots', 
                    action='store_true')
parser.add_argument('--ww', help="Choose if run pre-selection for p8_ee_WW_ee_ecm", 
                    type=str, default='both', choices=['ww', 'other', 'both'])
arg = parser.parse_args()



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

def run(cfg_dir: str, 
        cat: str, 
        ecm: int, 
        path: str,
        script: str,
        ww: bool = False
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
    mkdir(cfg_dir)
    cfg_file = Path(cfg_dir) / '3-run.json'
    
    # Build configuration dictionary
    lumi = 10.8 if ecm == 240 else (3.12 if ecm==365 else -1)
    config = {'cat': cat, 
              'ecm': ecm, 
              'lumi': lumi,
              'ww': ww}
    
    # Write configuration to temporary JSON file
    cfg_file.write_text(json.dumps(config))
    print(f'----->[Info] Wrote config file to {cfg_file}')
    
    # Set up environment with RUN flag for automated mode detection
    env = os.environ.copy()
    env['RUN'] = '1'

    script_path = f'{path}/{script}.py'
    
    # Display execution header with clear identification
    msg = f'▶ STARTING: [{script}] {cat = } | {ecm = } | {lumi = }'
    length = len(msg) + 2
    print('\n' + '=' * length)
    print(msg.center(length))
    print('=' * length)

    # Build per-stage arguments and apply plotting cutflow flags
    extra_args = ['--cat', cat, '--ecm', str(ecm)]
    if 'plots' in script:
        if arg.yields: extra_args.append('--yields')
        if arg.decay:  extra_args.append('--decay')
        if arg.make:   extra_args.append('--make')
        if arg.scan:   extra_args.append('--scan')
    elif 'cutflow' in script:
        if arg.tot:    extra_args.append('--tot')

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
    is_there_plots   = 'plots' in scripts
    is_there_cutflow = 'cutflow' in scripts

    if is_there_plots:   scripts.remove('plots')
    if is_there_cutflow: scripts.remove('cutflow')

    # Nested loops: iterate over energies, channels, and pipeline stages
    for ecm in ecms:
        # BATCH info for pre/final-selection
        if ('pre-selection' in scripts) or ('final-selection' in scripts):
            task_count = len(cats) * len([s for s in scripts if s in ['pre-selection', 'final-selection']])
            msg = f'BATCH: Running {task_count} task(s) for ecm={ecm}'
            length = len(msg) + 2
            print('\n' + '█' * length)
            print(msg.center(length))
            print('█' * length)
            for cat in cats:
                for script in scripts:
                    if 'pre-selection' in script:
                        if arg.ww == 'both':
                            result = run(loc.RUN, cat, ecm, path, script, ww=False)
                            if result != 0: sys.exit(result)
                            result = run(loc.RUN, cat, ecm, path, script, ww=True)
                            if result != 0: sys.exit(result)
                        elif arg.ww == 'ww':
                            result = run(loc.RUN, cat, ecm, path, script, ww=True)
                            if result != 0: sys.exit(result)
                        elif arg.ww == 'other':
                            result = run(loc.RUN, cat, ecm, path, script, ww=False)
                            if result != 0: sys.exit(result)
                        else:
                            raise ValueError('Wrong value selected for --ww')
                    else:
                        result = run(loc.RUN, cat, ecm, path, script)
                        if result != 0: sys.exit(result)

        # BATCH info for plots
        if is_there_plots:
            msg = f'BATCH: Running plots for ecm={ecm} | cat={arg.cat}'
            length = len(msg) + 2
            print('\n' + '█' * length)
            print(msg.center(length))
            print('█' * length)
            result = run(loc.RUN, arg.cat, ecm, path, 'plots')
            if result != 0: sys.exit(result)
        # BATCH info for cutflow
        if is_there_cutflow:
            msg = f'BATCH: Running cutflow for ecm={ecm} | cat={arg.cat}'
            length = len(msg) + 2
            print('\n' + '█' * length)
            print(msg.center(length))
            print('█' * length)
            result = run(loc.RUN, arg.cat, ecm, path, 'cutflow')
            if result != 0: sys.exit(result)

    timer(t)

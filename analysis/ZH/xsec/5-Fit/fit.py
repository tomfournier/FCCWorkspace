##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, subprocess, sys

from time import time
from uuid import uuid4
from datetime import datetime
from argparse import ArgumentParser

# Start timer for performance tracking
t = time()

from package.userConfig import loc
from package.config import timer, warning
from package.tools.utils import mkdir



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
# Define final state: ee or mumu
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
# Define selection strategy
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

# Pseudodata parameters
parser.add_argument('--pert',   help='Target pseudodata size', 
                    type=float, default=1.0)
parser.add_argument('--target', help='Target pseudodata', 
                    type=str, default='')

# Fit mode: nominal or bias test; combine channels; timer and print options
parser.add_argument('--bias',    help='Nominal fit or bias test', action='store_true')
parser.add_argument('--combine', '--comb', 
                                 help='Combine the channel to do the fit', action='store_true')
parser.add_argument('--t',       help='Compute the elapsed time to run the code', action='store_true')
parser.add_argument('--noprint', help='Do not display the uncertainty', action='store_true')
arg = parser.parse_args()

# Validate that either a final state or combine option is selected
if arg.cat=='' and not arg.combine:
    msg = 'Final state or combine were not selected, please select one to run this code'
    warning(msg)

# Set category to combined if combining channels
if arg.combine: arg.cat = 'combined'

# Bundle arguments for directory lookup
args = [arg.cat, arg.ecm, arg.sel]



###############################
### DIRECTORY CONFIGURATION ###
###############################

# Map fit mode (nominal vs bias) to corresponding directory locations
location_map = {
    False: {  # Nominal fit
        'dir': loc.COMBINE_NOMINAL,
        'dc':  loc.NOMINAL_DATACARD,
        'ws':  loc.NOMINAL_WS,
        'log': loc.NOMINAL_LOG,
        'res': loc.NOMINAL_RESULT,
        'tp':  'nominal'
    },
    True: {   # Bias test fit
        'dir': loc.COMBINE_BIAS,
        'dc':  loc.BIAS_DATACARD,
        'ws':  loc.BIAS_WS,
        'log': loc.BIAS_LOG,
        'res': loc.BIAS_FIT_RESULT,
        'tp': 'bias'
    }
}
# Select configuration based on fit mode
config = location_map[arg.bias]
# Resolve directory paths based on configuration and arguments
dir, dc = loc.get(config['dir'], *args), loc.get(config['dc'],  *args)
ws, log = loc.get(config['ws'],  *args), loc.get(config['log'], *args)
res, tp = loc.get(config['res'], *args), config['tp']

# Define naming suffixes for file outputs
comb = '_combined' if arg.combine else ''
tar = f'_{arg.target}' if arg.bias else ''
dc_comb = loc.get('COMBINE', '', arg.ecm, arg.sel)

# Define full file paths for workspace, logs, and results
ws_file    = f'{ws}/ws{tar}.root'
log_text   = f'{log}/log_text2workspace{tar}.txt'
result_log = f'{log}/log_results{tar}.txt'

# Set up environment for subprocess calls
env = os.environ.copy()

# Create necessary directories if they don't exist
for dir_path in [dc, ws, log, res]:
    mkdir(dir_path)



####################
### FITTING PART ###
####################

def add_stamp(path: str, 
              label: str,
              status: str = 'ok'
              ) -> None:
    """Add a timestamped stamp to a log file for tracking execution status."""

    # Generate timestamp and unique ID
    ts = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
    uniq = uuid4().hex[:8]
    stamp = f'\n\n---- STAMP {label} [{status}]: {ts} | id={uniq}\n'

    # Append stamp to log file
    with open(path, 'a') as log_file:
        log_file.write(stamp)
    print(f'----->[Info] Added STAMP: {ts} | id={uniq} | status={status} | file={label}')

def fitting(dir: str, 
        dc: str, 
        ws: str,  
        tp: str, 
        dc_comb: str, 
        env: os._Environ
        ) -> int:
    """Execute the fitting workflow: combine datacards, create workspace, and run fit."""

    # Track status of each fitting step
    text_status, fit_status = 'not-run', 'not-run'
    try:
        dc_combined = f'{dc}/datacard{tar}{comb}.txt'
        
        # Combine datacards from multiple channels if requested
        if arg.combine:
            dc_mu = f'{dc_comb}/mumu/{tp}/datacard/datacard{tar}.txt'
            dc_ee = f'{dc_comb}/ee/{tp}/datacard/datacard{tar}.txt'

            print('\n----->[Info] Combining datacards')
            with open(dc_combined, 'w') as out:
                subprocess.run(['combineCards.py', dc_mu, dc_ee], 
                            stdout=out, env=env, check=True)

        # Convert datacard to RooFit workspace
        with open(log_text, 'w') as log_out:
            enter = '' if arg.combine else '\n'
            print(f'{enter}----->[Info] Setting files for the fit')
            subprocess.run(['text2workspace.py', dc_combined, '-v', '10',
                            '--X-allow-no-background', '-m', '125', '-o', ws_file],
                            stdout=log_out, stderr=subprocess.STDOUT, 
                            cwd=dir, env=env, check=True)
        text_status = 'ok'
        
        # Run the maximum likelihood fit
        with open(result_log, 'w') as log_out:
            print('----->[Info] Doing the fit')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '10', '-t', '0', '--expectSignal=1', '-n', 'Xsec'],
                            stdout=log_out, stderr=subprocess.STDOUT, 
                            cwd=ws, env=env, check=True)
        fit_status = 'ok'
        return 0

    except subprocess.CalledProcessError as exc:
        # Mark which step failed
        if text_status=='not-run':
            text_status = f'error exit={exc.returncode}'
        elif fit_status=='not-run':
            fit_status = f'error exit={exc.returncode}'
        print(f'----->[Error] Fit command failed with exit code {exc.returncode}')
        return exc.returncode
    finally:
        # Add execution stamps to log files for tracking
        if os.path.exists(log_text):
            add_stamp(log_text, f'log_text2workspace{tar}', text_status)
        if os.path.exists(result_log):
            add_stamp(result_log, f'log_results{tar}', fit_status)



##########################
### RESULTS EXTRACTION ###
##########################

def res_extraction(res_log: str
                   ) -> tuple[float, 
                              float]:
    """Extract signal strength (mu) and uncertainty from fit results log."""
    print('----->[Info] Fit done, extracting results')
    mu, err = -100, -100  # Initialize with error values
    
    # Parse log file for signal strength result
    # (parse from end to find latest result)
    with open(res_log) as file:
        lines = file.readlines()
        for line in reversed(lines):
            parts = line.replace('\t', ' ').split()

            # Match line format: r = value +/- error (limited)
            if (len(parts)>=6 and
                parts[0]=='r' and
                parts[1]=='=' and 
                parts[3]=='+/-' and
                parts[-1]=='(limited)'):

                mu, err = float(parts[2]), float(parts[4])
                break
    return mu, err



######################
### RESULTS SAVING ###
######################

def res_saving(mu: float,
               err: float,
               res: str
               ) -> None:
    """Save fitting results (signal strength and uncertainty) to output file."""

    # Check if results were successfully extracted
    if (mu==-100) and (err==-100):
        print("----->[Error] Couldn't extract value of the fit, go to the log to have more informations")
        exit(0)
    else:
        # Display results unless suppressed by flag
        if not arg.noprint:
            print('----->[Info] Results successfully extracted')
            print(f'\tmu = {mu} +/- {err}')
            print(f'----->[Info] Uncertainty obtained on ZH cross-section: {err*100:.2f} %')

        # Write results to output file
        with open(f'{res}/results{tar}.txt', 'w') as f:
            f.write(f'{mu}\n{err}\n')

        print(f'----->[Info] Saved results in {res}/results{tar}.txt')



######################
### CODE EXECUTION ###
######################

# Execute the fitting pipeline
ret = fitting(dir, dc, ws, tp, dc_comb, env)
if ret != 0:
    sys.exit(ret)

# Extract results from fit
mu, err = res_extraction(result_log)

# Save results to output file
res_saving(mu, err, res)

# Print execution time if requested
if __name__=='__main__' and arg.t:
    timer(t)
    
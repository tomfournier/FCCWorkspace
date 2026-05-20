################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, subprocess

from time import time
from uuid import uuid4
from datetime import datetime

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories
    allow_empty=True,      # Allow empty category (for combined fits)
    include_sel=True,      # Include selection strategy options
    fit=True,              # Include fit-specific options
    description='Fit Script'
)
arg = parse_args(parser, comb=True)  # Parse with combination support
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and utilities
from package.userConfig import loc
from package.config import timer         # Timing utility
from package.tools.utils import mkdir    # Directory creation



###############################
### DIRECTORY CONFIGURATION ###
###############################

# Set category to combined if combining channels
if arg.combine: arg.cat = 'combined'

# Bundle arguments for directory lookup
args = [arg.cat, arg.ecm, arg.sel]

# Map fit mode (nominal vs bias test) to corresponding directory locations
# This allows reusing code for both standard fits and bias test procedures
location_map = {
    False: {  # Nominal fit (standard measurement)
        'dir': 'COMBINE_NOMINAL',      # Main working directory
        'dc':  'NOMINAL_DATACARD',     # Input datacard directory
        'ws':  'NOMINAL_WS',           # Workspace output directory
        'log': 'NOMINAL_LOG',          # Log file directory
        'res': 'NOMINAL_RESULT',       # Fit results output directory
        'tp':  'nominal'               # Type label for naming
    },
    True: {   # Bias test fit (for testing fit bias)
        'dir': 'COMBINE_BIAS',         # Bias test working directory
        'dc':  'BIAS_DATACARD',        # Input biased datacard directory
        'ws':  'BIAS_WS',              # Workspace output directory
        'log': 'BIAS_LOG',             # Log file directory
        'res': 'BIAS_FIT_RESULT',      # Fit results output directory
        'tp': 'bias'                   # Type label for naming
    }
}
# Select configuration based on fit mode (nominal vs bias)
config = location_map[arg.bias]

# Resolve directory paths based on configuration and arguments
dir = loc.get(config['dir'], *args)   # Working directory
dc  = loc.get(config['dc'],  *args)   # Input datacard directory
ws  = loc.get(config['ws'],  *args)   # Workspace directory
log = loc.get(config['log'], *args)   # Log output directory
res = loc.get(config['res'], *args)   # Results output directory
tp  = config['tp']                         # Type identifier

# Define naming suffixes for file outputs
comb = '_combined' if arg.combine else ''   # Combined channel suffix
tar = f'_{arg.target}' if arg.bias else ''  # Bias test target suffix (e.g., _bb)
dc_comb = loc.get('COMBINE', '', arg.ecm, arg.sel)  # Combined datacard location

# Define full file paths for workspace, logs, and results
ws_file    = f'{ws}/ws{tar}.root'                  # Workspace file (workspace.root or workspace_bb.root)
log_text   = f'{log}/log_text2workspace{tar}.txt'  # Text2workspace log
result_log = f'{log}/log_results{tar}.txt'         # Fit results log

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
    LOGGER.debug(f'Added STAMP: {ts} | id={uniq} | status={status} | file={label}')

def fitting(
        dir: str,
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

            LOGGER.info('Combining datacards')
            with open(dc_combined, 'w') as out:
                subprocess.run(['combineCards.py', dc_mu, dc_ee],
                               stdout=out, env=env, check=True)

        # Convert datacard to RooFit workspace
        with open(log_text, 'w') as log_out:
            LOGGER.info('Setting files for the fit')
            subprocess.run(['text2workspace.py', dc_combined, '-v', '10',
                            '--X-allow-no-background', '-m', '125', '-o', ws_file],
                           stdout=log_out, stderr=subprocess.STDOUT,
                           cwd=dir, env=env, check=True)
        text_status = 'ok'

        # Run the maximum likelihood fit
        with open(result_log, 'w') as log_out:
            LOGGER.info('Doing the fit')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '10', '-t', '0', '--expectSignal=1', '-n', 'Xsec',
                            '--rMin', '0', '--rMax', '2', '--alignEdges', '1', '--squareDistPoiStep',
                            '--algo', 'grid', '--points', '20', '--autoRange', '3'],
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
        LOGGER.error(f'Fit command failed with exit code {exc.returncode}')
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
    LOGGER.info('Fit done, extracting results')
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

def res_saving(
        mu: float,
        err: float,
        res: str
         ) -> None:
    """Save fitting results (signal strength and uncertainty) to output file."""

    # Check if results were successfully extracted
    if (mu==-100) and (err==-100):
        LOGGER.error("Couldn't extract values of the fit, go to the log file to have more information")
        exit(1)
    else:
        # Display results unless suppressed by flag
        if arg.print:
            LOGGER.info('Results successfully extracted\n'
                        f'mu = {mu} +/- {err}')
            LOGGER.info(f'Uncertainty obtained on ZH cross-section: {err*100:.2f} %')

        # Write results to output file
        with open(f'{res}/results{tar}.txt', 'w') as f:
            f.write(f'{mu}\n{err}\n')

        LOGGER.info(f'Saved results in {res}/results{tar}.txt')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = fitting(dir, dc, ws, tp, dc_comb, env)
        if ret != 0:
            sys.exit(ret)

        # Extract results from fit
        mu, err = res_extraction(result_log)

        # Save results to output file
        res_saving(mu, err, res)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

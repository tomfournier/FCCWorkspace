################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, subprocess

from time import time
from uuid import uuid4
from pathlib import Path
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
from package.userConfig import loc, PathObj
loc.set_default_type(Path)
from package.config import timer  # Timing utility



###############################
### DIRECTORY CONFIGURATION ###
###############################

# Set category to combined if combining channels
if arg.lep and arg.combine:
    raise ValueError("Cannot use '--lep' and '--combine' at the same time")
if arg.lep:     arg.cat = 'leptonic'
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
        'tp':  'bias'                  # Type label for naming
    }
}
# Select configuration based on fit mode (nominal vs bias)
config = location_map[arg.bias]

# Resolve directory paths based on configuration and arguments
dr:  PathObj = loc.get(config['dir'], *args)   # Working directory
dc:  PathObj = loc.get(config['dc'],  *args)   # Input datacard directory
ws:  PathObj = loc.get(config['ws'],  *args)   # Workspace directory
log: PathObj = loc.get(config['log'], *args)   # Log output directory
res: PathObj = loc.get(config['res'], *args)   # Results output directory
tp           = config['tp']                         # Type identifier

# Define naming suffixes for file outputs
comb    = '_combined' if arg.combine else ''                         # Combined channel suffix
tar     = f'_{arg.target}' if arg.bias else ''                       # Bias test target suffix (e.g., _bb)
dc_comb = loc.get('COMBINE', '', arg.ecm, arg.sel)  # Combined datacard location

# Define full file paths for workspace, logs, and results
ws_file    = ws  / f'ws{tar}.root'                   # Workspace file (workspace.root or workspace_bb.root)
log_text   = log / f'log_text2workspace{tar}.txt'    # Text2workspace log
result_log = log / f'log_results{tar}.txt'           # Fit results log

# Set up environment for subprocess calls
env = os.environ.copy()

# Create necessary directories if they don't exist
for dir_path in [dc, ws, log, res]:
    dir_path.mkdir(exist_ok=True, parents=True)



####################
### FITTING PART ###
####################

def add_stamp(
        path: PathObj,
        label: str,
        status: str = 'ok'
         ) -> None:
    '''Add a timestamped stamp to a log file for tracking execution status.'''

    # Generate timestamp and unique ID
    ts = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
    uniq = uuid4().hex[:8]
    stamp = f'\n\n---- STAMP {label} [{status}]: {ts} | id={uniq}\n'

    # Append stamp to log file
    with open(path, 'a') as log_file:
        log_file.write(stamp)
    LOGGER.debug(f'Added STAMP: {ts} | id={uniq} | status={status} | file={label}')

def fitting(
        dr: PathObj,
        dc: PathObj,
        ws: PathObj,
        tp: PathObj,
        dc_comb: PathObj,
        env: os._Environ
         ) -> int:
    '''Execute the fitting workflow: combine datacards, create workspace, and run fit.

    Two-stage fitting process:
    Stage 1: Quick fit (--algo none) to find best fit point and save RooFitResult
    Stage 2: Grid scan (--algo grid) using result from stage 1 for faster convergence
    '''

    # Track status of each fitting step
    text_status, fit_status, grid_status = 'not-run', 'not-run', 'not-run'
    try:
        dc_combined = dc / f'datacard{tar}{comb}.txt'

        # Combine datacards from multiple channels if requested
        if arg.lep:
            dc_mu = dc_comb / 'mumu' / tp / 'datacard' / f'datacard{tar}.txt'
            dc_ee = dc_comb /  'ee'  / tp / 'datacard' / f'datacard{tar}.txt'

            LOGGER.info('Combining datacards from ee and mumu channel')
            with open(dc_combined, 'w') as out:
                subprocess.run(['combineCards.py', dc_mu, dc_ee],
                               stdout=out, env=env, check=True)

        if arg.combine:
            dc_mu = dc_comb / 'mumu' / tp / 'datacard' / f'datacard{tar}.txt'
            dc_ee = dc_comb /  'ee'  / tp / 'datacard' / f'datacard{tar}.txt'
            dc_qq = dc_comb /  'qq'  / tp / 'datacard' / f'datacard{tar}.txt'

            LOGGER.info('Combining datacards from all channels')
            with open(dc_combined, 'w') as out:
                subprocess.run(['combineCards.py', dc_mu, dc_ee, dc_qq],
                               stdout=out, env=env, check=True)

        # Convert datacard to RooFit workspace
        with open(log_text, 'w') as log_out:
            LOGGER.info('Setting files for the fit')
            subprocess.run(['text2workspace.py', dc_combined, '-v', '10',
                            '--X-allow-no-background', '-m', '125', '-o', ws_file],
                           stdout=log_out, stderr=subprocess.STDOUT,
                           cwd=dr, env=env, check=True)
        text_status = 'ok'

        # Stage 1: Quick fit with --algo none to find best fit point and save RooFitResult
        with open(result_log, 'w') as log_out:
            LOGGER.info('Stage 1: Quick fit to find best fit point')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '10', '-t', '0', '--expectSignal=1', '-n', f'Xsec{tar}',
                            '--algo', 'none', '--saveFitResult'],
                           stdout=log_out, stderr=subprocess.STDOUT,
                           cwd=ws, env=env, check=True)
        fit_status = 'ok'

        # Stage 2: Grid scan using the fit result from stage 1 for faster convergence
        with open(result_log, 'a') as log_out:
            LOGGER.info('Stage 2: Grid scan for likelihood profile')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '10', '-t', '0', '--expectSignal=1', '-n', f'Xsec{tar}',
                            '--rMin', '0', '--rMax', '2', '--alignEdges', '1', '--squareDistPoiStep',
                            '--algo', 'grid', '--points', '20', '--autoRange', '3', '--skipInitialFit'],
                           stdout=log_out, stderr=subprocess.STDOUT,
                           cwd=ws, env=env, check=True)
        grid_status = 'ok'
        return 0

    except subprocess.CalledProcessError as exc:
        # Mark which step failed
        if   text_status =='not-run': text_status = f'error exit = {exc.returncode}'
        elif fit_status  =='not-run': fit_status  = f'error exit = {exc.returncode}'
        elif grid_status =='not-run': grid_status = f'error exit = {exc.returncode}'
        LOGGER.error(f'Fit command failed with exit code {exc.returncode}')
        return exc.returncode
    finally:
        # Add execution stamps to log files for tracking
        if log_text.exists():
            add_stamp(log_text, f'log_text2workspace{tar}', text_status)
        if result_log.exists():
            add_stamp(result_log, f'log_results{tar}', f'{fit_status},{grid_status}')



##########################
### RESULTS EXTRACTION ###
##########################

def root_extraction(
        file: str
         ) -> tuple[float,
                    float]:
    '''Extract signal strength (mu) and uncertainty from RooFitResult ROOT file.

    This is the primary extraction method using the RooFitResult object from
    the first fit stage (--algo none --saveFitResult). Uses ROOT for reliable deserialization.
    Falls back to log file parsing if ROOT is unavailable.
    '''
    LOGGER.debug('Extracting results from RooFitResult')
    try:
        import ROOT
        # Convert PathObj to string if necessary
        file = ROOT.TFile.Open(str(file))
        if not file or file.IsZombie():
            LOGGER.warning(f'Could not open {file}')
            return -100, -100

        fit_result = file.Get('fit_mdf')
        if not fit_result:
            LOGGER.warning(f'Could not find fit_mdf in {file}')
            file.Close()
            return -100, -100

        # Get the parameter of interest (r)
        pars = fit_result.floatParsFinal()
        r_param = pars.find('r')

        if not r_param:
            LOGGER.warning('Could not find parameter r in fit result')
            file.Close()
            return -100, -100

        mu, err = r_param.getVal(), r_param.getError()

        LOGGER.debug('Successfully extracted from RooFitResult')
        file.Close()
        return mu, err

    except ImportError:
        LOGGER.warning('ROOT not available, falling back to log file extraction')
        return -100, -100
    except Exception as e:
        LOGGER.warning(f'Failed to extract from ROOT file: {e}')
        return -100, -100

def res_extraction(res_log: str
                   ) -> tuple[float,
                              float]:
    '''Extract signal strength (mu) and uncertainty from fit results log.

    This is a fallback method that parses the log file for the fit result.
    Used only if RooFitResult extraction fails.
    '''
    LOGGER.info('Fit done, extracting results')
    mu, err = -100, -100  # Initialize with error values

    # Parse log file for signal strength result
    # (parse from end to find latest result)
    with open(str(res_log)) as file:
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
    '''Save fitting results (signal strength and uncertainty) to output file.'''

    # Check if results were successfully extracted
    if (mu==-100) and (err==-100):
        LOGGER.error("Couldn't extract values of the fit, go to the log file to have more information")
        exit(1)
    else:
        # Display results unless suppressed by flag
        if arg.print:
            LOGGER.info('Results successfully extracted\n'
                        f'mu = {mu:6f} +/- {err:6f}')
            LOGGER.info(f'Uncertainty obtained on ZH cross-section: {err*100:.2f} %')

        # Write results to output file
        with open(str(res) + f'/results{tar}.txt', 'w') as f:
            f.write(f'{mu}\n{err}\n')

        LOGGER.debug(f'Saved results in {res}/results{tar}.txt')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = fitting(dr, dc, ws, tp, dc_comb, env)
        if ret != 0: sys.exit(ret)

        # Extract results using primary method: log file extraction
        LOGGER.info('Falling back to log file extraction')
        mu, err = res_extraction(result_log)

        # Fallback to ROOFitResult extraction if log file extraction fails
        if (mu == -100) and (err == -100):
            file = ws / f'multidimfitXsec{tar}.root'
            mu, err = root_extraction(file)

        # Save results to output file
        res_saving(mu, err, res)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, sys, subprocess

from time import time
from pathlib import Path

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
from package.func.fit import (
    process_scan,
    add_stamp,
    check_log,
    res_saving
)



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
        'dir':  'COMBINE_NOMINAL',      # Main working directory
        'dc':   'NOMINAL_DATACARD',     # Input datacard directory
        'ws':   'NOMINAL_WS',           # Workspace output directory
        'log':  'NOMINAL_LOG',          # Log file directory
        'res':  'NOMINAL_RESULT',       # Fit results output directory
        'scan': 'NOMINAL_FASTSCAN',     # Check the fit stability
        'tp':   'nominal',              # Type label for naming
    },
    True: {   # Bias test fit (for testing fit bias)
        'dir':  'COMBINE_BIAS',         # Bias test working directory
        'dc':   'BIAS_DATACARD',        # Input biased datacard directory
        'ws':   'BIAS_WS',              # Workspace output directory
        'log':  'BIAS_LOG',             # Log file directory
        'res':  'BIAS_FIT_RESULT',      # Fit results output directory
        'scan': 'BIAS_FASTSCAN',        # Check the fit stability
        'tp':   'bias',                 # Type label for naming
    }
}
# Select configuration based on fit mode (nominal vs bias)
config = location_map[arg.bias]

# Resolve directory paths based on configuration and arguments
dr:   PathObj = loc.get(config['dir'],  *args)   # Working directory
dc:   PathObj = loc.get(config['dc'],   *args)   # Input datacard directory
ws:   PathObj = loc.get(config['ws'],   *args)   # Workspace directory
log:  PathObj = loc.get(config['log'],  *args)   # Log output directory
res:  PathObj = loc.get(config['res'],  *args)   # Results output directory
scan: PathObj = loc.get(config['scan'], *args)   # Fast scan directory
tp           = config['tp']                           # Type identifier

# Define naming suffixes for file outputs
comb    = '_combined' if arg.combine else ''                         # Combined channel suffix
tar     = f'_{arg.target}' if arg.bias else ''                       # Bias test target suffix (e.g., _bb)
dc_comb = loc.get('COMBINE', '', arg.ecm, arg.sel)  # Combined datacard location

# Define full file paths for workspace, logs, and results
ws_file     = ws  / f'ws{tar}.root'                   # Workspace file (workspace.root or workspace_bb.root)
log_text    = log / f'log_text2workspace{tar}.txt'    # Text2workspace log
result_scan = log / f'log_fastscan{tar}.txt'          # Scan results log (scan results)
result_fit  = log / f'log_results{tar}_fit.txt'       # Fit  results log (fit results)

# Set up environment for subprocess calls
env = os.environ.copy()

# Create necessary directories if they don't exist
for dir_path in [dc, ws, log, res, scan]:
    dir_path.mkdir(exist_ok=True, parents=True)



####################
### FITTING PART ###
####################

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
    text_status, scan_status, fit_status = 'not-run', 'not-run', 'not-run'
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

        if arg.fastscan:
            with open(result_scan, 'w') as log_scan:
                LOGGER.info('Doing a likelyhood scan to check the fit')
                subprocess.run(['combineTool.py', '-M', 'FastScan', '-w', str(ws_file)+':w'],
                               stdout=log_scan, stderr=subprocess.STDOUT,
                               cwd=scan, env=env, check=True)
            scan_status = 'ok'

        # Do the fit and a grid scan for likelyhood scan
        with open(result_fit, 'w') as log_out:
            LOGGER.info('Doing the fit')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '2', '-t', str(arg.toy), '--expectSignal=1', '-n', f'Xsec{tar}',
                            '--rMin', '0.9', '--rMax', '1.1', '--autoRange', '5'
                            '--alignEdges', '1', '--squareDistPoiStep',
                            '--algo', 'grid', '--points', '100'],
                           stdout=log_out, stderr=subprocess.STDOUT,
                           cwd=ws, env=env, check=True)

        fit_status = 'ok'
        return 0

    except subprocess.CalledProcessError as exc:
        # Mark which step failed
        if   text_status =='not-run': text_status = f'error exit = {exc.returncode}'
        elif fit_status  =='not-run': fit_status  = f'error exit = {exc.returncode}'
        LOGGER.error(f'Fit command failed with exit code {exc.returncode}')
        return exc.returncode
    finally:
        # Add execution stamps to log files for tracking
        if log_text.exists():
            add_stamp(log_text, f'log_text2workspace{tar}', text_status)
        if result_scan.exists() and arg.fastscan:
            add_stamp(result_scan, f'log_fastscan{tar}', scan_status)
        if result_fit.exists():
            add_stamp(result_fit, f'log_results{tar}_fit', fit_status)


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = fitting(dr, dc, ws, tp, dc_comb, env)
        if ret != 0: sys.exit(ret)

        LOGGER.info('Fit done, extracting results')

        # Check if the fit went well
        fit_status  = check_log(result_fit)
        out_file = f'higgsCombineXsec{tar}.MultiDimFit.mH125.123456.root' if arg.toy>0 \
            else f'higgsCombineXsec{tar}.MultiDimFit.mH125.root'
        mu, err_h, err_l = process_scan(ws / out_file,
                                        'r', 10, True)

        # Save results to output file
        res_saving(mu, [err_h, err_l],  res, arg.print, tar)

        cmd = ['python', 'plots.py', '--ecm', str(arg.ecm), '--sels', str(arg.sel), '--no-timer', '--sig2']
        if arg.lep:       cmd.append('--lep')
        elif arg.combine: cmd.append('--comb')
        elif arg.cat:     cmd.extend(['--cat', arg.cat])
        if arg.toy>0:     cmd.append('--toy')
        if arg.bias:
            cmd.append('--bias')
            cmd.append('--only1')
            cmd.extend(['--target', arg.target])

        status = subprocess.run(cmd, cwd=Path(__file__).parent,
                                env=env, check=False,
                                capture_output=False, text=True)
        if status != 0: exit(status.returncode)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

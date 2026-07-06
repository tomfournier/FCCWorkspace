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
    no_ecm=True,           # Do not include ecm argument
    include_sel=True,      # Include selection strategy options
    fit=True,              # Include fit-specific options
    is_nlo=True,           # Include NLO fit-specific options
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
args = [arg.cat, 0, arg.sel]

# Resolve directory paths based on configuration and arguments
dr:   PathObj = loc.get('NLO',          *args)   # Working directory
dc:   PathObj = loc.get('NLO_DATACARD', *args)   # Input datacard directory
ws:   PathObj = loc.get('NLO_WS',       *args)   # Workspace directory
log:  PathObj = loc.get('NLO_LOG',      *args)   # Log output directory
res:  PathObj = loc.get('NLO_RESULT',   *args)   # Results output directory
scan: PathObj = loc.get('NLO_FASTSCAN', *args)   # Fast scan directory

# Define naming suffixes for file outputs
comb    = '_combined' if arg.combine else ''                     # Combined channel suffix

# Define full file paths for workspace, logs, and results
ws_file     = ws  / 'ws.root'                   # Workspace file
log_text    = log / 'log_text2workspace.txt'    # Text2workspace log
result_scan = log / 'log_fastscan.txt'          # Scan results log (scan results)
result_fit  = log / 'log_results_fit.txt'       # Fit  results log (fit results)

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
        env: os._Environ,
        params: list[str]
         ) -> int:
    '''Execute the fitting workflow: combine datacards, create workspace, and run fit.

    Two-stage fitting process:
    Stage 1: Quick fit (--algo none) to find best fit point and save RooFitResult
    Stage 2: Grid scan (--algo grid) using result from stage 1 for faster convergence
    '''

    # Track status of each fitting step
    text_status, scan_status, fit_status = 'not-run', 'not-run', 'not-run'
    try:
        dc_combined = dc / f'datacard{comb}.txt'

        dc_240 = loc.get('NOMINAL_DATACARD', arg.cat, 240, arg.sel) / f'datacard{comb}.txt'
        dc_365 = loc.get('NOMINAL_DATACARD', arg.cat, 365, arg.sel) / f'datacard{comb}.txt'

        LOGGER.info('Combining datacards from 240 and 365 GeV channels')
        with open(dc_combined, 'w') as out:
            subprocess.run(['combineCards.py', f'low={dc_240}', f'high={dc_365}'],
                           stdout=out, env=env, check=True)

        # Convert datacard to RooFit workspace
        with open(log_text, 'w') as log_out:
            LOGGER.info('Setting files for the fit')
            subprocess.run(['text2workspace.py', dc_combined, '-v', '10',
                            '-P', f'package.func.self_coupling:{arg.model}',
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
        expected = ','.join(f'{p}=0' for p in params)
        ranges   = ':'.join(f'{p}=-1,1' if arg.combine else f'{p}=-3,3' for p in params)
        with open(result_fit, 'w') as log_out:
            LOGGER.info('Doing the fit')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '2', '-t', str(arg.toy), '-n', 'Xsec', '--setParameters', expected,
                            '--alignEdges', '1', '--robustHesse=1',
                            '--setParameterRanges', ranges,
                            '--algo', 'grid', '--points', '400' if len(params) > 1 else '100'],
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
            add_stamp(log_text, 'log_text2workspace', text_status)
        if result_scan.exists() and arg.fastscan:
            add_stamp(result_scan, 'log_fastscan', scan_status)
        if result_fit.exists():
            add_stamp(result_fit, 'log_results_fit', fit_status)


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Get the fitted parameters
        params = arg.model.replace('SMEFT_', '').split('_')

        # Execute the fitting pipeline
        ret = fitting(dr, dc, ws, env, params)
        if ret != 0: sys.exit(ret)

        LOGGER.info('Fit done, extracting results')

        # Check if the fit went well
        fit_status  = check_log(result_fit)
        out_file = 'higgsCombineXsec.MultiDimFit.mH125.123456.root' if arg.toy>0 \
            else 'higgsCombineXsec.MultiDimFit.mH125.root'

        for param in params:
            mu, err_h, err_l = process_scan(ws / out_file,
                                            param, 10, True)

            # Save results to output file
            res_saving(mu, [err_h, err_l],  res, arg.print, param, param)

        cmd = ['python', 'plots.py', '--sels', str(arg.sel), '--no-timer', '--sig2', '--param', '-'.join(params)]
        if arg.lep:       cmd.append('--lep')
        elif arg.combine: cmd.append('--comb')
        elif arg.cat:     cmd.extend(['--cat', arg.cat])
        if arg.toy>0:     cmd.append('--toy')

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

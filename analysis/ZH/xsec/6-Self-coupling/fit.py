################################
### STANDARD LIBRARY IMPORTS ###
################################

import os

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
    res_saving,
    run_cmd,
    get_results,
    get_grid_number,
    convert_to_kappa
)
from package.func.self_coupling import get_parameters



###############################
### VARIABLES CONFIGURATION ###
###############################

# Get the fitted parameters
params: list[str] = get_parameters(arg.model)
expected = ','.join(f'{p}=0' for p in params)

ranges = {
    'tight': {'Cphi': '-3,3', 'CphiD': '-0.8,0.8', 'Cbox': '-3,3'},
    'loose': {'Cphi': '-5,5', 'CphiD': '-2,2', 'Cbox': '-5,5'}
}
if arg.combine:
    param_ranges = ':'.join(f'{p}={ranges["tight"][p]}' for p in params)
else:
    param_ranges = ':'.join(f'{p}={ranges["loose"][p]}' for p in params)



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
comb = '_combined' if arg.combine else ''  # Combined channel suffix

# Define full file paths for workspace, logs, and results
ws_file   = ws  / 'ws.root'                                   # Workspace file
diag_file = ws  / 'higgsCombineDiag.MultiDimFit.mH125.root'   # Diagnostic fit file
fit_file  = ws  / 'higgsCombineXsec.MultiDimFit.mH125.root'   # Likelihood scan file

log_t2w   = log / 'log_text2workspace.txt'    # Text2workspace log
log_fscan = log / 'log_fastscan.txt'          # Scan results log (scan results)
log_diag  = log / 'log_diagnostic.txt'        # Diagnostic results log (for fit results)
log_fit   = log / 'log_results_fit.txt'       # Fit results log (fit results)

# Datacard file definition
dc_nom = dc / f'datacard{comb}.txt'
dc_240 = loc.get('NOMINAL_DATACARD', arg.cat, 240, arg.sel) / f'datacard{comb}.txt'
dc_365 = loc.get('NOMINAL_DATACARD', arg.cat, 365, arg.sel) / f'datacard{comb}.txt'

# Set up environment for subprocess calls
env = os.environ.copy()

# Create necessary directories if they don't exist
for dir_path in [dc, ws, log, res, scan]:
    dir_path.mkdir(exist_ok=True, parents=True)



####################
### FITTING PART ###
####################

def do_fit(
        dr: PathObj,
        ws: PathObj,
        env: os._Environ,
        params: list[str]
         ) -> int:
    '''Execute the fitting workflow: combine datacards, create workspace, and run fit.'''

    ##########################
    ### Command definition ###
    ##########################

    cmd_dc = ['combineCards.py', f'low={dc_240}', f'high={dc_365}']

    cmd_t2w = ['text2workspace.py', dc_nom, '-v', '2', '-P',
               f'package.func.self_coupling:{arg.model}',
               '--X-allow-no-signal', '--X-allow-no-background',
               '--for-fits', '--no-wrappers',
               '-m', '125', '-o', ws_file]

    cmd_fastscan = ['combineTool.py', '-M', 'FastScan', '-w', str(ws_file)+':w']

    cmd_diag = ['combine', ws_file, '-M', 'MultiDimFit', '-m', '125', '-t', '-1', '-v', '2', '-n', 'Diag',
                '--algo', 'singles', '--cl=0.68', '--cminDefaultMinimizerStrategy=0', '--saveWorkspace',
                '--setParameters', expected, '--setParameterRanges', param_ranges,
                '--cminPreFit', '1', '--cminInitialHesse', '1', '--robustFit=1']

    cmd_fit = ['combine', diag_file, '-M', 'MultiDimFit', '-m', '125', '-v', '2', '-t', '-1', '-n', 'Xsec',
               '--setParameters', expected, '--setParameterRanges', param_ranges, '-w', 'w',
               '--autoBoundsPOIs', '*', '--autoMaxPOIs', '*', '--snapshotName', 'MultiDimFit', '--alignEdges', '1',
               '--squareDistPoiStep', '--autoRange', '6', '--skipInitialFit', '--algo', 'grid',
               '--points', f'{get_grid_number(arg.points, params)}']
    cmd_fit += ['--fastScan'] if arg.fast_scan else []



    #########################
    ### COMMAND EXECUTION ###
    #########################

    if not (arg.skip_setup and dc_nom.exists()):
        LOGGER.info('Combining datacards from 240 and 365 GeV channels')
        run_cmd(cmd_dc, dc_nom, None, env)
    else:
        LOGGER.debug('Skipping the datacard setup')

    if not (arg.skip_setup and ws_file.exists()):
        LOGGER.info('Converting the datacard to RooFit workspace')
        run_cmd(cmd_t2w, log_t2w, dr, env)
    else:
        LOGGER.debug('Skipping the RooFit workspace setup')

    if arg.fastscan:
        LOGGER.info('Doing a fast likelyhood scan to check the fit')
        run_cmd(cmd_fastscan, log_fscan, scan, env)

    LOGGER.info('Doing the fit')
    run_cmd(cmd_diag, log_diag, ws, env)
    res_diag = get_results(diag_file, params, 'singles')
    res_saving(res_diag, res, arg.print, '_diag')
    convert_to_kappa(res_diag)

    if arg.only_diag:
        LOGGER.debug('Skipping the likelihood scan')
    else:
        LOGGER.info('Doing a likelihood scan')
        run_cmd(cmd_fit, log_fit, ws, env)
        res_fit = get_results(fit_file, params, 'grid')
        res_saving(res_fit, res, arg.print, '_fit')
        convert_to_kappa(res_fit)

    return 0


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = do_fit(dr, ws, env, params)
        if ret != 0: exit(ret)

        cmd = ['python', 'plots.py', '--sels', str(arg.sel), '--no-timer', '--sig2',
               '--param', '-'.join(params), '--y-cut', '100', '--y-max', '7']
        cmd += ['--lep'] if arg.lep else (['--comb'] if arg.combine else ['--cat', arg.cat])
        run_cmd(cmd, None, Path(__file__).parent, env)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

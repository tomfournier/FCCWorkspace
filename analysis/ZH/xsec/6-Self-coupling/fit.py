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
    process_scan,
    check_log,
    res_saving,
    run_cmd,
    get_grid_number
)
from package.func.self_coupling import get_parameters



###############################
### VARIABLES CONFIGURATION ###
###############################

ranges = {
    'tight': {'Cphi': '-3,3', 'CphiD': '-0.8,0.8', 'Cbox': '-3,3'},
    'loose': {'Cphi': '-5,5', 'CphiD': '-2,2', 'Cbox': '-5,5'}
}
Ranges = {'Cphi': [-2, 2], 'CphiD': [-1, 1], 'Cbox': [-2, 2]}
ngrids = {'Cphi': 10,      'CphiD': 10,      'Cbox': 10}

gridpoint = {
    'Cphi':  {'low': '11', 'med': '21', 'high': '31'},
    'CphiD': {'low': '11', 'med': '21', 'high': '31'},
    'Cbox':  {'low': '11', 'med': '21', 'high': '31'},
}

# Get the fitted parameters
params: list[str] = get_parameters(arg.model)
expected = ','.join(f'{p}=0' for p in params)
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
ws_file     = ws  / 'ws.root'                   # Workspace file
sn_file     = ws  / 'higgsCombineDiag.MultiDimFit.mH125.root'                   # Snapshot file
log_text    = log / 'log_text2workspace.txt'    # Text2workspace log
result_scan = log / 'log_fastscan.txt'          # Scan results log (scan results)
diagnostic_fit = log / 'log_diagnostic.txt'        # Diagnostic results log (for fit results)
diagnostic_fit = log / 'log_diagnostic.txt'        # Diagnostic results log (for fit results)
result_fit  = log / 'log_results_fit.txt'       # Fit results log (fit results)

# Datacard file definition
dc_combined = dc / f'datacard{comb}.txt'
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

    cmd_t2w = ['text2workspace.py', dc_combined, '-v', '2', '-P',
               f'package.func.self_coupling:{arg.model}',
               '--X-allow-no-signal', '--X-allow-no-background',
               '--for-fits', '--no-wrappers',
               '-m', '125', '-o', ws_file]

    cmd_fastscan = ['combineTool.py', '-M', 'FastScan', '-w', str(ws_file)+':w']

    cmd_diag = ['combine', ws_file, '-M', 'MultiDimFit', '-m', '125', '-t', '-1', '-v', '2', '-n', 'Diag',
                '--algo', 'singles', '--cl=0.68', '--cminDefaultMinimizerStrategy=0', '--saveWorkspace',
                '--setParameters', expected, '--setParameterRanges', param_ranges,
                '--cminPreFit', '1', '--cminInitialHesse', '1', '--robustFit=1']

    cmd_fit = ['combine', sn_file, '-M', 'MultiDimFit', '-m', '125', '-v', '2', '-t', '-1', '-n', 'Xsec',
               '--setParameters', expected, '--setParameterRanges', param_ranges, '-w', 'w', '--fastScan',
               '--snapshotName', 'MultiDimFit', '--alignEdges', '1', '--squareDistPoiStep',
               '--skipInitialFit', '--algo', 'grid', '--points', f'{get_grid_number(30, params)}']



    #########################
    ### COMMAND EXECUTION ###
    #########################

    LOGGER.info('Combining datacards from 240 and 365 GeV channels')
    run_cmd(cmd_dc, dc_combined, None, env)

    LOGGER.info('Converting the datacard to RooFit workspace')
    run_cmd(cmd_t2w, log_text, dr, env)

    if arg.fastscan:
        LOGGER.info('Doing a likelyhood scan to check the fit')
        run_cmd(cmd_fastscan, result_scan, scan, env)

    LOGGER.info('Doing diagnostic fit')
    run_cmd(cmd_diag, diagnostic_fit, ws, env)

    LOGGER.info('Doing the fit')
    run_cmd(cmd_fit, result_fit, ws, env)

    return 0


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = do_fit(dr, ws, env, params)
        if ret != 0: exit(ret)

        LOGGER.info('Fit done, extracting results')

        # Check if the fit went well
        fit_status  = check_log(result_fit)
        out_file = 'higgsCombineXsec.MultiDimFit.mH125.root'

        for param in params:
            other_params = [p for p in params if p!=param] if len(params)>1 else []
            mu, err_h, err_l = process_scan(
                ws / out_file, param,
                1000, True, other_params
            )

            # Save results to output file
            res_saving(mu, [err_h, err_l],  res, arg.print, param, param)

        cmd = ['python', 'plots.py', '--sels', str(arg.sel),
               '--no-timer', '--sig2', '--param', '-'.join(params),
               '--y-cut', '100', '--y-max', '7']
        if arg.lep:       cmd.append('--lep')
        elif arg.combine: cmd.append('--comb')
        elif arg.cat:     cmd.extend(['--cat', arg.cat])
        if arg.toy>0:     cmd.append('--toy')

        status = run_cmd(cmd, None, Path(__file__).parent, env)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

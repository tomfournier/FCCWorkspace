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

from parsing import set_log, parse_args
from zh_xsec.parsing import create_parser
from logger import get_logger
parser = create_parser('6-Self-coupling', 'fit')
arg = parse_args(parser, comb=True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory path manager and utilities
from path import PathObj
from zh_xsec.userConfig import loc
loc.set_default_type(Path)
from tools.utils import timer
from fitting import (
    res_saving, run_cmd, parse_sels,
    get_results, get_grid_number,
)
from run import get_extra_args, update_namespace
from zh_xsec.func.self_coupling import get_params, to_kappa



###############################
### VARIABLES CONFIGURATION ###
###############################

# Get the fitted parameters
params: list[str] = get_params(arg.model)
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
_, sel = parse_sels(arg.sel, arg.cat, bool(arg.lep or arg.combine))

# Bundle arguments for directory lookup
args = [arg.cat, 0, sel]
lumi = (10.8 if arg.ecm==240 else (3.12 if arg.ecm==365 else -1)) if arg.rescaled else 1

# Resolve directory paths based on configuration and arguments
dr:   PathObj = loc.get('SELF',        *args)   # Working directory
dc:   PathObj = loc.get('SELF_CARD',   *args)   # Input datacard directory
ws:   PathObj = loc.get('SELF_WS',     *args)   # Workspace directory
log:  PathObj = loc.get('SELF_LOG',    *args)   # Log output directory
res:  PathObj = loc.get('SELF_RESULT', *args)   # Results output directory
scan: PathObj = loc.get('SELF_SCAN',   *args)   # Fast scan directory

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
dc_240 = loc.get('FIT_CARD', arg.cat, 240, sel) / f'datacard{comb}.txt'
dc_365 = loc.get('FIT_CARD', arg.cat, 365, sel) / f'datacard{comb}.txt'

# Set up environment for subprocess calls
ENV = os.environ.copy()

# Create necessary directories if they don't exist
for dir_path in [dc, ws, log, res, scan]:
    dir_path.mkdir(exist_ok=True, parents=True)



####################
### FITTING PART ###
####################

def do_fit(
        dr: PathObj,
        ws: PathObj,
        params: list[str]
) -> int:
    '''Execute the fitting workflow: combine datacards, create workspace, and run fit.'''

    ##########################
    ### COMMAND DEFINITION ###
    ##########################

    cmd_dc = ['combineCards.py', f'low={dc_240}', f'high={dc_365}']

    cmd_t2w = ['text2workspace.py', dc_nom, '-v', '2', '-P',
               f'zh_xsec.func.self_coupling:{arg.model}',
               '--X-allow-no-signal', '--X-allow-no-background',
               '--for-fits', '--no-wrappers',
               '-m', '125', '-o', ws_file]

    cmd_fastscan = ['combineTool.py', '-M', 'FastScan', '-w', str(ws_file)+':w']

    cmd_diag = ['combine', ws_file, '-M', 'MultiDimFit', '-m', '125', '-t', '-1', '-v', '2', '-n', 'Diag',
                '--algo', 'singles', '--cl=0.68', '--cminDefaultMinimizerStrategy=0', '--saveWorkspace',
                '--setParameters', expected, '--setParameterRanges', param_ranges, '--robustHesse=1',
                '--cminPreFit', '1', '--cminInitialHesse', '1', '--robustFit=1']

    cmd_fit = ['combine', diag_file, '-M', 'MultiDimFit', '-m', '125', '-v', '2', '-t', '-1', '-n', 'Xsec',
               '--setParameters', expected, '--setParameterRanges', param_ranges, '-w', 'w', '--robustHesse=1',
               '--autoBoundsPOIs', '*', '--autoMaxPOIs', '*', '--snapshotName', 'MultiDimFit',
               '--squareDistPoiStep', '--autoRange', '10', '--skipInitialFit', '--algo', 'grid',
               '--alignEdges', '1', '--points', f'{get_grid_number(arg.npoints, params)}']
    cmd_fit += ['--fastScan'] if arg.fast_scan else []



    #########################
    ### COMMAND EXECUTION ###
    #########################

    if not (arg.skip_setup and dc_nom.exists()):
        LOGGER.info('Combining datacards from 240 and 365 GeV channels')
        run_cmd(cmd_dc, dc_nom, None, ENV)
    else:
        LOGGER.debug('Skipping the datacards combination')

    if not (arg.skip_setup and ws_file.exists()):
        LOGGER.info('Converting the datacard to RooFit workspace')
        run_cmd(cmd_t2w, log_t2w, dr, ENV)
    else:
        LOGGER.debug('Skipping the RooFit workspace setup')

    if arg.fastscan:
        LOGGER.info('Doing a fast likelyhood scan to check the fit')
        run_cmd(cmd_fastscan, log_fscan, scan, ENV)

    LOGGER.info('Doing the fit')
    run_cmd(cmd_diag, log_diag, ws, ENV)
    res_diag = get_results(diag_file, params, 'singles', lumi)
    res_saving(res_diag, res, arg.print, '_diag')
    to_kappa(res_diag, res, '_diag')

    if arg.only_diag:
        LOGGER.debug('Skipping the likelihood scan')
    else:
        LOGGER.info('Doing the likelihood scan')
        run_cmd(cmd_fit, log_fit, ws, ENV)
        res_fit = get_results(fit_file, params, 'grid', lumi)
        res_saving(res_fit, res, arg.print, '_fit')
        to_kappa(res_fit, res, '_fit')

    return 0


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = do_fit(dr, ws, params)
        if ret != 0: exit(ret)

        plot_args = update_namespace(arg, cat=None if (arg.lep or arg.combine) else arg.cat, sels=sel,
                                     sig2=True, timer=False, param='-'.join(params), y_cut=100, y_max=7)
        cmd = ['python', 'plots.py'] + get_extra_args(
            plot_args, {f'directory': '6-Self-coupling', 'script': 'plots'}, create_parser)
        run_cmd(cmd, None, Path(__file__).parent, ENV)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

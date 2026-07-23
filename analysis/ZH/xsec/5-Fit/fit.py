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
    get_results,
    res_saving,
    run_cmd
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
tp            = config['tp']                          # Type identifier

# Define naming suffixes for file outputs
comb    = '_combined' if arg.combine else ''                     # Combined channel suffix
tar     = f'_{arg.target}' if arg.bias else ''                   # Bias test target suffix (e.g., _bb)
dcs = loc.get('COMBINE', '', arg.ecm, arg.sel)  # Combined datacard location

# Define full file paths for workspace, logs, and results
ws_file   = ws  / f'ws{tar}.root'                                   # Workspace file (workspace.root or workspace_bb.root)
diag_file = ws  / f'higgsCombineDiag{tar}.MultiDimFit.mH125.root'   # Diagnostic fit file
fit_file  = ws  / f'higgsCombineXsec{tar}.MultiDimFit.mH125.root'   # Likekihood scan file

log_t2w   = log / f'log_text2workspace{tar}.txt'    # Text2workspace log
log_fscan = log / f'log_fastscan{tar}.txt'          # Scan results log (scan results)
log_diag  = log / f'log_diagnostic{tar}.txt'        # Diagnostic results log (for fit results)
log_fit   = log / f'log_results{tar}_fit.txt'       # Fit  results log (fit results)

# Datacard file definition
dc_nom = dc / f'datacard{tar}{comb}.txt'
dc_ee  = dcs /  'ee'  / tp / 'datacard' / f'datacard{tar}.txt'
dc_mu  = dcs / 'mumu' / tp / 'datacard' / f'datacard{tar}.txt'
dc_qq  = dcs /  'qq'  / tp / 'datacard' / f'datacard{tar}.txt'

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
        env: os._Environ
         ) -> int:
    '''Execute the fitting workflow: combine datacards, create workspace, and run fit.

    Two-stage fitting process:
    Stage 1: Quick fit (--algo none) to find best fit point and save RooFitResult
    Stage 2: Grid scan (--algo grid) using result from stage 1 for faster convergence
    '''

    ##########################
    ### COMMANDS DEFINTION ###
    ##########################

    cmd_comb  = ['combineCards.py', f'ee={dc_ee}', f'mumu={dc_mu}']
    cmd_comb += [f'qq={dc_qq}'] if arg.combine else []

    cmd_t2w = ['text2workspace.py', dc_nom, '-v', '10',
               '--X-allow-no-signal', '--X-allow-no-background',
               '--for-fits', '--no-wrappers', '-m', '125', '-o', ws_file]

    cmd_diag = ['combine', ws_file, '-M', 'MultiDimFit', '-m', '125', '-v', '2', '-t', '-1',
                '--algo', 'singles', '--cl=0.68', '--robustFit=1', '--expectSignal=1',
                '--cminDefaultMinimizerStrategy=0', '--saveWorkspace',
                '--rMin', '0.9', '--rMax', '1.1', '-n', f'Diag{tar}']

    cmd_fastscan = ['combineTools.py', '-M', 'FastScan', '-w', str(ws_file)+':w']

    cmd_fit = ['combine', diag_file, '-M', 'MultiDimFit', '-m', '125', '-v', '2', '-t', '-1',
               '--expectSignal=1', '-n', f'Xsec{tar}', '-w', 'w', '--snapshotName', 'MultiDimFit',
               '--autoRange', '5', '--alignEdges', '1', '--squareDistPoiStep',
               '--algo', 'grid', '--points', '200', '--skipInitialFit']
    cmd_fit += ['--fastScan'] if arg.fast_scan else []



    ##########################
    ### COMMANDS EXECUTION ###
    ##########################

    if (arg.lep or arg.combine) and not (arg.skip_setup and dc_nom.exists()):
        channels = 'ee and mumu' if arg.lep else ('ee, mumu and qq' if arg.comb else '')
        LOGGER.info(f'Combining datacards for {channels} channels')
        run_cmd(cmd_comb, dc_nom, None, env)
    else:
        LOGGER.debug('Skipping the datacards combination')

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
    res_diag = get_results(diag_file, 'r', 'singles')
    res_saving(res_diag, res, arg.print, '_diag')

    if arg.only_diag:
        LOGGER.debug('Skipping the likelihood scan')
    else:
        LOGGER.info('Doing a likelihood scan')
        run_cmd(cmd_fit, log_fit, ws, env)
        res_fit = get_results(fit_file, 'r', 'grid')
        res_saving(res_fit, res, arg.print, '_fit')

    return 0


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Execute the fitting pipeline
        ret = do_fit(dr, ws, env)
        if ret != 0: exit(ret)

        cmd = ['python', 'plots.py', '--ecm', str(arg.ecm),
               '--sels', str(arg.sel), '--no-timer', '--sig2']
        cmd += ['--lep'] if arg.lep else (['--comb'] if arg.combine else ['--cat', arg.cat])
        cmd += ['--bias', '--only1', '--target', arg.target] if arg.bias else []

        run_cmd(cmd, None, Path(__file__).parent, env)

    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time if requested
        if arg.timer: timer(t)

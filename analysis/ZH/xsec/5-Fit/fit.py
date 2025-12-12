import os, uuid, time, argparse, importlib, subprocess
from datetime import datetime
t = time.time()

userConfig = importlib.import_module('userConfig')
from package.userConfig import loc, get_loc

from package.config import timer, warning
from package.tools.utils import mkdir



########################
### ARGUMENT PARSING ###
########################

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

parser.add_argument('--pert',   help='Target pseudodata size', 
                    type=float, default=1.0)
parser.add_argument('--target', help='Target pseudodata', 
                    type=str, default='')

parser.add_argument('--bias',    help='Nominal fit or bias test', action='store_true')
parser.add_argument('--combine', help='Combine the channel to do the fit', action='store_true')
parser.add_argument('--t',       help='Compute the elapsed time to run the code', action='store_true')
parser.add_argument('--noprint', help='Do not display the uncertainty', action='store_true')
arg = parser.parse_args()

if arg.cat=='' and not arg.combine:
    msg = 'Final state or combine were not selected, please select one to run this code'
    warning(msg)

if arg.combine: arg.cat = 'combined'
args = [arg.cat, arg.ecm, arg.sel]



###############################
### DIRECTORY CONFIGURATION ###
###############################

location_map = {
    False: {
        'dir': loc.COMBINE_NOMINAL,
        'dc':  loc.NOMINAL_DATACARD,
        'ws':  loc.NOMINAL_WS,
        'log': loc.NOMINAL_LOG,
        'res': loc.NOMINAL_RESULT,
        'tp':  'nominal'
    },
    True: {
        'dir': loc.COMBINE_BIAS,
        'dc':  loc.BIAS_DATACARD,
        'ws':  loc.BIAS_WS,
        'log': loc.BIAS_LOG,
        'res': loc.BIAS_FIT_RESULT,
        'tp': 'bias'
    }
}
config = location_map[arg.bias]
dir, dc = get_loc(config['dir'], *args), get_loc(config['dc'],  *args)
ws, log = get_loc(config['ws'],  *args), get_loc(config['log'], *args)
res, tp = get_loc(config['res'], *args), config['tp']

comb = '_combined' if arg.combine else ''
tar = f'_{arg.target}' if arg.bias else ''
dc_comb = get_loc(loc.COMBINE, '', arg.ecm, arg.sel)

ws_file    = f'{ws}/ws{tar}.root'
log_text   = f'{log}/log_text2workspace{tar}.txt'
result_log = f'{log}/log_results{tar}.txt'

env = os.environ.copy()
setup_script = f'{loc.ROOT}/../../../HiggsAnalysis/CombinedLimit'
# result = subprocess.run(
#     f'source ./env_standalone.sh && python -c "import os, json; print(json.dumps(dict(os.environ)))"',
#     shell=True,
#     capture_output=True,
#     text=True,
#     cwd=setup_script,
#     executable='/bin/bash'
# )
# # print(env)
# print(result.stdout)
# if result.returncode==0:
#     import json
#     # print(env)
#     env.update(json.loads(result.stdout))

for dir_path in [dc, ws, log, res]:
    mkdir(dir_path)



####################
### FITTING PART ###
####################

def add_stamp(path: str, 
              label: str,
              status: str = 'ok'
              ) -> None:
    
    ts = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
    uniq = uuid.uuid4().hex[:8]
    stamp = f'\n\n---- STAMP {label} [{status}]: {ts} | id={uniq}\n'

    with open(path, 'a') as log_file:
        log_file.write(stamp)
    print(f'----->[Info] Added STAMP: {ts} | id={uniq} | status={status} | file={label}')

def fitting(dir: str, 
            dc: str, 
            ws: str,  
            tp: str, 
            dc_comb: str, 
            env: os._Environ
            ) -> None:

    text_status, fit_status = 'not-run', 'not-run'
    try:
        dc_combined = f'{dc}/datacard{tar}{comb}.txt'
        if arg.combine:
            dc_mu = f'{dc_comb}/mumu/{tp}/datacard/datacard{tar}.txt'
            dc_ee = f'{dc_comb}/ee/{tp}/datacard/datacard{tar}.txt'

            print('\n----->[Info] Combining datacards')
            with open(dc_combined, 'w') as out:
                subprocess.run(['combineCards.py', dc_mu, dc_ee], 
                            stdout=out, env=env, check=True)

        with open(log_text, 'w') as log_out:
            enter = '' if arg.combine else '\n'
            print(f'{enter}----->[Info] Setting files for the fit')
            subprocess.run(['text2workspace.py', dc_combined, '-v', '10',
                            '--X-allow-no-background', '-m', '125', '-o', ws_file],
                            stdout=log_out, stderr=subprocess.STDOUT, 
                            cwd=dir, env=env, check=True)
        text_status = 'ok'
        
        with open(result_log, 'w') as log_out:
            print('----->[Info] Doing the fit')
            subprocess.run(['combine', ws_file, '-M', 'MultiDimFit', '-m', '125',
                            '-v', '10', '-t', '0', '--expectSignal=1', '-n', 'Xsec'],
                            stdout=log_out, stderr=subprocess.STDOUT, 
                            cwd=ws, env=env, check=True)
        fit_status = 'ok'

    except subprocess.CalledProcessError as exc:
        if text_status=='not-run':
            text_status = f'error exit={exc.returncode}'
        elif fit_status=='not-run':
            fit_status = f'error exit={exc.returncode}'
        print(f'----->[Error] Fit command failed with exit code {exc.returncode}')
        raise
    finally:
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
    print('----->[Info] Fit done, extracting results')
    mu, err = -100, -100      
    with open(res_log) as file:
        lines = file.readlines()
        for line in reversed(lines):
            parts = line.replace('\t', ' ').split()

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

    if (mu==-100) and (err==-100):
        print("----->[Error] Couldn't extract value of the fit, go to the log to have more informations")
        exit(0)
    else:
        if not arg.noprint:
            print('----->[Info] Results successfully extracted')
            print(f'\tmu = {mu} +/- {err}')
            print(f'----->[Info] Uncertainty obtained on ZH cross-section: {err*100:.2f} %')

        with open(f'{res}/results{tar}.txt', 'w') as f:
            f.write(f'{mu}\n{err}\n')

        print(f'----->[Info] Saved results in {res}/results{tar}.txt')



######################
### CODE EXECUTION ###
######################

fitting(dir, dc, ws, tp, dc_comb, env)
mu, err = res_extraction(result_log)
res_saving(mu, err, res)

if __name__=='__main__' and arg.t:
    timer(t)
    
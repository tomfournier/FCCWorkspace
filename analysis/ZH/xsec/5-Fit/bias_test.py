import time, argparse, importlib, subprocess

import numpy as np
import pandas as pd

t = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc

from package.config import timer, warning, z_decays, H_decays
from package.plots.plotting import Bias, PseudoRatio
from package.tools.utils import mkdir



########################
### ARGUMENT PARSING ###
########################

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365],     type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', 
                    choices=[10.8, 3.1], type=float, default=10.8)
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

parser.add_argument('--pert',    help='Prior uncertainty on ZH cross-section used for the bias test', 
                    type=float, default=1.05)
parser.add_argument('--extra',   help='Extra argument for the fit', 
                    choices=['freeze', 'float', 'plot_dc', 'polL', 'polR', 'ILC', 'tot', 'onlyrun', 't'], 
                    type=str,   default='')
parser.add_argument('--combine', help='Combine the channel to do the fit', action='store_true')
arg = parser.parse_args()

if arg.cat=='' and not arg.combine:
    msg = 'Final state or combine were not selected, please select one to run this code'
    warning(msg)



###############################
### CONFIGURATION AND SETUP ###
###############################

cat, ecm, lumi, sel, pert = arg.cat, arg.ecm, arg.lumi, arg.sel, arg.pert
cat = 'combined' if arg.combine else cat

cmd_args = []
if arg.cat:
    cmd_args.extend(['--cat', f'{arg.cat}'])
if arg.combine:
    cmd_args.append('--combine')
cmd_args.extend(['--sel', f'{sel}'])

if arg.extra:
    cmd_args.extend(f'--{ext}' for ext in arg.extra.split('-'))


inputdir   = get_loc(loc.BIAS_FIT_RESULT,   cat, ecm, sel)
loc_result = get_loc(loc.BIAS_RESULT,       cat, ecm, sel)
nomDir     = get_loc(loc.NOMINAL_RESULT,    cat, ecm, sel)
inDir      = get_loc(loc.HIST_PREPROCESSED, cat, ecm, sel)



#################
### FUNCTIONS ###
#################

def run_fit(target: str, 
            pert: float, 
            cmd_args: list[str]
            ) -> float:
    
    cmd = ['python3', '5-Fit/make_pseudo.py', '--target', target,
           '--pert', str(pert), '--run'] + cmd_args
    subprocess.run(cmd, check=True, capture_output=False)

    mu = np.loadtxt(f'{inputdir}/results_{target}.txt')[0]
    return mu


def get_bias(inDir: str, 
             outDir: str, 
             z_decays: list[str], 
             h_decays: list[str],
             cat: str,
             sel: str,
             pert: float,
             cmd_args: list[str],
             ecm: int = 240,
             lumi: float = 10.8
             ) -> tuple[pd.DataFrame, 
                        list[float]]:

    bias = [0.0] * len(h_decays)
    for idx, h_decay in enumerate(h_decays):
        print(f'----->[Info] Running fit for {h_decay} channel')

        mu = run_fit(h_decay, pert, cmd_args)
        bias[idx] = 100 * (mu - pert)

        print(f'\n----->[Info] Bias obtained: {bias[idx]:.3f}\n')

        if not arg.combine:
            print('----->[Info] Making plots for pseudo-signal')
            args = {
                'variable': 'zll_recoil_m',
                'inDir': inDir, 'outDir': outDir,
                'cat': cat, 'target': h_decay,
                'z_decays': z_decays, 'h_decays': h_decays,
                'ecm':ecm, 'lumi': lumi, 'pert': pert
            }
            for sel_suffix in ['_high', '_low']:
                sel_full = sel + sel_suffix
                PseudoRatio(**args, sel=sel_full)

    print(f'----->[Info] Saving bias in a .csv file')
    df = pd.DataFrame({'mode':h_decays, 'bias':bias})

    result_csv = f'{loc_result}/bias_results.csv'
    df.to_csv(result_csv, index=False)
    print(f'----->[Info] Bias saved at {result_csv}')

    return df, bias


def bias_to_txt(outDir: str, 
                bias: list[float], 
                h_decays: list[str]
                ) -> None:
    print(f'----->[Info] Saving bias in a .txt file')

    out = f'{outDir}/bias_results.txt'
    ndecays, col_w = len(h_decays), 15

    header = f"{'Decay modes':<{col_w}}" + ''.join(f'{decay:<{col_w}}' for decay in h_decays)
    sep = '-' * col_w * (ndecays + 1)

    bias_3dec = [f'{b:.3f}' for b in bias]
    bias_2dec = [f'{b:.2f}' for b in bias]

    row_3dec = f"{'Bias':<{col_w}}" + ''.join(f'{val:<{col_w}}' for val in bias_3dec)
    row_2dec = f"{'Bias':<{col_w}}" + ''.join(f'{val:<{col_w}}' for val in bias_2dec)

    with open(out, 'w') as f:
        for row in [header, sep, row_3dec, row_2dec]:
            f.write(row + '\n')
    
    print(f'----->[Info] Bias saved at {out}')



######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    mkdir(loc_result)

    df, bias = get_bias(inDir, loc_result, 
                        z_decays, H_decays, 
                        cat, sel, pert, 
                        cmd_args,
                        ecm=ecm, lumi=lumi)
    
    print(f'----->[Info] Making plot of the bias')
    Bias(df, nomDir, loc_result, H_decays)

    bias_to_txt(loc_result, bias, H_decays)
    
    timer(t)

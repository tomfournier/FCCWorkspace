import os, sys, time, argparse, importlib
import numpy as np
import pandas as pd

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365],     type=int, default=240)
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

parser.add_argument('--pert',    help='Prior uncertainty on ZH cross-section used for the bias test', 
                    type=float, default=1.05)
parser.add_argument('--extra',   help='Extra argument for the fit', 
                    choices=['freeze', 'float', 'plot_dc', 'polL', 'polR', 'ILC', 'tot'], 
                    type=str, default='')
parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')
arg = parser.parse_args()

if arg.cat=='' and not arg.combine:
    print('\n----------------------------------------------------------------------------\n')
    print('Final state or combine were not selected, please select one to run this code')
    print('\n----------------------------------------------------------------------------\n')
    exit(0)

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, h_decays

cat, ecm, sel, pert = arg.cat, arg.ecm, arg.sel, arg.pert
arg_cat, comb, arg_sel = f'--cat {arg.cat}' if arg.cat!='' else '', '--combine' if arg.combine else '', f'--sel {sel}'
if arg.combine: cat = 'combined'
if arg.extra!='':
    extra = arg.extra.split('-')
    extraArgs = ' '.join('--'+ext for ext in extra)
else: extraArgs=''

inputdir   = get_loc(loc.BIAS_FIT_RESULT, cat, ecm, sel)
loc_result = get_loc(loc.BIAS_RESULT,     cat, ecm, sel)
nomDir     = get_loc(loc.NOMINAL_RESULT,  cat, ecm, sel)

def run_fit(target, pert, extraArgs=""):
    cmd = f"python3 5-Fit/make_pseudo.py {arg_cat} --target {target} --pert {pert} --run {comb} {arg_sel} {extraArgs}"
    os.system(cmd)
    mu, err = np.loadtxt(f'{inputdir}/results_{target}.txt')
    return mu, err

res, bias = [], []
for i, h_decay in enumerate(h_decays):
    print(f'----->[Info] Running fit for {h_decay} channel')
    mu, err = run_fit(h_decay, pert, extraArgs)
    b = 100*(mu - pert)
    res.append(mu)
    bias.append(b)
    print(f"\n----->[Info] Bias obtained: {b:.3f}\n")

print(f'----->[Info] Saving bias in a .csv file')
dict = {'mode':h_decays, 'bias':bias}
df = pd.DataFrame(dict)

if not os.path.isdir(loc_result):
    os.system(f'mkdir -p {loc_result}')

df.to_csv(f'{loc_result}/bias_results.csv')
print(f'----->[Info] Bias saved at {loc_result}/bias_results.csv')

print(f'----->[Info] Saving bias in a .txt file')
out_orig = sys.stdout
with open(f"{loc_result}/bias_results.txt", 'w') as f:
    sys.stdout = f
    formatted_row = '{:<15}' + ' '.join(['{:<15}']*len(h_decays))
    print(formatted_row.format(*(["Decay modes"]+h_decays)))
    print(formatted_row.format(*(["----------"]*(len(h_decays)+1))))

    row1 = ["Bias"]
    for i in bias:
        row1.append("%.3f" % i)
    print(formatted_row.format(*row1))
    row2 = ["Bias"]
    for i in bias:
        row2.append("%.2f" % i)
    print(formatted_row.format(*row2))
sys.stdout = out_orig
print(f'----->[Info] Bias saved at {loc_result}/bias_results.txt\n')

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
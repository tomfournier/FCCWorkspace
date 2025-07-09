import os
import sys
import time
import importlib
import argparse
import numpy as np
import pandas as pd

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', choices=[10.8, 3.1], type=float, default=10.8)

parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument('--leading', help='Add the p_leading and p_subleading cuts', action='store_true')
parser.add_argument('--vis', help='Add E_vis > 10 GeV cut', action='store_true')
parser.add_argument('--visbdt', help='Add E_vis in the training variables for the BDT', action='store_true')
parser.add_argument('--sep', help='Separate events by using E_vis', action='store_true')

parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')

arg = parser.parse_args()

if arg.cat=='' and not arg.combine:
    print('\n----------------------------------------------------------------------------\n')
    print('Final state or combine were not selected, please select one to run this code')
    print('\n----------------------------------------------------------------------------\n')
    exit(0)

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, h_decays

cat, comb        = f'--cat {arg.cat}' if arg.cat!='' else '', '--combine' if arg.combine else ''
mis, bdt, recoil = '--miss' if arg.miss else '', '--bdt' if arg.bdt else '', '--recoil120' if arg.recoil120 else ''
lead, vis, sep = '--leading' if arg.leading else '', '--vis' if arg.vis else '', '--sep' if arg.sep else ''
if arg.combine: arg.cat = 'combined'

sel = select(arg.recoil120, arg.miss, arg.bdt, arg.leading, arg.vis, arg.visbdt, arg.sep)
inputdir   = get_loc(loc.BIAS_FIT_RESULT, arg.cat, arg.ecm, sel)
loc_result = get_loc(loc.BIAS_RESULT, arg.cat, arg.ecm, sel)

def run_fit(target, pert, extraArgs=""):
    cmd = f"python3 4-Fit/make_pseudo.py {cat} --target {target} --pert {pert} --run {comb} {recoil} {lead} {mis} {bdt} {vis} {sep} {extraArgs}"
    os.system(cmd)
    mu, err = np.loadtxt(f'{inputdir}/results_{target}.txt')
    return mu, err

pert = 1.05

res, bias = [], []
for i, h_decay in enumerate(h_decays):
    print(f'----->[Info] Running fit for {h_decay} channel')
    mu, err = run_fit(h_decay, pert)
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
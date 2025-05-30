import os
import sys
import time
import importlib
import numpy as np
import pandas as pd

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, h_decays

inputdir = loc.BIAS_FIT_RESULT
loc_result = loc.BIAS_RESULT

def run_fit(target, pert, extraArgs=""):
    cmd = f"python3 4-Fit/make_pseudo.py --target {target} --pert {pert} --run {extraArgs}"
    os.system(cmd)
    mu, err = np.loadtxt(f'{inputdir}/results_{target}.txt')
    return mu, err

pert = 1.05

res, bias = [], []
for i, h_decay in enumerate(h_decays):
    print(f'----->[Info] Running fit for {h_decay} channel')
    mu, err = run_fit(h_decay, pert)
    b = 100*(mu - pert)
    bias.append(b)
    print(f"----->[Info] Bias obtained:\n\t{b:.3f}")

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
    row = ["Bias"]
    for i in bias:
        row.append("%.3f" % i)
    print(formatted_row.format(*row))
sys.stdout = out_orig
print(f'----->[Info] Bias saved at {loc_result}/bias_results.txt')

t2 = time.time()

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {t2-t1:.1f} s')
print('\n------------------------------------\n\n')
import os
import time
import importlib
import numpy as np
import pandas as pd

t1 = time.time()

userConfig = importlib.import_module('userConfig')
inputdir = userConfig.loc.BIAS_HIST

if not userConfig.combine:
    loc_result = f'{userConfig.loc.BIAS}/{userConfig.final_state}'
else:
    loc_result = f'{userConfig.loc.BIAS}/combined'

final_state = userConfig.final_state

def run_fit(target, pert, extraArgs=""):
    cmd = f"python3 5-Model-Independence/make_pseudo.py --target {target} --pert {pert} --run {extraArgs}"
    os.system(cmd)
    mu, err = np.loadtxt(f'{inputdir}/results/results_{target}.txt')
    return mu, err

if __name__ == "__main__":

    pert = 1.05
    h_decays = ["bb", "cc"] # , "gg", "ss", "mumu", "tautau", "ZZ", "WW", "Za", "aa"] #, "inv"]

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

t2 = time.time()

print('\n\n---------------------------------\n')
print(f'Time to run the code: {t2-t1:.1f} s')
print('\n---------------------------------\n\n')
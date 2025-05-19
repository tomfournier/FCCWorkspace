import os
import ROOT
import importlib
import pandas as pd

userConfig = importlib.import_module('userConfig')
inputdir = userConfig.COMBINE

final_state = userConfig.final_state

def run_fit(cats, target, pert, extraArgs):
    # modify fit.py to have argparse
    df = pd.read_csv(f'{inputdir}/log/results.csv')
    mu, err = df['mu'], df['err']
    return [mu, err]

if __name__ == "__main__":

    pert = 1.01
    h_decays = ["bb", "cc", "gg", "ss", "mumu", "tautau", "ZZ", "WW", "Za", "aa", "inv"]

    res, bias = [], []
    for i, h_decay in enumerate(h_decays):
        print(f'----->[Info] Running fit for {h_decay} channel')
        B = run_fit(final_state, h_decay, pert)[0]
        b = 100*(B - pert)
        bias.append(b)
        print(f"----->[Info] Bias obtained:\n\t{bias:.3f}")

    print(f'----->[Info] Saving bias in a .csv file')
    df = pd.Series(bias, index=h_decays)
    df.to_csv(f'{userConfig.loc.BIAS}/bias_results.csv')
    print(f'----->[Info] Bias saved at {userConfig.loc.BIAS}/bias_results.csv')
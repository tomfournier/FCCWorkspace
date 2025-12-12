import time, argparse, importlib

import pandas as pd

t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()


userconfig = importlib.import_module('userConfig')
from userConfig import loc, add_package_path, get_loc
add_package_path(loc.PACKAGE)

from package.config import timer, warning, vars

if arg.cat=='':
    warning(log_msg='Final state was not selected, please select one to run this script')

from package.func.bdt import print_stats, split_data
from package.func.bdt import train_model, save_model

cat, ecm = arg.cat, arg.ecm
sels = [
    'Baseline'
]

modes = [
    f'Z{cat}H', 
    'ZZ', f'WW{cat}', f'Z{cat}', 
    f'egamma_{cat}', f'gammae_{cat}', 
    f'gaga_{cat}'
]

config = {
    'n_estimators': 350, 
    'learning_rate': 0.20,
    'max_depth': 3, 
    'subsample': 0.5,
    'gamma': 3, 
    'min_child_weight': 10,
    'max_delta_step': 0, 
    'colsample_bytree': 0.5,
}



def run(sels: list[str], 
        modes: list[str], 
        vars: list[str], 
        config: dict[str, str], 
        early: int = 25
        ) -> None:

    print('TRAINING VARS')
    print(', '.join(var for var in vars))

    for sel in sels:
        inDir  = get_loc(loc.MVA_INPUTS, cat, ecm, sel)
        outDir = get_loc(loc.BDT,        cat, ecm, sel)

        df = pd.read_pickle(f'{inDir}/preprocessed.pkl')
        print_stats(df, modes)

        X_train, y_train, X_valid, y_valid = split_data(df, vars)
        bdt = train_model(X_train, y_train, 
                          X_valid, y_valid, 
                          config, early)
        save_model(bdt, vars, outDir)

        print('----->[Info] Writing variable inputs in a .txt file for evaluation')
        fmap = pd.DataFrame({'vars':vars, 'Q':list('q' * len(vars))})
        fmap.to_csv(f'{outDir}/feature.txt', sep='\t', header=False)
        print(f'----->[Info] Wrote variable input in {outDir}/feature.txt')



if __name__=='__main__':
    run(sels, modes, vars, config)
    timer(t)

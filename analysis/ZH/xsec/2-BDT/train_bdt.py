import pandas as pd
import importlib, time, argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)

from tools.utils import print_stats, split_data
from tools.utils import train_model, save_model

userconfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, selections, train_vars

cat, ecm = arg.cat, arg.ecm

modes = [f"{cat}H", "ZZ", f"WW{cat}", 
         f"Z{cat}", f"egamma_{cat}", f"gammae_{cat}", f"gaga_{cat}"]
vars_list = train_vars.copy()

config_dict = {
    'n_estimators': 350, 'learning_rate': 0.20,
    'max_depth': 3, 'subsample': 0.5,
    'gamma': 3, 'min_child_weight': 10,
    'max_delta_step': 0, 'colsample_bytree': 0.5,
}
early_stopping_round = 25

for sel in selections:

    inputDir = get_loc(loc.MVA_INPUTS, cat, ecm, sel)
    outDir   = get_loc(loc.BDT,        cat, ecm, sel)

    print("TRAINING VARS")
    print(vars_list)

    df = pd.read_pickle(f"{inputDir}/preprocessed.pkl")
    print_stats(df, modes)

    X_train, y_train, X_valid, y_valid = split_data(df, vars_list)

    bdt = train_model(X_train, y_train, X_valid, y_valid, 
                    config_dict, early_stopping_round)

    save_model(bdt, vars_list, outDir)

    print('----->[Info] Writing variable inputs in a .txt file for evaluation')
    q = list('q' * len(vars_list))
    fmap = {'vars':vars_list, 'Q':q}
    fmap = pd.DataFrame(fmap)
    fmap.to_csv(f'{outDir}/feature.txt', sep='\t', header=False)
    print(f'----->[Info] Wrote variable input in {outDir}/feature.txt')


print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
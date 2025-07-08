import pandas as pd
import importlib, time, argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)

parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) in the training variables of the BDT', action='store_true')
parser.add_argument('--leading', help='Add the p_leading and p_subleading cuts', action='store_true')
parser.add_argument('--vis', help='Add E_vis > 10 GeV cut', action='store_true')
parser.add_argument('--sep', help='Separate events by using E_vis', action='store_true')
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)

from tools.utils import print_stats, split_data
from tools.utils import train_model, save_model

userconfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, train_vars

final_state, ecm = arg.cat, arg.ecm
sel = select(arg.recoil120, arg.miss, arg.bdt, arg.leading, arg.vis, arg.sep)

modes = [f"{final_state}H", "ZZ", f"WW{final_state}", 
         f"Z{final_state}", f"egamma_{final_state}", f"gammae_{final_state}", f"gaga_{final_state}"]
vars_list = train_vars.copy()
if arg.bdt: vars_list.append("cosTheta_miss")

inputDir = get_loc(loc.MVA_PROCESSED, final_state, ecm, sel)
outDir   = get_loc(loc.BDT,           final_state, ecm, sel)

print("TRAINING VARS")
print(vars_list)

df = pd.read_pickle(f"{inputDir}/preprocessed.pkl")

print_stats(df, modes)

X_train, y_train, X_valid, y_valid = split_data(df, vars_list)

depth = 4 if arg.bdt else 3
config_dict = {
        'n_estimators': 350, 'learning_rate': 0.20,
        'max_depth': depth, 'subsample': 0.5,
        'gamma': 3, 'min_child_weight': 10,
        'max_delta_step': 0, 'colsample_bytree': 0.5,
}
early_stopping_round = 25

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
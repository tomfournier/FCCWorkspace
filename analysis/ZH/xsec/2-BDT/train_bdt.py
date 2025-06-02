import pandas as pd
import importlib, time

userconfig = importlib.import_module('userConfig')
from userConfig import loc, train_vars, final_state, miss_BDT

from tools.utils import print_stats, split_data
from tools.utils import train_model, save_model

modes = [f"{final_state}H", "ZZ", f"WW{final_state}", 
         "Zll", "egamma", "gammae", f"gaga_{final_state}"]
vars_list = train_vars
inputDir = loc.MVA_PROCESSED
outDir = loc.BDT

t1 = time.time()

print("TRAINING VARS")
print(vars_list)

df = pd.read_pickle(f"{inputDir}/preprocessed.pkl")

print_stats(df, modes)

X_train, y_train, X_valid, y_valid = split_data(df, vars_list)

config_dict = {
        'n_estimators': 350, 'learning_rate': 0.20,
        'max_depth': 3, 'subsample': 0.5,
        'gamma': 3, 'min_child_weight': 10,
        'max_delta_step': 0, 'colsample_bytree': 0.5,
}
early_stopping_round = 25

bdt = train_model(X_train, y_train, X_valid, y_valid, 
                  config_dict, early_stopping_round)

save_model(bdt, vars_list, outDir)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
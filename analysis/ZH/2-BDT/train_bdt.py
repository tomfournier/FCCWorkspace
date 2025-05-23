import pandas as pd
import xgboost as xgb
import ROOT
import joblib
from matplotlib import rc

from userConfig import loc, train_vars, final_state
import tools.utils as ut

rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)


def run():
    modes = [f"{final_state}H", "ZZ", f"WW{final_state}", "Zll", "egamma", "gammae", f"gaga_{final_state}"]
    vars_list = train_vars
    print("TRAINING VARS")
    print(vars_list)
    df = pd.read_pickle(f"{loc.PKL}/preprocessed.pkl")

    print_stats(df, modes)

    X_train, y_train, X_valid, y_valid = split_data(df, vars_list)

    config_dict = get_config_dict()
    early_stopping_round = 25

    bdt = train_model(X_train, y_train, X_valid, y_valid, config_dict, early_stopping_round)

    save_model(bdt, vars_list, loc.BDT)


def print_stats(df, modes):
    print("__________________________________________________________")
    print("Input number of events:")
    for cur_mode in modes:
        print(f"Number of training {cur_mode}: {int(len(df[(df['sample'] == cur_mode) & (df['valid'] == False)]))}")
        print(f"Number of validation {cur_mode}: {int(len(df[(df['sample'] == cur_mode) & (df['valid'] == True)]))}")
    print("__________________________________________________________")


def split_data(df, vars_list):
    X_train = df.loc[df['valid'] == False, vars_list].to_numpy()
    y_train = df.loc[df['valid'] == False, ['isSignal']].to_numpy()
    X_valid = df.loc[df['valid'] == True, vars_list].to_numpy()
    y_valid = df.loc[df['valid'] == True, ['isSignal']].to_numpy()
    return X_train, y_train, X_valid, y_valid


def get_config_dict():
    return {
        "n_estimators": 350,
        "learning_rate": 0.20,
        "max_depth": 3,
        'subsample': 0.5,
        'gamma': 3,
        'min_child_weight': 10,
        'max_delta_step': 0,
        'colsample_bytree': 0.5,
    }


def train_model(X_train, y_train, X_valid, y_valid, config_dict, early_stopping_round):
    bdt = xgb.XGBClassifier(**config_dict, eval_metric=["error", "logloss", "auc"], 
                            early_stopping_rounds=early_stopping_round)
    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    print("Training model")
    bdt.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    return bdt


def save_model(bdt, vars_list, output_path):
    ut.create_dir(output_path)
    print("--->Writing xgboost model:")
    print(f"------>Saving BDT in a .root file at {output_path}/xgb_bdt.root")
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, "ZH_Recoil_BDT", f"{output_path}/xgb_bdt.root", num_inputs=len(vars_list))
    print(f"------>Saving BDT in a .jotlib file at {output_path}/xgb_bdt.joblib")
    joblib.dump(bdt, f"{output_path}/xgb_bdt.joblib")

if __name__ == "__main__":
    run()   



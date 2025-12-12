import numpy as np
import pandas as pd
import xgboost as xgb

from typing import overload
from sklearn.model_selection import train_test_split

from ..tools.utils import get_df, mkdir

@overload
def counts_and_effs(files: list[str], 
                    vars: list[str], 
                    only_eff: bool = True
                    ) -> float:
    ...

@overload
def counts_and_effs(files: list[str], 
                    vars: list[str], 
                    only_eff: bool = False
                    ) -> tuple[int, pd.DataFrame, 
                               float]:
    ...

#________________________________________________
def counts_and_effs(files: list[str], 
                    vars: list[str], 
                    only_eff: bool = False
                    ) -> tuple[int, pd.DataFrame, 
                               float] | float:
    import uproot
    N_events = sum(uproot.open(f)['eventsProcessed'].value 
                   for f in files)
    df = pd.concat((get_df(f, vars) for f in files), 
                   ignore_index=True, copy=False)
    eff = df.shape[0] / N_events
    if only_eff: 
        return eff
    else: 
        return df, eff, N_events

#_____________________________________
def additional_info(df: pd.DataFrame, 
                    mode: str, 
                    sig: str
                    ) -> pd.DataFrame:
    df = df.copy(deep=False)
    df['sample'], df['isSignal'] = mode, int(mode == sig)
    return df

#____________________________________________
def BDT_input_numbers(df: pd.DataFrame, 
                      modes: list[str], 
                      sig: str, 
                      eff: dict[str, float], 
                      xsec: dict[str, float], 
                      frac: dict[str, float]
                      ) -> dict[str, int]:
    N_BDT_inputs: dict[str, int] = {}
    xsec_tot_bkg = sum(eff[mode] * xsec[mode] for mode in modes if mode != sig)
    for m in modes:
        N_BDT_inputs[m] = (int(frac[m] * len(df[m])) if m == sig else 
                           int(frac[m] * len(df[sig]) * (eff[m] * xsec[m] \
                                                         / xsec_tot_bkg)))
    return N_BDT_inputs

#______________________________________________
def df_split_data(df: pd.DataFrame, 
                  N_BDT_inputs: dict[str, int], 
                  eff: dict[str, float], 
                  xsec: dict[str, float], 
                  N_events: dict[str, int], 
                  mode: str, 
                  lumi: float = 10.8
                  ) -> pd.DataFrame:
    sampled = df.sample(n=N_BDT_inputs[mode], random_state=1)
    df0, df1 = train_test_split(sampled, test_size=0.5, random_state=7)

    sampled.loc[:, 'norm_weight'] = xsec[mode] / N_events[mode]

    sampled.loc[df0.index, 'valid'] = False
    sampled.loc[df1.index, 'valid'] = True
    
    coeff = eff[mode] * xsec[mode] * lumi * 1e6
    sampled.loc[df0.index, 'weights'] = coeff / df0.shape[0]
    sampled.loc[df1.index, 'weights'] = coeff / df1.shape[0]
    return sampled

#________________________________
def print_stats(df: pd.DataFrame, 
                modes: list
                ) -> None:
    separator = '=' * 55
    print(f'{'NUMBER OF BDT INPUT EVENTS':=^55}')
    train = df['valid']==False
    for m in modes:
        m_mask = df['sample']==m
        print(f'{f'Number of training for {m}':<40}: {int((m_mask &  train).sum())}')
        print(f'{f'Number of validation for {m}':<40}: {int((m_mask & ~train).sum())}')
        print(f'{separator:^55}')

#_____________________________________
def split_data(df: pd.DataFrame, 
               vars: list[str]
               ) -> tuple[np.ndarray, 
                          np.ndarray, 
                          np.ndarray, 
                          np.ndarray]:
    train = df['valid']==False
    X_train = df.loc[train,  vars].to_numpy(dtype=np.float32, copy=False)
    X_valid = df.loc[~train, vars].to_numpy(dtype=np.float32, copy=False)
    y_train = df.loc[train,  'isSignal'].to_numpy(np.int8, copy=False).ravel()
    y_valid = df.loc[~train, 'isSignal'].to_numpy(np.int8, copy=False).ravel()
    return X_train, y_train, X_valid, y_valid

#_________________________________________
def train_model(X_train: np.ndarray, 
                y_train: np.ndarray, 
                X_valid: np.ndarray, 
                y_valid: np.ndarray, 
                config: dict[str, 
                             int | float], 
                early: int
                ) -> xgb.XGBClassifier:
    
    cfg = dict(config)
    cfg.setdefault('use_label_encoder', False)
    cfg.setdefault('n_jobs', -1)
    cfg.setdefault('verbosity', 1)
    cfg.setdefault('tree_method', 'hist')

    bdt = xgb.XGBClassifier(**cfg, eval_metric=['error', 
                                                'logloss', 
                                                'auc'], 
                            early_stopping_rounds=early)
    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    print('Beginning the training')
    bdt.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    return bdt

#_______________________________________
def evaluate_bdt(df: pd.DataFrame, 
                 bdt: xgb.XGBClassifier, 
                 vars: list[str]
                 ) -> pd.DataFrame:
    print(f'--->Evaluating BDT model')
    X = df.loc[:, vars].to_numpy(dtype=np.float32, 
                                 copy=False)
    df['BDTscore'] = bdt.predict_proba(X)[:, 1]
    return df

#_______________________________________________________
def get_metrics(bdt: xgb.XGBClassifier
                ) -> tuple[dict[str, 
                                dict[str, list[float]]], 
                           int, np.ndarray, int]:
    print('------>Retrieving performance metrics')
    results = bdt.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = np.arange(0, epochs, 1)
    best_iteration = (bdt.best_iteration + 1) if \
        (bdt.best_iteration is not None) else epochs
    return results, epochs, x_axis, best_iteration

#__________________________________________________
def load_model(inputDir: str) -> xgb.XGBClassifier:
    import joblib
    print(f'--->Loading BDT model {inputDir}/xgb_bdt.joblib')
    return joblib.load(f'{inputDir}/xgb_bdt.joblib')

#_____________________________________
def save_model(bdt: xgb.XGBClassifier, 
               vars: list[str], 
               path: str) -> None:
    import joblib, ROOT
    mkdir(path)
    print('--->Writing xgboost model:')
    print(f'------>[Info] Saving BDT in a .root file at {path}/xgb_bdt.root')
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, 'ZH_Recoil_BDT', 
                                       f'{path}/xgb_bdt.root', 
                                       num_inputs=len(vars))
    variables = ROOT.TList()
    for var in vars: 
        variables.Add(ROOT.TObjString(var))
    
    with ROOT.TFile(f'{path}/xgb_bdt.root', 'UPDATE') as fOut:
        fOut.WriteObject(variables, 'variables')

    print(f'------>[Info] Saving BDT in a .joblib file at {path}/xgb_bdt.joblib')
    joblib.dump(bdt, f'{path}/xgb_bdt.joblib', compress=2)

#_____________________________________________
def def_bdt(vars: str, 
            loc_bdt: str, 
            MVAVec:  str = 'MVAVec', 
            score:   str = 'BDTscore',
            defineList: dict[str, str] = {},
            suffix: str = ''
            ) -> tuple[dict[str, str], float]:
    import ROOT
    ROOT.gInterpreter.ProcessLine(f'''
        TMVA::Experimental::RBDT<> tmva('ZH_Recoil_BDT', '{loc_bdt}/xgb_bdt.root');
    ''')

    var_list = ', (float)'.join(vars)
    if not MVAVec in defineList:
        defineList[MVAVec] = f'ROOT::VecOps::RVec<float>{{{var_list}}}'
    defineList['mva_score'+suffix] = f'tmva.Compute({MVAVec})'
    defineList[score+suffix] = 'mva_score.at(0)'

    bdt_cut  = float(np.loadtxt(f'{loc_bdt}/BDT_cut.txt'))
    return defineList, bdt_cut

#_________________________________________
def make_high_low(cutList: dict[str, str], 
                  bdt_cut: float, 
                  sels: list[str], 
                  score: str = 'BDTscore'
                  ) -> dict[str, str]:
    for sel in sels:
        if sel in cutList:
            cutList[sel+'_high'] += f' && {score} > {bdt_cut}'
            cutList[sel+'_low']  += f' && {score} < {bdt_cut}'
    return cutList

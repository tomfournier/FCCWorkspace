import os, sys, json
import uproot, glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import xgboost as xgb
import ROOT
import joblib
from matplotlib import rc
from sklearn.model_selection import train_test_split

from math import sqrt, log

rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)

#__________________________________________________________
def get_df(root_file_name, branches):
  file = uproot.open(root_file_name)
  tree = file['events']

  if len(file) == 0:
    return pd.DataFrame()
  df = tree.arrays(library="pd", how="zip", filter_name=branches)
  return df

#__________________________________________________________
def Z0(S, B):
  if B<=0:
    return -100
  return sqrt(2*((S+B)*log(1+S/B)-S))

#__________________________________________________________
def Zmu(S, B):
  if B<=0:
    return -100
  return sqrt(2*(S-B*log(1+S/B)))

#__________________________________________________________
def Z(S, B):
  if B<=0:
    return -100
  return S/sqrt(S+B)

#__________________________________________________________
def Significance(df_s, df_b, score_column='BDTscore', func=Z0, score_range=(0, 1), nbins=50):
  S0 = np.sum(df_s.loc[df_s.index,'norm_weight'])
  B0 = np.sum(df_b.loc[df_b.index,'norm_weight']) 
  print('initial: S0 = {:.2f}, B0 = {:.2f}'.format(S0, B0))
  print('inclusive Z: {:.2f}'.format(func(S0, B0)))

  wid = (score_range[1]-score_range[0])/nbins
  arr_x = np.round(np.array([score_range[0]+i*wid for i in range(nbins)]), decimals=2)
  arr_Z=np.zeros([nbins])

  for i in tqdm(range(nbins)):
    xi = score_range[0]+i*wid
    Si = np.sum(df_s.loc[df_s.query(f'{score_column} >= {str(xi)}').index,'norm_weight'])
    Bi = np.sum(df_b.loc[df_b.query(f'{score_column} >= {str(xi)}').index,'norm_weight'])
    Zi = func(Si, Bi)
    if Bi<0: continue
    if Zi<0: continue
    arr_Z[i]=Zi
          
  df_Z = pd.DataFrame(data=arr_Z, index=arr_x, columns=["Z"])
  
  return df_Z

#__________________________________________________________
def thres_opt(df, score_column = 'BDTscore', func=Z0, n_spliter=2, score_range=(0, 1), nbins=50, precut='test==True',b_scale=1.):
  df_s = df.query(precut+' & isSignal==1')
  df_b = df.query(precut+' & isSignal==0')
  S0 = len(df_s.index)
  B0 = b_scale*len(df_b.index)
  print('initial: S0={:.2f}, B0={:.2f}'.format(S0, B0))
  print('inclusive Z: {:.2f}'.format(func(S0, B0)))

  wid = (score_range[1]-score_range[0])/nbins
  arr_x = np.round(np.array([score_range[0]+i*wid for i in range(nbins)]), decimals=2)
  arr_Ztot=np.zeros([nbins, nbins])

  for i in tqdm(range(nbins)):
    xi = score_range[0]+i*wid
    Si = len(df_s.query(f'{score_column} >= {str(xi)}').index)
    Bi = b_scale*len(df_b.query(f'{score_column} >= {str(xi)}').index)
    Zi = func(Si, Bi)
    if Bi<=11: continue
    if Zi<0: continue

    for j in range(i):
      xj = score_range[0]+j*wid
      Sj = len(df_s.query(f'{score_column} >= {str(xj)} & {score_column} < {str(xi)}').index)
      Bj = b_scale*len(df_b.query(f'{score_column} >= {str(xj)} & {score_column} < {str(xi)}').index)
      Zj = func(Sj, Bj)
      if Bj<=11: continue
      if Zj<0: continue
      Ztot = sqrt(Zi**2+Zj**2)
      arr_Ztot[i][j] = Ztot

  df_Z = pd.DataFrame(data=arr_Ztot, index=arr_x, columns=arr_x)

  return df_Z

#__________________________________________________________
def dir_exist(mydir):
    import os.path
    if os.path.exists(mydir): return True
    else: return False

#__________________________________________________________
def create_dir(mydir):
    if not dir_exist(mydir):
        import os
        os.system('mkdir -p {}'.format(mydir))

#__________________________________________________________
def get_procDict(procFile):
    procDict = None
    if 'http://' in procFile or 'https://' in procFile:
        print ('----> getting process dictionary from the web')
        import urllib.request
        req = urllib.request.urlopen(procFile).read()
        procDict = json.loads(req.decode('utf-8'))
    else:
        if not ('eos' in procFile):
            procFile = os.path.join(os.getenv('FCCDICTSDIR').split(':')[0], '') + procFile 
            print(f"procFile is {procFile}")
        if not os.path.isfile(procFile):
            print ('----> No procDict found: ==={}===, exit'.format(procFile))
            sys.exit(3)
        with open(procFile, 'r') as f:
            procDict=json.load(f)
    return procDict

#__________________________________________________________
def update_keys(procDict, mode_names):
    # Reverse the mode_names dictionary
    reversed_mode_names = {v: k for k, v in mode_names.items()}
    updated_dict = {}
    for key, value in procDict.items():
        new_key = reversed_mode_names.get(key, key)
        updated_dict[new_key] = value
    return updated_dict

#__________________________________________________________
def get_data_paths(cur_mode, data_path, mode_names):
    path = f"{data_path}/{mode_names[cur_mode]}"
    return glob.glob(f"{path}.root")

#__________________________________________________________
def counts_and_efficiencies(cur_mode, files, vars_list):
    N_events = sum([uproot.open(f)["eventsProcessed"].value for f in files])
    df = pd.concat((get_df(f, vars_list) for f in files), ignore_index=True)
    eff = len(df) / N_events
    return N_events, df, eff

#__________________________________________________________
def additional_info(df, cur_mode, sig):
    df['sample'] = cur_mode
    df['isSignal'] = int(cur_mode == sig)
    return df

#__________________________________________________________
def BDT_input_numbers(mode_names, sig, df, eff, xsec, frac):
    N_BDT_inputs = {}
    # print(f"eff = {eff*100:.3f}%")
    # print(f"xsec = {xsec}")
    xsec_tot_bkg = sum(eff[mode] * xsec[mode] for mode in mode_names if mode != sig)
    for cur_mode in mode_names:
        print(f"Calculating number of BDT inputs for {cur_mode}")
        N_BDT_inputs[cur_mode] = (int(frac[cur_mode] * len(df[cur_mode])) if cur_mode == sig else
                                  int(frac[cur_mode] * len(df[sig]) * (eff[cur_mode] * xsec[cur_mode] / xsec_tot_bkg)))
    return N_BDT_inputs

#__________________________________________________________
def df_split_data(df, N_BDT_inputs, xsec, N_events, cur_mode):
    df = df.sample(n=N_BDT_inputs[cur_mode], random_state=1)
    df0, df1 = train_test_split(df, test_size=0.5, random_state=7)
    df.loc[df0.index, "valid"] = False
    df.loc[df1.index, "valid"] = True
    df.loc[df.index, "norm_weight"] = xsec[cur_mode] / N_events[cur_mode]
    return df

#__________________________________________________________
def save_to_pickle(dfsum, pkl_path):
    print("Writing output to pickle file")
    create_dir(pkl_path)
    print(f"--->Preprocessed saved {pkl_path}/preprocessed.pkl")
    dfsum.to_pickle(f"{pkl_path}/preprocessed.pkl")

#__________________________________________________________
def print_stats(df, modes):
    print("__________________________________________________________")
    print("Input number of events:")
    for cur_mode in modes:
        print(f"Number of training {cur_mode}: {int(len(df[(df['sample'] == cur_mode) & (df['valid'] == False)]))}")
        print(f"Number of validation {cur_mode}: {int(len(df[(df['sample'] == cur_mode) & (df['valid'] == True)]))}")
    print("__________________________________________________________")

#__________________________________________________________
def split_data(df, vars_list):
    X_train = df.loc[df['valid'] == False, vars_list].to_numpy()
    y_train = df.loc[df['valid'] == False, ['isSignal']].to_numpy()
    X_valid = df.loc[df['valid'] == True, vars_list].to_numpy()
    y_valid = df.loc[df['valid'] == True, ['isSignal']].to_numpy()
    return X_train, y_train, X_valid, y_valid

#__________________________________________________________
def train_model(X_train, y_train, X_valid, y_valid, config_dict, early_stopping_round):
    bdt = xgb.XGBClassifier(**config_dict, eval_metric=["error", "logloss", "auc"], 
                            early_stopping_rounds=early_stopping_round)
    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    print("Training model")
    bdt.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    return bdt

#__________________________________________________________
def save_model(bdt, vars_list, output_path):
    create_dir(output_path)
    print("--->Writing xgboost model:")
    print(f"------>Saving BDT in a .root file at {output_path}/xgb_bdt.root")
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, "ZH_Recoil_BDT", f"{output_path}/xgb_bdt.root", num_inputs=len(vars_list))

    variables = ROOT.TList()
    for var in vars_list:
        variables.Add(ROOT.TObjString(var))
    fOut = ROOT.TFile(f"{output_path}/xgb_bdt.root", "UPDATE")
    fOut.WriteObject(variables, "variables")

    print(f"------>Saving BDT in a .jotlib file at {output_path}/xgb_bdt.joblib")
    joblib.dump(bdt, f"{output_path}/xgb_bdt.joblib")

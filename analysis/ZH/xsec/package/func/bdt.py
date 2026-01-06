'''BDT training and evaluation utilities for multivariate analysis.

Provides:
- Data preparation and loading: `counts_and_effs()`, `additional_info()`, `split_data()`.
- Sample balancing: `BDT_input_numbers()`, `df_split_data()`.
- Model training and evaluation: `train_model()`, `evaluate_bdt()`, `get_metrics()`.
- Model persistence: `save_model()`, `load_model()`.
- ROOT integration: `def_bdt()`, `make_high_low()`.
- Utilities for hyperparameter validation: `print_stats()`.
- Support for XGBoost classifiers with early stopping and multi-metric evaluation.
- Seamless integration with ROOT TMVA framework for physics analysis workflows.

Functions:
- `counts_and_effs()`: Load events from files and compute selection efficiency.
- `additional_info()`: Add sample mode and signal classification columns to dataframe.
- `BDT_input_numbers()`: Compute balanced event counts per process based on cross-sections.
- `df_split_data()`: Split samples into training/validation with luminosity weights.
- `print_stats()`: Display training/validation event counts per process.
- `split_data()`: Extract features and labels into train/validation numpy arrays.
- `train_model()`: Train XGBoost classifier with early stopping on validation set.
- `evaluate_bdt()`: Compute BDT scores for all events in a dataframe.
- `get_metrics()`: Extract training curves and optimal iteration from trained model.
- `save_model()`: Export trained model to ROOT TMVA and joblib formats.
- `load_model()`: Load previously trained XGBoost model from joblib file.
- `def_bdt()`: Define BDT computation in ROOT RDataFrame with TMVA integration.
- `make_high_low()`: Create high/low BDT score cut regions for analysis selections.

Conventions:
- Signal/background classification encoded as binary labels: signal=1, background=0.
- Cross-section and efficiency used to balance signal and background training samples.
- Training/validation split 50/50 after balanced sampling per process.
- Event weights scaled by luminosity (ab⁻¹ → pb conversion), cross-section, and efficiency.
- BDT scores output as probability of signal class (range 0-1).
- XGBoost uses histogram tree construction with early stopping on validation loss.
- Models exported to ROOT TMVA format (xgb_bdt.root) for ROOT analysis and joblib (xgb_bdt.joblib) for Python.
- BDT cut threshold stored separately (BDT_cut.txt) and applied via high/low selection variants.
- High/low selections created by adding `_high` and `_low` suffixes to base selection names.
- Overloaded `counts_and_effs()` supports returning efficiency only or full (df, eff, N_events) tuple.

Usage:
- Prepare data for BDT training from ROOT files with automatic efficiency calculation.
- Balance signal and background based on physics cross-sections for unbiased training.
- Train and evaluate BDT with configurable XGBoost hyperparameters and early stopping.
- Export models to both Python and ROOT ecosystems for downstream analysis.
- Define BDT computations in ROOT RDataFrame for efficient event selection.
- Create signal-like (high BDT) and background-like (low BDT) analysis regions.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

# To remove numpy warning
import warnings
warnings.filterwarnings('ignore', message='The value of the smallest subnormal for')

import numpy as np
import pandas as pd
import xgboost as xgb

from typing import overload, Sequence
from sklearn.model_selection import train_test_split

from ..tools.utils import get_df, mkdir



######################
### MAIN FUNCTIONS ###
######################

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
                    ) -> tuple[pd.DataFrame, 
                               float,
                               int]:
    ...

#________________________________________________
def counts_and_effs(files: list[str], 
                    vars: list[str], 
                    only_eff: bool = False
                    ) -> tuple[pd.DataFrame, 
                               float,
                               int] | float:
    '''Calculate event counts, dataframe, and selection efficiency.
    
    Args:
        files (list[str]): List of ROOT file paths.
        vars (list[str]): Variables to extract from files.
        only_eff (bool, optional): If True, return only efficiency; if False, return (df, eff, N_events). Defaults to False.
    
    Returns:
        float: Efficiency value when only_eff=True.
        tuple: (dataframe, efficiency, N_events) when only_eff=False.
    '''
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
    '''Add sample mode and signal label columns to dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): Sample mode/process name.
        sig (str): Signal process name for classification.
    
    Returns:
        pd.DataFrame: Dataframe with added 'sample' and 'isSignal' columns.
    '''
    df = df.copy(deep=False)
    df['sample'], df['isSignal'] = mode, int(mode == sig)  # Add sample identifier and signal flag
    return df

#____________________________________________
def BDT_input_numbers(df: pd.DataFrame, 
                      modes: list[str], 
                      sig: str, 
                      eff: dict[str, float], 
                      xsec: dict[str, float], 
                      frac: dict[str, float]
                      ) -> dict[str, int]:
    '''Calculate number of events to use for BDT training per process.
    
    Balances signal and background using cross-sections and efficiencies.
    
    Args:
        df (pd.DataFrame): Dataframe grouped by mode containing events.
        modes (list[str]): List of process modes.
        sig (str): Signal process name.
        eff (dict[str, float]): Selection efficiency per mode.
        xsec (dict[str, float]): Cross-section per mode (in pb).
        frac (dict[str, float]): Fraction of available events to use per mode.
    
    Returns:
        dict[str, int]: Dictionary with number of BDT input events per mode.
    '''
    N_BDT_inputs: dict[str, int] = {}
    # Total background cross-section weighted by efficiency
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
    '''Sample events are split into training/validation sets with weights.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        N_BDT_inputs (dict[str, int]): Number of events to sample per process.
        eff (dict[str, float]): Selection efficiency per mode.
        xsec (dict[str, float]): Cross-section per mode (in pb).
        N_events (dict[str, int]): Total generated events per mode.
        mode (str): Current process mode.
        lumi (float, optional): Integrated luminosity in ab-1. Defaults to 10.8.
    
    Returns:
        pd.DataFrame: Dataframe with 'valid', 'norm_weight', and 'weights' columns added.
    '''
    sampled = df.sample(n=N_BDT_inputs[mode], random_state=1)
    # Split 50/50 into training and validation sets
    df0, df1 = train_test_split(sampled, test_size=0.5, random_state=7)

    # Normalization weight per event
    sampled.loc[:, 'norm_weight'] = xsec[mode] / N_events[mode]

    # Mark validation set
    sampled.loc[df0.index, 'valid'] = False  # Training set
    sampled.loc[df1.index, 'valid'] = True   # Validation set
    
    # Calculate event weights accounting for efficiency, cross-section, and luminosity
    coeff = eff[mode] * xsec[mode] * lumi * 1e6
    sampled.loc[df0.index, 'weights'] = coeff / df0.shape[0]
    sampled.loc[df1.index, 'weights'] = coeff / df1.shape[0]
    return sampled

#________________________________
def print_stats(df: pd.DataFrame, 
                modes: list
                ) -> None:
    '''Print training and validation event counts per process.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'sample', 'isSignal', and 'valid' columns.
        modes (list): List of process modes to display.
    
    Returns:
        None
    '''

    print(f"{'NUMBER OF BDT INPUT EVENTS':=^55}")
    train = df['valid']==False
    for m in modes:
        m_mask = df['sample']==m
        print(f"{f'Number of training for {m}':<40}: {int((m_mask &  train).sum())}")
        print(f"{f'Number of validation for {m}':<40}: {int((m_mask & ~train).sum())}")
        print(f"{'=' * 55:^55}")

#_____________________________________
def split_data(df: pd.DataFrame, 
               vars: list[str]
               ) -> tuple[np.ndarray, 
                          np.ndarray, 
                          np.ndarray, 
                          np.ndarray]:
    '''Split data into training and validation sets for features and labels.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'valid' column marking validation set.
        vars (list[str]): Feature variable names to extract.
    
    Returns:
        tuple: (X_train, y_train, X_valid, y_valid) as numpy arrays.
    '''
    train = df['valid']==False
    # Features for training and validation sets
    X_train = df.loc[train,  vars].to_numpy(copy=False)
    X_valid = df.loc[~train, vars].to_numpy(copy=False)
    # Labels (signal/background) for training and validation sets
    y_train = df.loc[train,  'isSignal'].to_numpy(np.int8, copy=False).ravel()
    y_valid = df.loc[~train, 'isSignal'].to_numpy(np.int8, copy=False).ravel()
    return X_train, y_train, X_valid, y_valid

#_________________________________________
def train_model(X_train: np.ndarray, 
                y_train: np.ndarray, 
                X_valid: np.ndarray, 
                y_valid: np.ndarray, 
                config: dict[str, 
                             str | int | float], 
                early: int
                ) -> xgb.XGBClassifier:
    '''Train XGBoost model with early stopping.
    
    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_valid (np.ndarray): Validation feature matrix.
        y_valid (np.ndarray): Validation labels.
        config (dict[str, int | float]): XGBoost hyperparameters.
        early (int): Number of rounds for early stopping.
    
    Returns:
        xgb.XGBClassifier: Trained XGBoost classifier.
    '''
    cfg = dict(config)
    # Set default XGBoost parameters
    cfg.setdefault('use_label_encoder', False)
    # cfg.setdefault('n_jobs', -1)
    cfg.setdefault('verbosity', 1)
    cfg.setdefault('tree_method', 'auto')

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
    '''Compute BDT scores and add to dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        bdt (xgb.XGBClassifier): Trained XGBoost classifier.
        vars (list[str]): Feature variables for prediction.
    
    Returns:
        pd.DataFrame: Dataframe with 'BDTscore' column added.
    '''
    print(f'--->Evaluating BDT model')
    # Extract features as float32 numpy array
    X = df.loc[:, vars].to_numpy(dtype=np.float32, 
                                 copy=False)
    # Add probability of signal class as BDT score
    df['BDTscore'] = bdt.predict_proba(X)[:, 1]
    return df

#_______________________________________________________
def get_metrics(bdt: xgb.XGBClassifier
                ) -> tuple[dict[str, 
                                dict[str, list[float]]], 
                           int, np.ndarray, int]:
    '''Extract training metrics from trained model.
    
    Args:
        bdt (xgb.XGBClassifier): Trained XGBoost classifier.
    
    Returns:
        tuple: (results dict, num_epochs, epoch_axis, best_epoch).
    '''
    print('------>Retrieving performance metrics')
    results = bdt.evals_result()
    epochs = len(results['validation_0']['error'])
    # X-axis values for epoch numbering
    x_axis = np.arange(0, epochs, 1)
    # Best iteration from early stopping or total epochs
    best_iteration = (bdt.best_iteration + 1) if \
        (bdt.best_iteration is not None) else epochs
    return results, epochs, x_axis, best_iteration

#_______________________________________________
def load_model(inDir: str) -> xgb.XGBClassifier:
    '''Load previously trained XGBoost model.
    
    Args:
        inDir (str): Directory path containing xgb_bdt.joblib file.
    
    Returns:
        xgb.XGBClassifier: Loaded XGBoost classifier.
    '''
    import joblib
    fpath = f'{inDir}/xgb_bdt.joblib'
    print(f'--->Loading BDT model {fpath}')
    return joblib.load(fpath)

#_____________________________________
def save_model(bdt: xgb.XGBClassifier, 
               vars: list[str], 
               path: str) -> None:
    '''Save trained model in ROOT and joblib formats.
    
    Args:
        bdt (xgb.XGBClassifier): Trained XGBoost classifier.
        vars (list[str]): Feature variable names.
        path (str): Directory path for saving model files.
    
    Returns:
        None
    '''
    import joblib, ROOT
    mkdir(path)
    froot, fjob = f'{path}/xgb_bdt.root', f'{path}/xgb_bdt.joblib'
    print('--->Writing xgboost model:')
    print(f'------>[Info] Saving BDT in a .root file at {froot}')
    # Save model in TMVA ROOT format
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, 'ZH_Recoil_BDT', 
                                       froot, 
                                       num_inputs=len(vars))
    # Create list of feature variable names
    variables = ROOT.TList()
    for var in vars: 
        variables.Add(ROOT.TObjString(var))
    
    # Write variable list to ROOT file
    with ROOT.TFile(froot, 'UPDATE') as fOut:
        fOut.WriteObject(variables, 'variables')

    print(f'------>[Info] Saving BDT in a .joblib file at {fjob}')
    # Save model in joblib format for Python
    joblib.dump(bdt, fjob, compress=2)

#_____________________________________________
def def_bdt(vars: Sequence[str], 
            loc_bdt: str, 
            MVAVec:  str = 'MVAVec', 
            score:   str = 'BDTscore',
            defineList: dict[str, str] = {},
            suffix: str = ''
            ) -> tuple[dict[str, str], float]:
    '''Define BDT computation in ROOT RDataFrame and load cut value.
    
    Args:
        vars (Sequence[str]): Comma-separated feature variable names.
        loc_bdt (str): Directory containing xgb_bdt.root model file.
        MVAVec (str, optional): Name for the feature vector column. Defaults to 'MVAVec'.
        score (str, optional): Name for BDT score column. Defaults to 'BDTscore'.
        defineList (dict[str, str], optional): Dictionary to append column definitions to. Defaults to {}.
        suffix (str, optional): Suffix to append to output column names. Defaults to ''.
    
    Returns:
        tuple: (updated defineList, BDT cut threshold).
    '''
    import ROOT
    # Load TMVA model from ROOT file
    ROOT.gInterpreter.ProcessLine(f'''
        TMVA::Experimental::RBDT<> tmva("ZH_Recoil_BDT", "{loc_bdt}/xgb_bdt.root");
    ''')

    var_list = ', (float)'.join(vars)
    # Define feature vector if not already present
    if not MVAVec in defineList:
        defineList[MVAVec] = f'ROOT::VecOps::RVec<float>{{{var_list}}}'
    # Compute BDT score
    defineList['mva_score'+suffix] = f'tmva.Compute({MVAVec})'
    defineList[score+suffix] = 'mva_score.at(0)'

    # Load BDT cut value from file
    bdt_cut  = float(np.loadtxt(f'{loc_bdt}/BDT_cut.txt'))
    return defineList, bdt_cut

#_________________________________________
def make_high_low(cutList: dict[str, str], 
                  bdt_cut: float, 
                  sels: list[str], 
                  score: str = 'BDTscore'
                  ) -> dict[str, str]:
    '''Create high/low BDT score cut regions for selected criteria.
    
    Args:
        cutList (dict[str, str]): Dictionary of selection criteria strings.
        bdt_cut (float): BDT cut threshold value.
        sels (list[str]): List of selection names to split into high/low regions.
        score (str, optional): BDT score column name. Defaults to 'BDTscore'.
    
    Returns:
        dict[str, str]: Updated cutList with '_high' and '_low' cut variants.
    '''
    for sel in sels:
        if sel in cutList:
            old_cut = cutList[sel]
            # Add high BDT score region (signal-like)
            cutList[sel+'_high'] = old_cut + f' && {score} > {bdt_cut}'
            # Add low BDT score region (background-like)
            cutList[sel+'_low'] = old_cut + f' && {score} < {bdt_cut}'
    return cutList

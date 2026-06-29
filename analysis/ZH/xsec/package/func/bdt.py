'''BDT training and evaluation utilities for multivariate analysis.

Provides:
- Data preparation and loading: `counts_and_effs()`, `additional_info()`, `split_data()`.
- Sample balancing: `BDT_input_numbers()`, `df_split_data()`.
- Sample balancing: `sample_df_by_xsec()`: Sample process dataframes within a mode in proportion to `eff * xsec`.
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

Lazy Imports:
- numpy, pandas, xgboost are lazy-loaded only when their functions are called
- Use TYPE_CHECKING for type hints to avoid import overhead at module load time
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

# To remove numpy warning
import warnings
warnings.filterwarnings('ignore', message='The value of the smallest subnormal for')

from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xgboost as xgb

from ..tools.utils import mkdir
from ..logger import get_logger

LOGGER = get_logger(__name__)



######################
### MAIN FUNCTIONS ###
######################

@overload
def counts_and_effs(
    files: list[str],
    vars: list[str],
    only_eff: bool = True
     ) -> float:
    ...

@overload
def counts_and_effs(
    files: list[str],
    vars: list[str],
    only_eff: bool = False
     ) -> tuple['pd.DataFrame',
                float,
                int]:
    ...

# ___________________________
def counts_and_effs(
    files: list[str],
    vars: list[str],
    only_eff: bool = False
     ) -> tuple['pd.DataFrame',
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
    import pandas as pd

    N_events = 0
    selected_events = 0

    # Fast path: only count efficiency without loading data into memory
    # Significantly faster for large datasets (millions of events)
    if only_eff:
        for f in files:
            with uproot.open(f) as file:
                N_events += file['eventsProcessed'].value
                tree = file['events']
                # Count entries directly without loading data
                selected_events += tree.num_entries
        return selected_events / N_events if N_events > 0 else 0.0

    # Full path: load data and return dataframe with efficiency
    dfs = []
    for f in files:
        with uproot.open(f) as file:
            N_events += file['eventsProcessed'].value
            # Read events tree
            tree = file['events']
            if tree.num_entries > 0:
                df_chunk = tree.arrays(vars, library='pd') if vars else tree.arrays(library='pd')
                dfs.append(df_chunk)

    # Concatenate all dataframes at once
    df = pd.concat(dfs, ignore_index=True, copy=False) if dfs else pd.DataFrame()
    eff = df.shape[0] / N_events if N_events > 0 else 0.0

    return df, eff, N_events

# ______________________
def additional_info(
    df: 'pd.DataFrame',
    mode: str,
    proc: str,
    sig: str
     ) -> 'pd.DataFrame':
    '''Add sample mode and signal label columns to dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): Sample mode/process name.
        proc (str): Underlying process name within the mode.
        sig (str): Signal process name for classification.

    Returns:
        pd.DataFrame: Dataframe with added 'sample' and 'isSignal' columns.
    '''
    df['sample']   = mode
    df['proc']     = proc
    df['isSignal'] = int(mode == sig)  # Add sample identifier and signal flag
    return df

# __________________________
def BDT_input_numbers(
    df: 'pd.DataFrame',
    modes: dict[str, list[str]],
    sig: str,
    eff: dict[str, float],
    xsec: dict[str, float],
    frac: dict[str, float],
    all_inputs: bool = False
     ) -> dict[str, int]:
    '''Calculate number of events to use for BDT training per process.

    Balances signal and background using cross-sections and efficiencies.

    Args:
        df (pd.DataFrame): Dataframe grouped by mode containing events.
        modes (dict[str, list[str]]): Mapping of modes to their process lists.
        sig (str): Signal process name.
        eff (dict[str, float]): Selection efficiency per mode.
        xsec (dict[str, float]): Cross-section per mode (in pb).
        frac (dict[str, float]): Fraction of available events to use per mode.

    Returns:
        dict[str, int]: Dictionary with number of BDT input events per mode.
    '''
    N_BDT_inputs: dict[str, int] = {}
    if all_inputs:
        LOGGER.info('Take all the events in the dataframes for training')
        return {m:df[m].shape[0] for m in modes}

    # Total background cross-section weighted by efficiency
    xsec_tot_bkg = sum(eff[mode] * xsec[mode] for mode in modes if mode != sig)
    if xsec_tot_bkg <= 0:
        LOGGER.warning('Total background normalization is zero; returning zero BDT inputs for backgrounds')
    for m in modes:
        N_BDT_inputs[m] = (
            int(frac[m] * df[m].shape[0]) if m == sig else
            int(frac[m] * df[sig].shape[0] * frac[sig] * (eff[m] * xsec[m] / xsec_tot_bkg)) if xsec_tot_bkg > 0 else 0)
    return N_BDT_inputs

# __________________________
def sample_df_by_xsec(
    df_mode: dict[str, 'pd.DataFrame'],
    proc_xsec: dict[str, float],
    proc_eff: dict[str, float],
    target_events: int,
    mode: str = '',
    random_state: int = 1,
    all_inputs: bool = False
     ) -> 'pd.DataFrame':
    '''Sample and concatenate process dataframes in proportion to eff * xsec.

    Args:
        df_mode (dict[str, pd.DataFrame]): Dataframes keyed by process name.
        proc_xsec (dict[str, float]): Cross-sections keyed by process name.
        proc_eff (dict[str, float]): Selection efficiencies keyed by process name.
        target_events (int): Total number of events to keep after sampling.
        mode (str, optional): Mode name used for log messages. Defaults to ''.
        random_state (int, optional): Random seed used for sampling. Defaults to 1.

    Returns:
        pd.DataFrame: Concatenated dataframe sampled in proportion to eff * xsec.
    '''
    import math
    import pandas as pd

    if not df_mode:
        return pd.DataFrame()


    available = {
        proc: df.shape[0]
        for proc, df in df_mode.items()
        if df.shape[0] > 0 and proc_xsec.get(proc, 0) > 0 and proc_eff.get(proc, 0) > 0
    }

    if not available:
        return pd.DataFrame()
    if all_inputs:
        LOGGER.debug('Returning the concatenation of all available process dataframe')
        return pd.concat([df_mode[proc] for proc in available], ignore_index=True)

    if len(available) == 1:
        proc = next(iter(available))
        LOGGER.debug(f'Only one process found for {mode}; keeping {proc} without resampling.')
        return df_mode[proc]

    total_available = sum(available.values())
    proc_weight = {
        proc: proc_xsec[proc] * proc_eff[proc]
        for proc in available
    }
    total_weight = sum(proc_weight.values())
    if total_weight <= 0:
        LOGGER.warning(
            f'Cannot sample {mode} proportionally: total eff*xsec is non-positive. '
            'Returning the concatenation of all available process dataframes.'
        )
        return pd.concat([df_mode[proc] for proc in available], ignore_index=True)

    max_feasible = min(
        math.floor(available[proc] * total_weight / proc_weight[proc])
        for proc in available
    )
    total_sampled = min(target_events, total_available, max_feasible)
    if total_sampled < target_events:
        LOGGER.info(
            f'Reducing total events for {mode} from {target_events:,} to {total_sampled:,} '
            'to preserve the expected process proportions.'
        )
    proc_width    = max(len(proc) for proc in available)

    ideal_counts = {
        proc: total_sampled * proc_weight[proc] / total_weight
        for proc in available
    }
    proc_targets = {
        proc: min(available[proc], int(ideal_counts[proc]))
        for proc in available
    }
    remaining = total_sampled - sum(proc_targets.values())
    ordered_procs = sorted(
        available,
        key=lambda proc: ideal_counts[proc] - proc_targets[proc],
        reverse=True,
    )

    while remaining > 0:
        progressed = False
        for proc in ordered_procs:
            if remaining == 0:
                break
            if proc_targets[proc] < available[proc]:
                proc_targets[proc] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    sampled_mode = []
    sampled_counts = {}
    for proc in available:
        n_target = proc_targets[proc]
        proc_df = df_mode[proc]
        if n_target < proc_df.shape[0]:
            sampled_df = proc_df.sample(n_target, random_state=random_state)
        else:
            sampled_df = proc_df
        sampled_mode.append(sampled_df)
        sampled_counts[proc] = sampled_df.shape[0]

    total_sampled = sum(sampled_counts.values())
    for proc in available:
        expected_fraction = (proc_weight[proc] / total_weight) * 100 if total_weight > 0 else 0.0
        actual_fraction = sampled_counts[proc] / total_sampled * 100 if total_sampled > 0 else 0.0
        if abs(expected_fraction - actual_fraction) > 1e-3:
            LOGGER.warning(f'Fraction in {mode:<{max(len(mode), 1)}} from {proc:<{proc_width}} = '
                           f'expected {expected_fraction:.3f}% | actual {actual_fraction:.3f}%')

    return pd.concat(sampled_mode, ignore_index=True)

# ________________________________
def df_split_data(
    df: 'pd.DataFrame',
    N_BDT_inputs: dict[str, int],
    eff: dict[str, float],
    xsec: dict[str, float],
    N_events: dict[str, int],
    mode: str,
    lumi: float = 10.8,
    test_size: float = 0.5
     ) -> 'pd.DataFrame':
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
    from sklearn.model_selection import train_test_split

    n_events = df.shape[0]
    if n_events == 0:
        LOGGER.error(f'No event for mode {mode}')
        exit(1)
    elif n_events < N_BDT_inputs[mode]:
        LOGGER.warning(f'For mode {mode}: n_events < BDT input ({n_events:,} vs {N_BDT_inputs[mode]:,})\n'
                       'Using the total number of events available')
        sampled: pd.DataFrame = df.sample(n_events, random_state=1)
    else:
        sampled: pd.DataFrame = df.sample(N_BDT_inputs[mode], random_state=1)

    # Split 50/50 into training and validation sets
    df0: pd.DataFrame; df1: pd.DataFrame
    df0, df1 = train_test_split(sampled, test_size=test_size, random_state=7)

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

# ______________________
def print_stats(
    df: 'pd.DataFrame',
    modes: list
     ) -> None:
    '''Print training and validation event counts per process.

    Args:
        df (pd.DataFrame): Dataframe containing 'sample', 'isSignal', and 'valid' columns.
        modes (list): List of process modes to display.

    Returns:
        None
    '''

    lenght, n = max(len(m) for m in modes), 110
    LOGGER.info(f"{' NUMBER OF BDT INPUT EVENTS ':=^{n}}")
    train = df['valid'] == False
    for m in modes:
        m_mask = df['sample']==m
        LOGGER.info(f"{f'Number of training for {m:<{lenght+2}} : {int((m_mask &  train).sum()):<10,}':^45}"
                    f"  {f'Number of validation for {m:<{lenght}} : {int((m_mask & ~train).sum()):<10,}':^45}".center(n))
    LOGGER.info(f"{'=' * n:^{n}}\n")

# ____________________________
def split_data(
    df: 'pd.DataFrame',
    vars: list[str],
    weight: str = 'norm_weight'
     ) -> tuple['np.ndarray',
                'np.ndarray',
                'np.ndarray',
                'np.ndarray',
                'np.ndarray',
                'np.ndarray']:
    '''Split data into training and validation sets for features and labels.

    Args:
        df (pd.DataFrame): Input dataframe with 'valid' column marking validation set.
        vars (list[str]): Feature variable names to extract.

    Returns:
        tuple: (X_train, y_train, X_valid, y_valid) as numpy arrays.
    '''
    import numpy as np

    train = df['valid'] == False
    # Features for training and validation sets
    X_train = df.loc[train,  vars].to_numpy(np.float32, copy=False)
    X_valid = df.loc[~train, vars].to_numpy(np.float32, copy=False)
    # Labels (signal/background) for training and validation sets
    y_train = df.loc[train,  'isSignal'].to_numpy(np.int8, copy=False).ravel()
    y_valid = df.loc[~train, 'isSignal'].to_numpy(np.int8, copy=False).ravel()

    train_weight = df.loc[train,  weight].to_numpy(copy=False).ravel()
    valid_weight = df.loc[~train, weight].to_numpy(copy=False).ravel()
    return X_train, y_train, X_valid, y_valid, train_weight, valid_weight

# ____________________________________
def train_model(
    X_train: 'np.ndarray',
    y_train: 'np.ndarray',
    X_valid: 'np.ndarray',
    y_valid: 'np.ndarray',
    train_weight: 'np.ndarray',
    valid_weight: 'np.ndarray',
    config: dict[str,
                 str | int | float | list[str]],
     ) -> 'xgb.XGBClassifier':
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
    import xgboost as xgb

    bdt = xgb.XGBClassifier(**config)
    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    LOGGER.info('Beginning the training')
    bdt.fit(X_train, y_train, eval_set=eval_set, verbose=True, sample_weight_eval_set=[train_weight, valid_weight])
    return bdt

# ____________________________
def evaluate_bdt(
    df: 'pd.DataFrame',
    bdt: 'xgb.XGBClassifier',
    vars: list[str]
     ) -> 'pd.DataFrame':
    '''Compute BDT scores and add to dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        bdt (xgb.XGBClassifier): Trained XGBoost classifier.
        vars (list[str]): Feature variables for prediction.

    Returns:
        pd.DataFrame: Dataframe with 'BDTscore' column added.
    '''
    import numpy as np

    # Extract features as float32 numpy array
    X = df.loc[:, vars].to_numpy(
        dtype=np.float32, copy=False
    )
    # Add probability of signal class as BDT score
    df['BDTscore'] = bdt.predict_proba(X)[:, 1]
    return df

# ___________________________________________
def get_metrics(
    bdt: 'xgb.XGBClassifier'
     ) -> tuple[dict[str,
                     dict[str, list[float]]],
                int, 'np.ndarray', int]:
    '''Extract training metrics from trained model.

    Args:
        bdt (xgb.XGBClassifier): Trained XGBoost classifier.

    Returns:
        tuple: (results dict, num_epochs, epoch_axis, best_epoch).
    '''
    import numpy as np

    results = bdt.evals_result()
    epochs = len(results['validation_0']['error'])
    # X-axis values for epoch numbering
    x_axis = np.arange(0, epochs, 1)
    # Best iteration from early stopping or total epochs
    best_iteration = (bdt.best_iteration + 1) if \
        (bdt.best_iteration is not None) else epochs
    return results, epochs, x_axis, best_iteration

# _________________________________________________
def load_model(inDir: str) -> 'xgb.XGBClassifier':
    '''Load previously trained XGBoost model.

    Args:
        inDir (str): Directory path containing xgb_bdt.joblib file.

    Returns:
        xgb.XGBClassifier: Loaded XGBoost classifier.
    '''
    import joblib
    fpath = f'{inDir}/xgb_bdt.joblib'
    LOGGER.info(f'Loading BDT model from {fpath}')
    return joblib.load(fpath)

# ____________________________
def save_model(
    bdt: 'xgb.XGBClassifier',
    vars: list[str],
    path: str
     ) -> None:
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
    LOGGER.info(f'Saving BDT in a .root file at {froot}')
    # Save model in TMVA ROOT format
    ROOT.TMVA.Experimental.SaveXGBoost(
        bdt, 'ZH_Recoil_BDT', froot, num_inputs=len(vars)
    )
    # Create list of feature variable names
    variables = ROOT.TList()
    for var in vars:
        variables.Add(ROOT.TObjString(var))

    # Write variable list to ROOT file
    with ROOT.TFile(froot, 'UPDATE') as fOut:
        fOut.WriteObject(variables, 'variables')

    LOGGER.info(f'Saving BDT in a .joblib file at {fjob}')
    # Save model in joblib format for Python
    joblib.dump(bdt, fjob, compress=2)

# ______________________________________
def def_bdt(
    loc_bdt: str,
    MVAVec:  str = 'MVAVec',
    score:   str = 'BDTscore',
    defineList: dict[str, str] = {},
    suffix: str = ''
     ) -> tuple[dict[str, str], float]:
    '''Define BDT computation in ROOT RDataFrame and load cut value.

    Args:
        vars (Sequence[str]): Sequence of feature variable names to use for BDT scoring.
        loc_bdt (str): Directory containing xgb_bdt.root model file.
        MVAVec (str, optional): Name for the feature vector column. Defaults to 'MVAVec'.
        score (str, optional): Name for BDT score column. Defaults to 'BDTscore'.
        defineList (dict[str, str], optional): Dictionary to append column definitions to. Defaults to {}.
        suffix (str, optional): Suffix to append to output column names. Defaults to ''.

    Returns:
        tuple: (updated defineList, BDT cut threshold).
    '''
    import uproot, ROOT
    import numpy as np

    # Load TMVA model from ROOT file
    ROOT.gInterpreter.ProcessLine(f'''
        TMVA::Experimental::RBDT<> tmva("ZH_Recoil_BDT", "{loc_bdt}/xgb_bdt.root");
    ''')

    # Get the BDT inputs from the .root file
    tlist = uproot.open(f'{loc_bdt}/xgb_bdt.root')['variables']
    var_list = ', (float)'.join([str(x) for x in tlist])


    # Define feature vector if not already present
    if MVAVec not in defineList:
        defineList[MVAVec] = f'ROOT::VecOps::RVec<float>{{{var_list}}}'
    # Compute BDT score
    defineList['mva_score'+suffix] = f'tmva.Compute({MVAVec})'
    defineList[score+suffix] = 'mva_score.at(0)'

    # Load BDT cut value from file
    bdt_cut  = float(np.loadtxt(f'{loc_bdt}/BDT_cut.txt'))
    return defineList, bdt_cut

# ___________________________
def make_high_low(
    cutList: dict[str, str],
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

    valid_sels = [sel for sel in sels if sel in cutList]
    LOGGER.info(f'Adding selection with BDT separation for\n{" ".join(valid_sels)}')
    for sel in valid_sels:
        old_cut = cutList[sel]
        # Add high BDT score region (signal-like)
        cutList[sel+'_high'] = old_cut + f' && {score} > {bdt_cut}'
        # Add low BDT score region (background-like)
        cutList[sel+'_low'] = old_cut  + f' && {score} < {bdt_cut}'
    return cutList

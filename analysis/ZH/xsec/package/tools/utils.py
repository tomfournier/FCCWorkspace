'''Utility functions for tdata procession and analysis.

Provides:
- File and metadata I/O: `get_paths()`, `get_df()`, `mkdir()`, `load_data()`,
    `to_pkl()`, `dump_json()`, `load_json()`, `get_procDict()`, `update_keys()`,
    `get_xsec()`.
- Significance calculators: `Z0()`, `Zmu()`, `Z()`, `Significance()`.
- ROOT histogram helpers: `get_stack()`, `get_xrange()`, `get_yrange()`,
    `get_range()`, `get_range_decay()`.

Conventions:
- ROOT input is expected to contain a TTree named 'events' (used by `get_df`).
- Process dictionaries are searched via `$FCCDICTSDIR` (first path segment) and
    fall back to `/cvmfs/fcc.cern.ch/FCCDicts`.
- `get_paths()` uses a `modes` mapping to build file globs and returns `.root`
    file paths, optionally appending a `suffix`.
- Range utilities: `strict=True` excludes empty bins; `logY=True` ignores zero/
    negative contents when computing minima; `stack=True` bases y-max on sums.
- Significance helpers return `nan` for invalid inputs (e.g., `B<=0`).

Lazy Imports:
- Heavy dependencies (numpy, pandas) are only imported when needed in functions.
- Type hints use TYPE_CHECKING guard to avoid circular imports and startup time.
'''
from __future__ import annotations

import os, json

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

#_____________________________
def get_paths(mode: str, 
              path: str, 
              modes: str, 
              suffix: str = ''
              ) -> list:
    '''
    Retrieve ROOT file paths based on mode and suffix.

    Args:
        mode (str): The mode key to filter paths.
        path (str): Base directory path to search.
        modes (dict): Mapping of mode names to directory patterns.
        suffix (str, optional): File suffix to append. Defaults to ''.

    Returns:
        list: Matching ROOT file paths.
    '''
    from glob import glob

    # Construct full path from base path and mode pattern
    fpath = os.path.join(path, modes[mode] + suffix)
    return glob(f'{fpath}.root')


#__________________________________
def get_df(filename: str, 
           branches: list[str] = []
           ) -> pd.DataFrame:
    '''
    Load a DataFrame from a ROOT file.

    Args:
        filename (str): Path to the ROOT file.
        branches (list[str], optional): Specific branches to load. If empty, loads all. Defaults to [].

    Returns:
        pd.DataFrame: DataFrame containing the 'events' tree data.
    '''
    import pandas as pd
    from uproot import open

    with open(filename) as file:
        tree = file['events']
        # Return empty DataFrame if tree has no entries
        if tree.num_entries == 0:
            return pd.DataFrame()
        # Load specific branches or all branches
        if branches:
            return tree.arrays(branches, library='pd')
        return tree.arrays(library='pd')

#___________________
def mkdir(mydir: str
          ) -> None:
    '''
    Create a directory if it does not exist.

    Args:
        mydir (str): The directory path to create.
    '''

    os.makedirs(mydir, exist_ok=True)


#________________________________________________________
def get_procDict(procFile: str, 
                 fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts'
                 ) -> dict[str, 
                           dict[str, 
                                float]]:
    '''
    Load process dictionary from a JSON file.

    Args:
        procFile (str): Name of the process dictionary file.
        fcc (str, optional): Base directory for FCC dictionaries. Defaults to '/cvmfs/fcc.cern.ch/FCCDicts'.

    Returns:
        dict: Process dictionary with cross-section and other metadata.

    Raises:
        FileNotFoundError: If the process dictionary file is not found.
    '''

    # Check environment variable for FCC dictionaries directory
    env = os.getenv('FCCDICTSDIR')
    base_dir = env.split(':')[0] if env else fcc
    proc_path = os.path.join(base_dir, procFile)

    if not os.path.isfile(proc_path):
        raise FileNotFoundError(f'----> No procDict found: ==={proc_path}===')

    with open(proc_path, 'r') as f:
        procDict = json.load(f)
    return procDict


#________________________________________________
def update_keys(procDict: dict[str,
                               dict[str,
                                    float]], 
                modes: list
                ) -> dict[str, 
                          dict[str, 
                               float]]:
    '''
    Update dictionary keys by reversing mode name mappings.

    Args:
        procDict (dict): Original process dictionary.
        modes (list): Mode name mappings.

    Returns:
        dict: Dictionary with updated keys.
    '''

    # Create reverse mapping from mode values to keys
    reversed_mode_names = {v: k for k, v in modes.items()}
    
    # Apply reverse mapping to dictionary keys
    updated_dict = {}
    for key, value in procDict.items():
        new_key = reversed_mode_names.get(key, key)
        updated_dict[new_key] = value
    return updated_dict


#__________________________________
def get_xsec(modes: list, 
             training: bool = True
             ) -> dict[str, 
                       float]:
    '''
    Retrieve cross-section values for specified modes.

    Args:
        modes (list): List of modes to retrieve cross-sections for.
        training (bool, optional): Use training dataset dictionary if True. Defaults to True.

    Returns:
        dict: Dictionary mapping modes to their cross-section values.
    '''

    # Select appropriate process dictionary based on training flag
    if training:
        procFile = 'FCCee_procDict_winter2023_training_IDEA.json'
    else:
        procFile = 'FCCee_procDict_winter2023_IDEA.json'
    
    proc_dict = get_procDict(procFile)
    procDict  = update_keys(proc_dict, modes)

    # Extract cross-section values for specified modes
    xsec = {}
    for key, value in procDict.items(): 
        if key in modes: xsec[key] = value['crossSection']
    return xsec


#___________________________________________
def load_data(inDir: str, 
              filename: str = 'preprocessed'
              ) -> pd.DataFrame:
    '''
    Load preprocessed data from a pickle file.

    Args:
        inDir (str): Input directory path.
        filename (str, optional): Filename without extension. Defaults to 'preprocessed'.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    '''
    import pandas as pd

    # Construct pickle file path and load
    fpath = os.path.join(inDir, filename+'.pkl')
    df = pd.read_pickle(fpath)
    return df


#________________________________________
def to_pkl(df: pd.DataFrame, 
           path: str, 
           filename: str = 'preprocessed'
           ) -> None:
    '''
    Save a DataFrame to a pickle file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Output directory path.
        filename (str, optional): Filename without extension. Defaults to 'preprocessed'.
    '''

    mkdir(path)
    fpath = os.path.join(path, filename+'.pkl')
    df.to_pickle(fpath)
    print(f'\n----->[Info] Preprocessed saved {fpath}\n')

#____________________________
def dump_json(arg: dict, 
              file: str, 
              indent: int = 4
              ) -> None:
    '''
    Dump a dictionary to a JSON file.

    Args:
        arg (dict): Dictionary to save.
        file (str): Output file path.
        indent (int, optional): JSON indentation level. Defaults to 4.
    '''

    with open(file, mode='w', encoding='utf-8') as fOut:
        json.dump(arg, fOut, indent=indent)

#_______________________
def load_json(file: str
              ) -> dict:
    '''
    Load a dictionary from a JSON file.

    Args:
        file (str): Path to JSON file.

    Returns:
        dict: Loaded dictionary.
    '''

    with open(file, mode='r', 
              encoding='utf-8') as fIn:
        arg = json.load(fIn)
    return arg


#_________________
def Z0(S: float, 
       B: float
       ) -> float:
    '''
    Calculate significance using the Z0 method.

    Args:
        S (float): Signal value.
        B (float): Background value.

    Returns:
        float: Calculated significance (NaN if B <= 0).
    '''
    import numpy as np
    
    if B<=0:
        return np.nan
    return np.sqrt( 2*( (S + B)*np.log(1 + S/B) - S ) )


#__________________
def Zmu(S: float, 
        B: float
        ) -> float:
    '''
    Calculate significance using the Zmu method.

    Args:
        S (float): Signal value.
        B (float): Background value.

    Returns:
        float: Calculated significance (NaN if B <= 0).
    '''
    import numpy as np
    
    if B<=0:
        return np.nan
    return np.sqrt( 2*( S - B*np.log(1 + S/B) ) )


#________________
def Z(S: float, 
      B: float
      ) -> float:
    '''
    Calculate significance using the Z method (simple S/sqrt(S+B)).

    Args:
        S (float): Signal value.
        B (float): Background value.

    Returns:
        float: Calculated significance (0.0 if both S and B are <= 0, NaN if B < 0).
    '''
    import numpy as np
    
    if B<0:
        return np.nan
    if S<=0 and B<=0:
        return 0.0
    return S/np.sqrt(S + B)


#___________________________________________________________
def Significance(df_s: pd.DataFrame, 
                 df_b: pd.DataFrame, 
                 column: str = 'BDTscore',
                 weight: str = 'norm_weight',
                 func: Callable[[float, float], float] = Z0, 
                 score_range: tuple[float, float] = (0, 1), 
                 nbins: int = 50) -> pd.DataFrame:      
    '''Calculate significance from signal and background DataFrames.
    
    Optimized for speed: vectorized numpy operations, single pass binning.

    Args:
        df_s (pd.DataFrame): DataFrame containing signal data.
        df_b (pd.DataFrame): DataFrame containing background data.
        column (str, optional): Column name for scoring. Defaults to 'BDTscore'.
        weight (str, optional): Column name for event weights. Defaults to 'norm_weight'.
        func (Callable, optional): Function to calculate significance. Defaults to Z0.
        score_range (tuple, optional): Score range (min, max) for binning. Defaults to (0, 1).
        nbins (int, optional): Number of histogram bins. Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame with columns ['S', 'B', 'Z'] for signal, background, and significance at each bin edge.
    '''
    import numpy as np
    import pandas as pd

    # Extract values and weights as numpy arrays (no intermediate copies)
    s_vals, s_w = df_s[column].values, df_s[weight].values
    b_vals, b_w = df_b[column].values, df_b[weight].values

    S0, B0 = s_w.sum(), b_w.sum()
    print('initial: S0 = {:.2f}, B0 = {:.2f}'.format(S0, B0))
    print('inclusive Z: {:.2f}'.format(func(S0, B0)))

    # Bin data once and compute cumulative sums
    edges = np.linspace(*score_range, nbins + 1)
    hist_s, _ = np.histogram(s_vals, bins=edges, weights=s_w)
    hist_b, _ = np.histogram(b_vals, bins=edges, weights=b_w)

    # Cumulative sums from high to low score (avoid loop)
    S_cum = np.cumsum(hist_s[::-1])[::-1]
    B_cum = np.cumsum(hist_b[::-1])[::-1]

    # Vectorized significance calculation
    Z_vals = np.array([func(Si, Bi) for Si, Bi in zip(S_cum, B_cum)])
    
    return pd.DataFrame(data={'S': S_cum, 'B': B_cum, 'Z': Z_vals}, 
                        index=edges[:-1])

#_________________________________________
def high_low_sels(sels: list[str], 
                  list_hl: str | list[str]
                  ) -> list[str]:
    if isinstance(list_hl, str): list_hl = [list_hl]

    valid_sels = [hl for hl in list_hl if hl in sels]
    for sel in valid_sels:
        sels.extend([sel+'_high', sel+'_low'])
    return sels

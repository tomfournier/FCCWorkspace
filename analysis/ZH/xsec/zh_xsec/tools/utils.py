'''Utility functions for data processing and analysis.

Provides:
- File and metadata I/O: `get_paths()`, `get_df()`, `mkdir()`, `load_data()`,
    `to_pkl()`, `dump_json()`, `load_json()`, `get_procDict()`, `update_keys()`,
    `get_xsec()`.
- Significance calculators: `Z0()`, `Zmu()`, `Z()`, `Significance()`.
- Selection utilities: `high_low_sels()`.

Conventions:
- ROOT input is expected to contain a TTree named 'events' (used by `get_df`).
- Process dictionaries are searched via `$FCCDICTSDIR` (first path segment) and
    fall back to `/cvmfs/fcc.cern.ch/FCCDicts`.
- `get_paths()` uses a `modes` mapping to build file globs and returns `.root`
    file paths, optionally appending a `suffix`.
- Significance helpers return `nan` for invalid inputs (e.g., `B<=0`).

Lazy Imports:
- Heavy dependencies (numpy, pandas) are only imported when needed in functions.
- Type hints use TYPE_CHECKING guard to avoid circular imports and startup time.
'''
from __future__ import annotations

import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from logger import get_logger
LOGGER = get_logger(__name__)


def update_keys(
    procDict: dict[str, dict[str, float]],
    modes: dict[str, list[str]]
     ) -> dict[str, dict[str, float]]:
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


def get_xsec(
    modes: dict[str, list[str]],
    training: bool = True
     ) -> dict[str, float]:
    '''
    Retrieve cross-section values for specified modes.

    Args:
        modes (list): List of modes to retrieve cross-sections for.
        training (bool, optional): Use training dataset dictionary if True. Defaults to True.

    Returns:
        dict: Dictionary mapping modes to their cross-section values.
    '''
    from tools.utils import get_procDict

    # Select appropriate process dictionary based on training flag
    procFile = 'FCCee_procDict_winter2023{}_IDEA.json'.format(
        '_training' if training else '')

    proc_dict = get_procDict(procFile)
    procDict  = update_keys(proc_dict, modes)

    # Extract cross-section values for specified modes
    xsec = {}
    for key, value in procDict.items():
        if key in modes: xsec[key] = value['crossSection']
    return xsec


def data_from_pkl(
    inDir: str,
    filename: str = 'preprocessed'
     ) -> tuple[pd.DataFrame, list[str]]:
    '''
    Load preprocessed data from a pickle file.

    Args:
        inDir (str): Input directory path.
        filename (str, optional): Filename without extension. Defaults to 'preprocessed'.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    '''
    import pickle

    # Construct pickle file path and load
    fpath = os.path.join(inDir, filename+'.pkl')
    data  = pickle.load(open(fpath, 'rb'))
    df, input_vars = data['data'], data['variables']
    LOGGER.info('Training variable used for the training\n' +
                ', '.join(input_vars) + '\n')
    return df, input_vars


def data_to_pkl(
    df: pd.DataFrame,
    input_vars: list[str],
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
    import pickle
    from tools.utils import mkdir

    mkdir(path)
    save  = {'data': df, 'variables': input_vars}
    fpath = os.path.join(path, filename+'.pkl')
    pickle.dump(save, open(fpath, 'wb'))
    LOGGER.info(f'Preprocessed saved {fpath}')


def high_low_sels(
    sels: list[str],
    list_hl: str | list[str]
     ) -> list[str]:
    '''Extend selection list with '_high' and '_low' variants.

    For each selection name in list_hl that exists in sels, adds both
    sel+'_high' and sel+'_low' variants to the list.

    Args:
        sels (list[str]): List of selection names to extend.
        list_hl (str | list[str]): Selection name(s) to add high/low variants for.
            Can be a single string or list of strings.

    Returns:
        list[str]: Extended selection list with '_high' and '_low' variants.
    '''
    if isinstance(list_hl, str):
        list_hl = [list_hl]

    valid_sels = [hl for hl in list_hl if hl in sels]
    for sel in valid_sels:
        sels.extend([sel+'_high', sel+'_low'])
    return sels

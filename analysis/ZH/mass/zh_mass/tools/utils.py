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

from ..logger import get_logger

LOGGER = get_logger(__name__)

# __________________
def get_paths(
    mode: str,
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


# __________________________________________
def get_procDict(
    procFile: str,
    fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts'
     ) -> dict[str, dict[str, float]]:
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
        LOGGER.error(f'No procDict found: {proc_path}')
        exit(1)

    with open(proc_path, 'r') as f:
        procDict = json.load(f)
    return procDict


# ________________________________________
def update_keys(
    procDict: dict[str, dict[str, float]],
    modes: list[str]
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


# ________________________
def get_xsec(
    modes: list[str],
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


# _________________
def dump_json(
    arg: dict,
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

# _____________
def load_json(
    file: str
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


# __________________________
def high_low_sels(
    sels: list[str],
    list_hl: str | list[str]
     ) -> list[str]:
    if isinstance(list_hl, str):
        list_hl = [list_hl]

    valid_sels = [hl for hl in list_hl if hl in sels]
    for sel in valid_sels:
        sels.extend([sel+'_high', sel+'_low'])
    return sels

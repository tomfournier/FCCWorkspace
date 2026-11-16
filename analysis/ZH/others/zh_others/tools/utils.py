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


# __________________________________________
def get_procDict(
    procFile: str,
    fcc_path: str = '/cvmfs/fcc.cern.ch/FCCDicts'
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

    import json

    # Check environment variable for FCC dictionaries directory
    env = os.getenv('FCCDICTSDIR')
    base_dir = env.split(':')[0] if env else fcc_path
    proc_path = os.path.join(base_dir, procFile)

    if not os.path.isfile(proc_path):
        LOGGER.error(f'No procDict found: {proc_path}')
        exit(1)

    with open(proc_path, 'r') as f:
        procDict = json.load(f)
    return procDict



# ________________________________
def load_data(
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
    data = pickle.load(open(fpath, 'rb'))
    df, input_vars = data['data'], data['variables']
    LOGGER.info('Training variable used for the training\n' +
                ', '.join(input_vars) + '\n')
    return df, input_vars


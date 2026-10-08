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

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from logger import get_logger
LOGGER = get_logger(__name__)


def get_paths(
    proc: str,
    path: str,
    suffix: str = ''
) -> list[str]:
    '''
    Retrieve ROOT file paths based on mode and suffix.

    Args:
        proc (str): The proc key to filter paths.
        path (str): Base directory path to search.
        suffix (str, optional): File suffix to append. Defaults to ''.

    Returns:
        list: Matching ROOT file paths.
    '''
    from glob import glob

    # Construct full path from base path and mode pattern
    fpath = os.path.join(path, proc + suffix)
    if os.path.exists(fpath+'.root'):
        return [fpath+'.root']
    elif os.path.exists(fpath):
        return glob(f'{fpath}/*')
    else:
        LOGGER.error(f'{fpath} not found')
        exit(1)


def get_df(
    filename: str,
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
    import uproot, pandas as pd

    with uproot.open(filename) as file:
        tree = file['events']
        # Return empty DataFrame if tree has no entries
        if tree.num_entries == 0:
            return pd.DataFrame()
        # Load specific branches or all branches
        if branches:
            return tree.arrays(branches, library='pd')
        return tree.arrays(library='pd')


def mkdir(mydir: str) -> None:
    '''
    Create a directory if it does not exist.

    Args:
        mydir (str): The directory path to create.
    '''

    os.makedirs(mydir, exist_ok=True)


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
    base_dir  = env.split(':')[0] if env else fcc_path
    proc_path = os.path.join(base_dir, procFile)

    if not os.path.isfile(proc_path):
        LOGGER.error(f'No procDict found: {proc_path}')
        exit(1)

    with open(proc_path, 'r') as f:
        procDict = json.load(f)
    return procDict


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
    import json

    with open(file, mode='r', encoding='utf-8') as fIn:
        arg = json.load(fIn)
    return arg


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
    import json

    with open(file, mode='w', encoding='utf-8') as fOut:
        json.dump(arg, fOut, indent=indent)


def Z0(S: float | int, B: float | int) -> float:
    '''
    Calculate significance using the Z0 method.

    Args:
        S (float): Signal value.
        B (float): Background value.

    Returns:
        float: Calculated significance (NaN if B <= 0).
    '''
    import numpy as np

    if B<=0: return np.nan
    return np.sqrt(2*((S + B)*np.log(1 + S/B) - S))


def Zmu(S: float | int, B: float | int) -> float:
    '''
    Calculate significance using the Zmu method.

    Args:
        S (float): Signal value.
        B (float): Background value.

    Returns:
        float: Calculated significance (NaN if B <= 0).
    '''
    import numpy as np

    if B<=0: return np.nan
    return np.sqrt(2*(S - B*np.log(1 + S/B)))


def Z(S: float | int, B: float | int) -> float:
    '''
    Calculate significance using the Z method (simple S/sqrt(S+B)).

    Args:
        S (float): Signal value.
        B (float): Background value.

    Returns:
        float: Calculated significance (0.0 if both S and B are <= 0, NaN if B < 0).
    '''
    import numpy as np

    if B<0: return np.nan
    if S<=0 and B<=0: return 0.0
    return S/np.sqrt(S + B)


def Significance(
        df_s: pd.DataFrame,
        df_b: pd.DataFrame,
        column: str = 'BDTscore',
        weight: str = 'norm_weight',
        func: Callable[[float | int, float | int], float] = Z0,
        score_range: tuple[float, float] = (0, 1),
        nbins: int = 50
) -> pd.DataFrame:
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
    LOGGER.debug(f'Initial:   S0 = {S0:.2f}, B0 = {B0:.2f}')
    LOGGER.debug(f'Inclusive: Z  = {func(S0, B0):.2f}')

    # Bin data once and compute cumulative sums
    edges     = np.linspace(*score_range, nbins + 1)
    hist_s, _ = np.histogram(s_vals, edges, weights=s_w)
    hist_b, _ = np.histogram(b_vals, edges, weights=b_w)

    # Cumulative sums from high to low score (avoid loop)
    S_cum = np.cumsum(hist_s[::-1])[::-1]
    B_cum = np.cumsum(hist_b[::-1])[::-1]

    # Vectorized significance calculation
    Z_vals = np.array([func(Si, Bi) for Si, Bi in zip(S_cum, B_cum)])

    return pd.DataFrame({'S': S_cum, 'B': B_cum, 'Z': Z_vals}, edges[:-1])


def timer(t: float) -> None:
    '''Log formatted elapsed time since provided timestamp.

    Calculates and logs elapsed time in human-readable format (hours, minutes,
    seconds, milliseconds) with formatted header and footer separators.

    Args:
        t: Starting timestamp from time.time().
    '''
    import time
    dt = time.time() - t

    # Split time into components
    h, m  = int(dt // 3600), int(dt // 60 % 60),
    s, ms = int(dt % 60),    int((dt % 1) * 1000)

    # Build time string with non-zero components
    time_parts = []
    if h  > 0: time_parts.append(f'{h} h')
    if m  > 0: time_parts.append(f'{m} min')
    if s  > 0: time_parts.append(f'{s} s')
    if ms > 0: time_parts.append(f'{ms} ms')
    if not time_parts: time_parts.append('0 ms')

    elapsed = f"Elapsed time: {' '.join(time_parts)}"
    lenght = len(elapsed) + 4

    LOGGER.info(f'\n{" CODE ENDED ":=^{lenght}}\n{elapsed:^{lenght}}\n{"="*lenght}\n')
    return None

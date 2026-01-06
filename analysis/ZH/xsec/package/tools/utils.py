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
'''

import os, json, ROOT

import numpy as np
import pandas as pd

from glob import glob
from typing import Callable, Union

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
    print(f'--->Preprocessed saved {fpath}')

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
    '''
    Calculate significance from signal and background DataFrames.

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

    # Extract values and weights from DataFrames
    s_vals, s_w = df_s[column].values, df_s[weight].values
    b_vals, b_w = df_b[column].values, df_b[weight].values

    S0, B0 = s_w.sum(), b_w.sum()
    print('initial: S0 = {:.2f}, B0 = {:.2f}'.format(S0, B0))
    print('inclusive Z: {:.2f}'.format(func(S0, B0)))

    # Build weighted histograms with cumulative sums from high score to low score
    edges     = np.linspace(*score_range, nbins + 1)
    hist_s, _ = np.histogram(s_vals, bins=edges, weights=s_w)
    hist_b, _ = np.histogram(b_vals, bins=edges, weights=b_w)

    # Cumulative sums from high score to low score (optimization)
    S_cum = np.cumsum(hist_s[::-1])[::-1]
    B_cum = np.cumsum(hist_b[::-1])[::-1]

    Z_vals = np.array([(Si, Bi, func(Si, Bi)) for Si, Bi in zip(S_cum, B_cum)])
    return pd.DataFrame(data=Z_vals, index=edges[:-1], columns=['S', 'B', 'Z'])

#__________________________________
def get_stack(hists: list[ROOT.TH1]
              ) -> ROOT.TH1:
    '''
    Create a stacked histogram from a list of histograms.

    Args:
        hists (list[ROOT.TH1]): List of histograms to stack.

    Returns:
        ROOT.TH1: Combined stacked histogram.

    Raises:
        ValueError: If the input list is empty.
    '''

    if not hists:
        raise ValueError('get_stack requires at least one histogram')
    
    # Clone first histogram and add remaining histograms
    hist = hists[0].Clone()
    hist.SetDirectory(0)
    hist.SetName(hist.GetName() + '_stack')
    for h in hists[1:]: 
        hist.Add(h)
    return hist

#___________________________________________________
def get_xrange(hist: ROOT.TH1, 
               strict: bool = True, 
               xmin: Union[float, int, None] = None,
               xmax: Union[float, int, None] = None
               ) -> tuple[Union[float, int], 
                          Union[float, int]]:
    '''
    Get the x-range of a histogram based on bin content.

    Args:
        hist (ROOT.TH1): Histogram to analyze.
        strict (bool, optional): Only consider bins with non-zero content. Defaults to True.
        xmin (float | int | None, optional): Minimum x boundary to consider. Defaults to None.
        xmax (float | int | None, optional): Maximum x boundary to consider. Defaults to None.

    Returns:
        tuple: (x_min, x_max) range of the histogram.
    '''

    nbins = hist.GetNbinsX()
    bin_data = [(hist.GetBinLowEdge(i+1), 
                 hist.GetBinLowEdge(i+2), 
                 hist.GetBinContent(i+1)) for i in range(nbins)]

    # Filter bins based on strict mode and boundary conditions
    mask = [(le, he) for le, he, c in bin_data 
            if (xmin is None or le > xmin) 
            and (xmax is None or he < xmax)
            and (not strict or c != 0)]

    if not mask:
        # Fallback to full range if nothing matched
        return bin_data[0][0], bin_data[-1][1]

    return min(m[0] for m in mask), max(m[1] for m in mask)

#___________________________________________________
def get_yrange(hist: ROOT.TH1, 
               logY: bool,
               ymin: Union[float, int, None] = None,
               ymax: Union[float, int, None] = None,
               scale_min: float = 1.,
               scale_max: float = 1.,
               ) -> tuple[Union[float, int], 
                          Union[float, int]]:
    '''
    Get the y-range of a histogram.

    Args:
        hist (ROOT.TH1): Histogram to analyze.
        logY (bool): Flag indicating if the y-axis is logarithmic.
        ymin (float | int | None, optional): Minimum y value. Defaults to None.
        ymax (float | int | None, optional): Maximum y value. Defaults to None.
        scale_min (float, optional): Minimum scale for y-axis. Defaults to 1.
        scale_max (float, optional): Maximum scale for y-axis. Defaults to 1.

    Returns:
        tuple: (y_min, y_max) range of the histogram.
    '''

    nbins = hist.GetNbinsX()
    contents = np.array([hist.GetBinContent(i+1) \
                         for i in range(nbins)], dtype=float)

    if logY:
        # Handle log scale: exclude zero and negative values
        nonzero = contents[contents != 0]
        if nonzero.size == 0:
            return np.nan, np.nan
        yMin = float(nonzero.min()) * scale_min
    else:
        yMin = float(contents.min()) * scale_min

    yMax = float(contents.max()) * scale_max
    if (ymin is not None) and ymin > yMin: yMin = ymin
    if (ymax is not None) and ymax < yMax: yMax = ymax
    return yMin, yMax

#__________________________________________________
def get_range(h_sigs: list[ROOT.TH1], 
              h_bkgs: list[ROOT.TH1],
              logY: bool = False, 
              strict: bool = True, 
              stack: bool = False,
              scale_min: float = 1., 
              scale_max: float = 1.,
              xmin: Union[float, int, None] = None,
              xmax: Union[float, int, None] = None,
              ymin: Union[float, int, None] = None,
              ymax: Union[float, int, None] = None
              ) -> tuple[float, 
                         float, 
                         float, 
                         float]:
    '''
    Determine the range for signal and background histograms.

    Args:
        h_sigs (list[ROOT.TH1]): List of signal histograms.
        h_bkgs (list[ROOT.TH1]): List of background histograms.
        logY (bool, optional): Flag for logarithmic y-axis. Defaults to False.
        strict (bool, optional): Flag for strict range checking. Defaults to True.
        stack (bool, optional): Flag for stacking histograms. Defaults to False.
        scale_min (float, optional): Minimum scale for y-axis. Defaults to 1.
        scale_max (float, optional): Maximum scale for y-axis. Defaults to 1.
        xmin (float | int | None, optional): Minimum x value. Defaults to None.
        xmax (float | int | None, optional): Maximum x value. Defaults to None.
        ymin (float | int | None, optional): Minimum y value. Defaults to None.
        ymax (float | int | None, optional): Maximum y value. Defaults to None.

    Returns:
        tuple: (xmin, xmax, ymin, ymax) ranges for plotting.
    '''
    # Stack all signal and background histograms
    h_stack = get_stack(h_sigs + h_bkgs)

    # Determine x-range from stacked histogram
    xMin, xMax = get_xrange(h_stack, 
                            strict=strict, 
                            xmin=xmin, 
                            xmax=xmax)
    h_bkg = [get_stack(h_bkgs)]
    
    # Get y-min values from all histograms
    yMin = np.array([
        get_yrange(h, 
                    logY=logY,
                    ymin=ymin,
                    ymax=ymax,
                    scale_min=scale_min,
                    scale_max=scale_max)[0] 
                    for h in h_sigs + h_bkgs
    ])
    
    # Get y-max values based on stacking option
    if stack:
        yMax = get_yrange(h_stack, 
                          logY=logY, 
                          ymin=ymin, 
                          ymax=ymax,
                          scale_min=scale_min,
                          scale_max=scale_max)[1]
    else:
        yMax = np.array([
            get_yrange(h, 
                       logY=logY, 
                       ymin=ymin, 
                       ymax=ymax,
                       scale_min=scale_min,
                       scale_max=scale_max)[1] 
                       for h in h_sigs + h_bkg
        ])
        
    yMin = yMin.min()
    yMax = yMax.max()
    return xMin, xMax, yMin, yMax

#________________________________________________________
def get_range_decay(h_sigs: list[ROOT.TH1], 
                    logY: bool = False, 
                    strict: bool = True,
                    scale_min: float = 1., 
                    scale_max: float = 1.,
                    xmin: Union[float, int, None] = None,
                    xmax: Union[float, int, None] = None,
                    ymin: Union[float, int, None] = None,
                    ymax: Union[float, int, None] = None
                    ) -> tuple[float, 
                               float, 
                               float, 
                               float]:
    '''
    Determine the range for decay mode histograms.

    Args:
        h_sigs (list[ROOT.TH1]): List of signal histograms.
        logY (bool, optional): Flag for logarithmic y-axis. Defaults to False.
        strict (bool, optional): Flag for strict range checking. Defaults to True.
        scale_min (float, optional): Minimum scale for y-axis. Defaults to 1.
        scale_max (float, optional): Maximum scale for y-axis. Defaults to 1.
        xmin (float | int | None, optional): Minimum x value. Defaults to None.
        xmax (float | int | None, optional): Maximum x value. Defaults to None.
        ymin (float | int | None, optional): Minimum y value. Defaults to None.
        ymax (float | int | None, optional): Maximum y value. Defaults to None.

    Returns:
        tuple: (xmin, xmax, ymin, ymax) ranges for plotting decay modes.
    '''
    # Stack signal histograms
    h_sig = get_stack(h_sigs)
    
    # Get x-range from stacked histogram
    xMin, xMax = get_xrange(h_sig, 
                            strict=strict, 
                            xmin=xmin, 
                            xmax=xmax)

    # Get y-range values from all signal histograms
    y_ranges = np.array([get_yrange(h, 
                                    logY=logY, 
                                    ymin=ymin, 
                                    ymax=ymax) 
                                    for h in h_sigs])
    yMin = y_ranges[:,0].min()*scale_min
    yMax = y_ranges[:,1].max()*scale_max

    return xMin, xMax, yMin, yMax
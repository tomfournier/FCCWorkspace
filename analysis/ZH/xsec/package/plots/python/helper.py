'''Event flow and data processing helpers for cutflow analysis.

Provides:
- File I/O and metadata extraction: `find_sample_files()`, `get_processed()`, `is_there_events()`.
- Cached histogram and branch loading: `_key_from_file()`, `_cut_from_file()`, `_col_from_file()`.
- Event filtering and counting: `getcut()`, `get_cut()`, `get_count()`.
- Flow histogram construction: `get_flow()`, `get_flow_decay()`, `get_flows()`.
- Data persistence: `dump_json()`.
- Integration with uproot and ROOT for efficient file access.

Functions:
- `find_sample_files()`: Locate ROOT files for a specified sample or directory.
- `get_processed()`: Sum processed event counts across files using metadata keys.
- `get_cut()`: Sum cut values from events branch across files.
- `is_there_events()`: Check if ROOT files contain an 'events' tree.
- `getcut()`: Apply filter expression to dataframe with boolean mask combination.
- `get_count()`: Get event count using file metadata or dataframe filtering with fallback.
- `get_flow()`: Build event flow histograms and cutflow data for multiple processes.
- `get_flow_decay()`: Build event flow histograms for Higgs decay channels.
- `get_flows()`: Generate both process-level and decay-level event flows.
- `dump_json()`: Save event flow dictionary to JSON file.
- `_key_from_file()`: Extract integer value from ROOT file key (cached).
- `_cut_from_file()`: Retrieve cut value from events branch (cached).
- `_col_from_file()`: Extract column names from events tree (cached).

Conventions:
- All file metadata access cached via LRU cache (no size limit) to minimize I/O.
- Event counts represent cumulative passing events after applying cuts sequentially.
- Yield uncertainties computed as sqrt(N) on raw counts, then scaled by luminosity/cross-section.
- Histograms binned by cut step with bin N corresponding to cutN in cuts dictionary.
- Process flows accumulate yields across all samples in `processes[proc]` list.
- Decay flows accumulate yields across all Z decay channels for each Higgs decay mode.
- Sample names matching 'Hinv' skipped when aggregating flow data.
- JSON output includes 'cut' and 'err' dictionaries per process with cut names as keys.
- Optional 'hist' key contains ROOT.TH1 objects and can be excluded from JSON via flag.

Usage:
- Load and process events from ROOT files with automatic file discovery.
- Compute cumulative event counts across sequential cut selection steps.
- Build event flow histograms for cutflow analysis and efficiency calculations.
- Export event flow data to JSON for downstream analysis and bookkeeping.
- Validate data availability (events trees, cut branches) before processing.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ROOT
    import numpy as np
    import pandas as pd

import os, json, copy, uproot

from functools import lru_cache
from glob import glob

from ...tools.utils import mkdir



########################
### HELPER FUNCTIONS ###
########################

@lru_cache(maxsize=None)
def _key_from_file(
    filepath: str, 
    key: str
    ) -> int:
    '''Extract an integer value from a ROOT file key.
    
    Args:
        filepath (str): Path to the ROOT file.
        key (str): Key name to retrieve from the file.
    
    Returns:
        int: Integer value from the key, or 0 if not found.
    '''
    f = uproot.open(filepath)
    try:
        return int(f[key].value)
    except Exception:
        try:
            return int(f[key].array(entry_stop=1)[0])
        except Exception:
            print(f"WARNING: Couldn't find {key} in {filepath}, returning 0")
            return 0
        
@lru_cache(maxsize=None)
def _cut_from_file(
    filepath: str, 
    branch: str
    ) -> int:
    '''Retrieve cut value from events branch in a ROOT file.
    
    Args:
        filepath (str): Path to the ROOT file.
        branch (str): Branch name in the 'events' tree.
    
    Returns:
        int: Integer value from the branch, or 0 if not found.
    '''
    try:
        return int(uproot.open(filepath)['events'][branch].array(entry_stop=1)[0])
    except Exception:
        return 0
    
@lru_cache(maxsize=None)
def _col_from_file(
    filepath: str
    ) -> set[str]:
    '''Extract column names from the events tree in a ROOT file.
    
    Args:
        filepath (str): Path to the ROOT file.
    
    Returns:
        set[str]: Set of column names, or empty set if not found.
    '''
    try:
        f = uproot.open(filepath)
        if 'events' in f:
            return set(map(str, f['events'].keys()))
    except Exception:
        pass
    return set()



######################
### MAIN FUNCTIONS ###
######################

#_____________________
def find_sample_files(
    inDir: str, 
    sample: str
    ) -> list[str]:
    '''Locate ROOT files for a specified sample.
    
    Searches for .root files in a directory or accepts a single file path.
    
    Args:
        inDir (str): Input directory path.
        sample (str): Sample name (directory or file prefix).
    
    Returns:
        list[str]: List of absolute paths to ROOT files for the sample.
    '''
    result: list[str] = []
    full_input_path = os.path.abspath(os.path.join(inDir, sample))

    # Find all input files ending with .root
    if os.path.isdir(full_input_path):
        all_files = os.listdir(full_input_path)
        # Remove files not ending with `.root`
        all_files = [f for f in all_files if f.endswith('.root')]
        # Remove directories
        all_files = [f for f in all_files
                     if os.path.isfile(os.path.join(full_input_path, f))]
        result = [os.path.join(full_input_path, f) for f in all_files]

    # Handle case when there is just one input file
    if len(result) < 1:
        if os.path.isfile(full_input_path + '.root'):
            result.append(full_input_path + '.root')

    return result
        
#_______________________________
def get_processed(
    files: list[str] | str, 
    arg: str = 'eventsProcessed'
    ) -> int:
    '''Sum the number of processed events across files.
    
    Args:
        files (list[str] | str): Single file path or list of file paths.
        arg (str, optional): Key name to retrieve from files. Defaults to 'eventsProcessed'.
    
    Returns:
        int: Total count of processed events.
    '''
    if isinstance(files, str):
        files = [files]
    return int(sum(_key_from_file(os.fspath(f), arg) 
                   for f in files))

#____________________
def get_cut(
    files: list[str], 
    cut: str
    ) -> int:
    '''Sum cut values across files.
    
    Args:
        files (list[str]): Single file path or list of file paths.
        cut (str): Cut branch name in the events tree.
    
    Returns:
        int: Total count from the cut branch.
    '''
    if isinstance(files, str):
        files = [files]
    return int(sum(_cut_from_file(os.fspath(f), cut) 
                   for f in files))

#____________________________________
def getcut(
    df: 'pd.DataFrame', 
    filter: str,
    mask: 'np.ndarray' | None = None,
    ) -> tuple[int, 'np.ndarray']:
    '''Apply a filter to dataframe and combine with existing mask.
    
    Args:
        df (pd.DataFrame): Input dataframe to filter.
        filter (str): Filter expression to evaluate on dataframe.
        mask (np.ndarray | None, optional): Existing boolean mask to combine with filter (AND operation). Defaults to None.
    
    Returns:
        tuple: (count of rows matching filter, combined boolean mask).
    '''
    import numpy as np
    if df is None or df.empty:
        return 0, mask
    sel_mask = np.asarray(df.eval(filter), 
                          dtype=bool)
    if mask is None:
        mask = sel_mask
    else:
        mask &= sel_mask
    return int(mask.sum()), mask

#________________________________________
def get_count(
    df: 'pd.DataFrame',
    df_mask: 'np.ndarray' | None,
    file_list: list[str],
    cut_name: str,
    filter_expr: str,
    columns: set | None = None
    ) -> tuple[int, 'np.ndarray' | None]:
    '''Get event count for a cut using file metadata or dataframe filtering.
    
    Prefers reading cut values directly from files if available,
    otherwise applies filter expression to dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe containing event data.
        df_mask (np.ndarray | None): Existing boolean mask for events.
        file_list (list[str]): List of ROOT file paths.
        cut_name (str): Name of the cut to retrieve.
        filter_expr (str): Filter expression to apply if cut not in files.
        columns (set | None, optional): Set of available columns in the files. Defaults to None.
    
    Returns:
        tuple: (event count, updated boolean mask).
    '''
    if df is None or df.empty:
        return 0, df_mask

    if cut_name in columns:
        count = get_cut(file_list, cut_name)
        return count, df_mask

    try:
        count, df_mask = getcut(df, filter_expr, df_mask)
        return count, df_mask
    except Exception:
        print("WARNING: Couldn't compute the count for this cut")
        return 0, df_mask

#___________________
def is_there_events(
    proc: str, 
    path: str = '', 
    end='.root'
    ) -> bool:
    '''Check if a process has an events tree in ROOT files.
    
    Args:
        proc (str): Process name.
        path (str, optional): Directory path or empty string for current directory. Defaults to ''.
        end (str, optional): File extension. Defaults to '.root'.
    
    Returns:
        bool: True if all files contain an 'events' tree, False otherwise.
    
    Raises:
        SystemExit: If ROOT file not found for the process.
    '''
    fpath = os.path.join(path, proc+end)
    if os.path.exists(fpath):
        filename = fpath
        file = uproot.open(filename)
        return 'events' in file
    elif os.path.isdir(f'{path}/{proc}'):
        filenames = glob(f'{path}/{proc}/*')
        isTTree = []
        for i, filename in enumerate(filenames):
            file = uproot.open(filename)
            if 'events' in file:
                isTTree.append(i)
        return len(isTTree)==len(filenames)
    else: 
        print(f'ERROR: Could not find ROOT file for {proc}')
        quit()

#_________________________________
def dump_json(
    flow: dict[str, 
               dict[str, 
                    dict[str, 
                         float]]], 
    outDir: str, 
    outName: str, 
    hist: bool = False, 
    procs: list = []
    ) -> None:
    '''Save event flow dictionary to a JSON file.
    
    Args:
        flow (dict[str, dict[str, dict[str, float]]]): Dictionary containing event flow data.
        outDir (str): Output directory path.
        outName (str): Output file name (without extension).
        hist (bool, optional): If True, remove 'hist' key from each process. Defaults to False.
        procs (list, optional): List of process names to clean. Defaults to [].
    
    Returns:
        None
    '''
    dictio = copy.deepcopy(flow)
    if hist:
        for proc in procs:
            del dictio[proc]['hist']
    
    mkdir(outDir)
    with open(f'{outDir}/{outName}.json', 'w') as fOut:
        json.dump(dictio, fOut, indent=4)

#_______________________________________________________
def get_flow(
    events: dict[str, 
                 dict[str, 
                      float | int | dict[str, 
                                         float | str]]], 
    procs: list[str], 
    processes: dict[str, str], 
    cuts: dict[str, dict[str, str]], 
    cat: str, 
    sel: str, 
    tot: bool = False,
    json_file: bool = False, 
    loc: str = '', 
    outName: str = 'flow', 
    suffix: str = ''
    ) -> dict[str, 
              dict[str, 
                   'ROOT.TH1' | dict[str, 
                                     float]]]:
    '''Build event flow histograms and cutflow data for processes.
    
    Accumulates cut yields and errors across samples for each process,
    and optionally saves results to JSON.
    
    Args:
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data indexed by sample and selection.
        procs (list[str]): List of process names.
        processes (dict[str, str]): Mapping of process names to sample names.
        cuts (dict[str, dict[str, str]]): Dictionary of selections and their cut definitions.
        cat (str): Category label (e.g., 'ee' or 'mumu').
        sel (str): Selection name (e.g., 'pre-selection').
        tot (bool, optional): If True, combine all categories. Defaults to False.
        json_file (bool, optional): If True, save results to JSON. Defaults to False.
        loc (str, optional): Output directory for JSON files. Defaults to ''.
        outName (str, optional): Base name for JSON output file. Defaults to 'flow'.
        suffix (str, optional): Suffix to append to output file name. Defaults to ''.
    
    Returns:
        dict: Dictionary with process flow data including histograms, cuts, and errors.
    '''
    procs[0] = f'Z{cat}H' if not tot else 'ZH'
    _cat, _sel, _tot = f'_{cat}', f'_{sel}', '_tot' if tot else ''

    flow = {}
    for proc in procs:
        flow[proc] = {}
        # Initialize flow containers for process
        flow[proc]['hist'], flow[proc]['cut'], flow[proc]['err']  = [], {}, {}
        hist = ROOT.TH1D(proc+_cat+_sel+_tot, proc+_cat+_sel+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            flow[proc]['cut'][cut], flow[proc]['err'][cut]  = 0, 0
            for sample in processes[proc]:
                if 'Hinv' in sample: continue
                # Accumulate yields and errors from all samples
                flow[proc]['cut'][cut] += events[sample][sel]['cut'][cut]
                flow[proc]['err'][cut] += events[sample][sel]['err'][cut]**2
            
            # Fill histogram bin with accumulated count and error
            hist.SetBinContent(i+1, flow[proc]['cut'][cut])
            hist.SetBinError(i+1,   flow[proc]['err'][cut]**0.5)
        flow[proc]['hist'].append(hist)
    
    if json_file:
        dump_json(flow, loc, outName+_sel+suffix, hist=True, procs=procs)
    return flow

#_______________________________________________________
def get_flow_decay(
    events: dict[str,
                 dict[str,
                      float | int | dict[str,
                                         float | str]]], 
    z_decays: list[str], 
    h_decays: list[str], 
    cuts: dict[str, dict[str, str]], 
    cat: str, 
    sel: str, 
    outName: str = 'flow_decay',
    ecm: int = 240, 
    json_file: bool = False, 
    loc: str = '', 
    suffix: str = '', 
    tot: bool = False
    ) -> dict[str,
              dict[str,
                   'ROOT.TH1' | dict[str,
                                     float]]]:
    '''Build event flow histograms for Higgs decay channels.
    
    Accumulates cut yields for each Higgs decay mode across Z decay channels,
    and optionally saves results to JSON.
    
    Args:
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data indexed by sample and selection.
        z_decays (list[str]): List of Z boson decay modes (e.g., ['ee', 'mumu']).
        h_decays (list[str]): List of Higgs decay modes (e.g., ['bb', 'tautau']).
        cuts (dict[str, dict[str, str]]): Dictionary of selections and their cut definitions.
        cat (str): Category label (single decay mode).
        sel (str): Selection name (e.g., 'pre-selection').
        outName (str, optional): Base name for JSON output file. Defaults to 'flow_decay'.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        json_file (bool, optional): If True, save results to JSON. Defaults to False.
        loc (str, optional): Output directory for JSON files. Defaults to ''.
        suffix (str, optional): Suffix to append to output file name. Defaults to ''.
        tot (bool, optional): If True, combine all Z decay modes. Defaults to False.
    
    Returns:
        dict: Dictionary with Higgs decay flow data including histograms, cuts, and errors.
    '''
    import ROOT

    cats = z_decays if tot else [cat]
    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in cats] for y in h_decays]
    _cat, _sel, _tot = f'_{cat}', f'_{sel}', '_tot' if tot else ''

    flow = {}
    for h, sig in zip(h_decays, sigs):
        flow[h] = {}
        # Initialize flow containers for Higgs decay
        flow[h]['hist'], flow[h]['cut'], flow[h]['err'] = [], {}, {}
        hist = ROOT.TH1D('H'+h+_cat+_sel+_tot, 'H'+h+_cat+_sel+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            flow[h]['cut'][cut], flow[h]['err'][cut] = 0, 0
            # Accumulate yields across signal samples for this decay
            for s in sig:
                flow[h]['cut'][cut] += events[s][sel]['cut'][cut]
                flow[h]['err'][cut] += events[s][sel]['err'][cut]**2
            # Fill histogram bin with accumulated count and error
            hist.SetBinContent(i+1, flow[h]['cut'][cut])
            hist.SetBinError(i+1,   flow[h]['err'][cut]**0.5)
        flow[h]['hist'].append(hist)

    if json_file:
        dump_json(flow, loc, outName+_sel+suffix, hist=True, procs=h_decays)
    return flow

#_______________________________________________________
def get_flows(
    procs: list[str], 
    processes: dict[str, list[str]], 
    cuts: dict[str, dict[str, str]], 
    events: dict[str,
                 dict[str,
                      float | int | dict[str,
                                         float | str]]], 
    cat: str, 
    sel: str, 
    z_decays: list[str], 
    h_decays: list[str], 
    ecm: int = 240, 
    json_file: bool = False, 
    tot: bool = False, 
    loc_json: str = ''
    ) ->tuple[dict, dict]:
    '''Generate both process and decay-specific event flows.
    
    Wrapper function that computes event flow histograms for both
    individual processes and Higgs decay channels.
    
    Args:
        procs (list[str]): List of process names.
        processes (dict[str, list[str]]): Mapping of process names to sample names.
        cuts (dict[str, dict[str, str]]): Dictionary of selections and their cut definitions.
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data indexed by sample and selection.
        cat (str): Category label (e.g., 'ee' or 'mumu').
        sel (str): Selection name.
        z_decays (list[str]): List of Z boson decay modes.
        h_decays (list[str]): List of Higgs decay modes.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        json_file (bool, optional): If True, save results to JSON. Defaults to False.
        tot (bool, optional): If True, combine all categories. Defaults to False.
        loc_json (str, optional): Output directory for JSON files. Defaults to ''.
    
    Returns:
        tuple: (process flow dict, decay flow dict).
    '''
    suffix = '_tot' if tot else ''
    # Generate process-level flow
    flow = get_flow(
        events, procs, 
        processes, cuts, 
        cat, sel, tot=tot,
        json_file=json_file, 
        loc=loc_json, 
        suffix=suffix
    )
    
    # Generate decay-level flow
    flow_decay = get_flow_decay(
        events, z_decays, h_decays, 
        cuts, cat, sel, ecm=ecm, 
        json_file=json_file, 
        loc=loc_json, 
        suffix=suffix, tot=tot
    )
    return flow, flow_decay

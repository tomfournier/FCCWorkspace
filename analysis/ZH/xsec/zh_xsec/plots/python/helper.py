'''Event flow and data processing helpers for cutflow analysis.

Provides utilities for:
- File discovery and metadata extraction: `find_sample_files()`, `get_processed()`, `is_there_events()`.
- Cached histogram and column access: `_key_from_file()`, `_cut_from_file()`, `_col_from_file()`.
- Event filtering and counting: `getcut()`, `get_cut()`, `get_count()`.
- Event flow and cutflow histogram construction: `get_flow()`, `get_flow_decay()`, `get_flows()`.
- Data serialization: `dump_json()`.
- Efficient ROOT file access using uproot with LRU caching.

Functions:
- `find_sample_files(inDir, sample)`: Locate ROOT files for a sample within a directory.
- `get_processed(files, arg)`: Sum processed event counts from ROOT file metadata.
- `get_cut(files, cut_name)`: Sum event yields for a cut from pre-computed histograms.
- `is_there_events(proc, path, end)`: Check if ROOT files contain an 'events' tree.
- `getcut(df, filter, mask)`: Apply filter expression to dataframe with mask combination.
- `get_count(df, df_mask, file_list, cut_name, filter_expr)`: Get count from histogram or dataframe.
- `get_flow(events, procs, processes, cuts, ...)`: Build process-level event flow histograms.
- `get_flow_decay(events, z_decays, h_decays, cuts, ...)`: Build Higgs-decay-level flow histograms.
- `get_flows(procs, processes, cuts, ...)`: Generate both process and decay-level flows together.
- `dump_json(flow, outDir, outName, hist, procs)`: Serialize flow data to JSON file.
- `_key_from_file(filepath, key)`: Extract metadata value from ROOT file (cached).
- `_cut_from_file(filepath, cut_name)`: Retrieve cut bin content from histogram (cached).
- `_col_from_file(filepath)`: Extract column names from events tree (cached).
- `_get_bin_index(cut_name)`: Convert cut name string to ROOT histogram bin number.

Conventions and Data Structures:
- All file metadata and histogram access use LRU caching (no size limit) to minimize I/O.
- Cut histograms stored as ROOT.TH1D/TH1F/etc. at path: custom_objects/{HIST_TYPE}/cutFlow
- Cut naming: \"cut0\", \"cut1\", ... where ROOT bin 1 = cut0, bin 2 = cut1, etc.
- Event counts are cumulative (events passing up to and including that cut).
- Flow data structure: events[sample][selection][\"cut\"][cut_name] = int (count)
- Errors structure: events[sample][selection][\"err\"][cut_name] = float (uncertainty)
- Process flows accumulate yields from all samples in processes[proc_name] list.
- Decay flows accumulate yields from signal samples matching pattern: wzp6_ee_{Z_decay}H_H{Higgs_decay}_ecm{ECM}
- Samples with \"Hinv\" in name are skipped during aggregation.
- JSON output includes 'cut', 'err' dicts per process/decay. ROOT 'hist' objects excluded by default.

Usage Examples:
- Discover files: find_sample_files('./data', 'signal') -> list of .root files
- Get processed count: get_processed(files_list, 'eventsProcessed') -> int
- Build flow: get_flow(events_dict, procs, processes, cuts, 'ee', 'pre-selection', json_file=True)
- Export to JSON: dump_json(flow, './output', 'flow_ee', hist=True, procs=['ZeeH'])
'''

from __future__ import annotations

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

from tools.utils import mkdir
from logger import get_logger

LOGGER = get_logger(__name__)



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
            LOGGER.warning(f"Couldn't find {key} in {filepath}, returning 0")
            return 0

def _get_bin_index(cut_name: str) -> int:
    '''Extract bin index from cut name.

    Converts cut name (e.g., "cut0", "cut5") to histogram bin number.
    Bin numbering in ROOT: bin 1 = cut0, bin 2 = cut1, etc.

    Args:
        cut_name (str): Cut identifier (e.g., "cut0", "cut1").

    Returns:
        int: ROOT histogram bin number (1-indexed), or 0 if invalid.
    '''
    try:
        if cut_name.startswith('cut'):
            return int(cut_name[3:]) + 1  # "cut0" -> 1, "cut1" -> 2, etc.
    except (ValueError, IndexError):
        pass
    return 0


@lru_cache(maxsize=None)
def _cut_from_file(
    filepath: str,
    cut_name: str
     ) -> int:
    '''Retrieve cut value from histogram in a ROOT file.

    Reads cut yields from pre-computed histogram stored in custom_objects/{HIST_TYPE}/cutFlow.
    Tries all 1D histogram types (TH1D, TH1F, TH1I, TH1S, TH1C) in order of likelihood
    to maximize compatibility across different ROOT file formats. Uses LRU caching to
    minimize I/O overhead when called multiple times.

    Args:
        filepath (str): Path to the ROOT file.
        cut_name (str): Cut identifier (e.g., "cut0", "cut1", "cut2").

    Returns:
        int: Event count for the specified cut (bin content), or 0 if not found.
    '''
    try:
        f = uproot.open(filepath)
        # Get bin number from cut name
        bin_idx = _get_bin_index(cut_name)
        if bin_idx <= 0:
            return 0

        # Try all 1D histogram types in order of likelihood
        hist_types = ['TH1D', 'TH1F', 'TH1I', 'TH1S', 'TH1C']

        for hist_type in hist_types:
            hist_path = f'custom_objects/{hist_type}/cutFlow'
            try:
                hist = f[hist_path]
                # Read bin content (number of events passing the cut)
                return int(hist.values()[bin_idx - 1])
            except (KeyError, IndexError, TypeError):
                continue

        # No valid histogram found
        LOGGER.warning("Didn't find cutFlow in custom_objects, returning 0")
        return 0
    except Exception:
        return 0

@lru_cache(maxsize=None)
def _col_from_file(
    filepath: str
     ) -> set[str]:
    '''Extract column names from the events tree in a ROOT file.

    Reads the branch names from the 'events' tree in the specified ROOT file.
    Useful for validating that required columns are present before filtering.
    Results are cached to avoid repeated file I/O.

    Args:
        filepath (str): Path to the ROOT file.

    Returns:
        set[str]: Set of column/branch names from events tree, or empty set if not found.
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

# _____________________
def find_sample_files(
    inDir: str,
    sample: str
     ) -> list[str]:
    '''Locate ROOT files for a specified sample.

    Searches for .root files in a directory structure or returns a single file path.
    Automatically filters out non-ROOT files and directory entries. Handles both cases:
    - sample is a directory: returns all .root files within that directory
    - sample is a file prefix: returns [sample.root] if it exists

    Args:
        inDir (str): Input directory path to search within.
        sample (str): Sample name (directory name) or file prefix (without .root extension).

    Returns:
        list[str]: List of absolute paths to ROOT files for the sample (empty list if none found).
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

# _______________________________
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

# ____________________
def get_cut(
    files: list[str],
    cut_name: str
     ) -> int:
    '''Sum cut yields from histogram across multiple files.

    Aggregates event counts from pre-computed cutflow histograms across all
    input files for a specific cut step. If a single file path (string) is
    provided, it is converted to a list internally.

    Args:
        files (list[str]): List of ROOT file paths to aggregate.
        cut_name (str): Cut identifier (e.g., "cut0", "cut1", "cut2").

    Returns:
        int: Total event count passing the specified cut summed across all files.
    '''
    if isinstance(files, str):
        files = [files]
    return int(sum(_cut_from_file(os.fspath(f), cut_name)
                   for f in files))

def get_cuts(
    files: list[str],
    cut_names: list[str]
     ) -> dict[str, int]:
    '''Read several precomputed cut yields while opening each file once.'''
    if isinstance(files, str):
        files = [files]
    totals = {cut: 0 for cut in cut_names}
    hist_types = ['TH1D', 'TH1F', 'TH1I', 'TH1S', 'TH1C']
    for filepath in files:
        try:
            with uproot.open(os.fspath(filepath)) as root_file:
                histogram = None
                for hist_type in hist_types:
                    try:
                        histogram = root_file[f'custom_objects/{hist_type}/cutFlow']
                        break
                    except KeyError:
                        continue
                if histogram is None:
                    continue
                values = histogram.values()
                for cut in cut_names:
                    bin_index = _get_bin_index(cut)
                    if bin_index > 0 and bin_index <= len(values):
                        totals[cut] += int(values[bin_index - 1])
        except Exception:
            continue
    return totals

# ____________________________________
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
    # Evaluate filter expression as boolean mask on dataframe
    sel_mask = np.asarray(df.eval(filter),
                          dtype=bool)
    # Combine with existing mask using logical AND
    if mask is None:
        mask = sel_mask
    else:
        mask &= sel_mask
    # Return count of True values in combined mask
    return int(mask.sum()), mask

# ________________________________________
def get_count(
    df: 'pd.DataFrame' | None,
    df_mask: 'np.ndarray' | None,
    file_list: list[str],
    cut_name: str,
    filter_expr: str,
     ) -> tuple[int, 'np.ndarray' | None]:
    '''Get event count for a cut using histogram or dataframe filtering.

    Prioritizes reading pre-computed cut values from ROOT file histograms for efficiency.
    Falls back to applying filter expression to dataframe if histogram data unavailable
    or if filter_expr is explicitly provided. Returns empty results if dataframe is None
    or empty.

    Args:
        df (pd.DataFrame): Dataframe containing event data.
        df_mask (np.ndarray | None): Existing boolean mask for events to combine with filter.
        file_list (list[str]): List of ROOT file paths for histogram lookup.
        cut_name (str): Cut identifier (e.g., "cut0", "cut1").
        filter_expr (str): Filter expression to evaluate on dataframe; if empty string,
                          uses histogram lookup instead.

    Returns:
        tuple: (event count, updated boolean mask combining previous mask with new filter).
    '''

    # Try to read directly from pre-computed histogram in ROOT files
    if filter_expr == '':
        count = get_cut(file_list, cut_name)
        return count, df_mask

    if df is None or df.empty:
        return 0, df_mask

    # Fall back to dataframe filtering
    try:
        count, df_mask = getcut(df, filter_expr, df_mask)
        return count, df_mask
    except Exception:
        LOGGER.warning("Couldn't compute the count for this cut")
        return 0, df_mask

# ___________________
def is_there_events(
    proc: str,
    path: str = '',
    end='.root'
     ) -> bool:
    '''Check if a process has an events tree in ROOT files.

    Validates that event data is available for a given process. Handles both:
    - Single file: proc.root exists and contains 'events' tree
    - Directory: all .root files in proc/ directory contain 'events' tree

    Args:
        proc (str): Process name (file prefix or directory name).
        path (str, optional): Base directory path to search within. Defaults to ''.
        end (str, optional): File extension. Defaults to '.root'.

    Returns:
        bool: True if events tree exists in all files; False otherwise.

    Raises:
        SystemExit: Calls quit() if no ROOT files found for the process.
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
        LOGGER.error(f'Could not find ROOT file for {proc}')
        quit()


def dump_json(flow: dict, outDir: str, outName: str) -> None:
    '''Save event flow dictionary to a JSON file.

    Creates a deep copy of the flow dictionary and optionally removes the 'hist' key
    (containing non-serializable ROOT objects) before saving to JSON. Automatically
    creates output directory if it does not exist.

    Args:
        flow (dict[str, dict[str, dict[str, float]]]): Nested dictionary containing
               event flow data (typically from get_flow() or get_flow_decay()).
        outDir (str): Output directory path (created if necessary).
        outName (str): Output file name without extension (saves as outName.json).

    Returns:
        None
    '''
    dictio = copy.deepcopy(flow)

    mkdir(outDir)
    with open(f'{outDir}/{outName}.json', 'w') as fOut:
        json.dump(dictio, fOut, indent=4)


def get_flows(
    events: dict[str, dict[str, float | int | dict[str, float | str]]],
    plots: dict[str, dict[str, list[str]]],
    cuts: dict[str, dict[str, str]],
    sel: str,
    json_file: bool = False,
    tot: bool = False,
    loc_json: str = ''
     ) ->tuple[dict, dict]:
    '''Generate process-level and decay-level flows from one plot mapping.

    ``plots`` contains ``signals`` and ``backgrounds`` for the process flow and
    ``decays`` for the Higgs-decay flow. Each group maps an output label to its
    list of input samples. The returned tuple remains ``(flow, flow_decay)``.

    Args:
        plots (dict[str, dict[str, list[str]]]): Process and decay groups.
        cuts (dict[str, dict[str, str]]): Dictionary of selections with cut name lists.
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data
               indexed as events[sample_name][selection_name] containing 'cut' and 'err' dicts.
        sel (str): Selection name key (e.g., 'pre-selection', 'final-selection').
        json_file (bool, optional): If True, save both flows to JSON files. Defaults to False.
        tot (bool, optional): If True, combines categories with '_tot' suffix. Defaults to False.
        loc_json (str, optional): Output directory for JSON files. Defaults to ''.

    Returns:
        tuple: (flow_by_process, flow_by_decay) where each is a dict with process/decay
               names as keys and dicts containing 'hist', 'cut', 'err' keys.
    '''
    suffix = '_tot' if tot else ''
    cut_names = list(cuts[sel])

    def aggregate(groups: dict[str, list[str]], skip_hinv: bool = False) -> dict:
        flow = {}
        for label, samples in groups.items():
            flow[label] = {'cut': {}, 'err': {}}
            for cut in cut_names:
                yields = [events[sample][sel]['cut'][cut] for sample in samples
                          if not skip_hinv or 'Hinv' not in sample]
                errors = [events[sample][sel]['err'][cut] for sample in samples
                          if not skip_hinv or 'Hinv' not in sample]
                flow[label]['cut'][cut] = sum(yields)
                flow[label]['err'][cut] = sum(error**2 for error in errors)**0.5
        return flow

    groups = {label: samples for group in ('signals', 'backgrounds')
              for label, samples in plots.get(group, {}).items()}
    flow_procs = aggregate(groups, True)
    flow_decay = aggregate(plots.get('decays', {}), False)

    if json_file:
        mkdir(loc_json)
        dump_json(flow_procs, loc_json, f'flow_procs_{sel}{suffix}')
        dump_json(flow_decay, loc_json, f'flow_decay_{sel}{suffix}')
    return flow_procs, flow_decay

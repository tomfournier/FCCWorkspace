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

from ...tools.utils import mkdir
from ...logger import get_logger

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

# _________________________________
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

    Creates a deep copy of the flow dictionary and optionally removes the 'hist' key
    (containing non-serializable ROOT objects) before saving to JSON. Automatically
    creates output directory if it does not exist.

    Args:
        flow (dict[str, dict[str, dict[str, float]]]): Nested dictionary containing
               event flow data (typically from get_flow() or get_flow_decay()).
        outDir (str): Output directory path (created if necessary).
        outName (str): Output file name without extension (saves as outName.json).
        hist (bool, optional): If True, removes 'hist' key from each process to
               exclude non-JSON-serializable ROOT objects. Defaults to False.
        procs (list, optional): List of process names (strings) corresponding to
               keys in flow dict. Only processed if hist=True. Defaults to [].

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

# _______________________________________________________
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
    suffix: str = '',
    save_hist: bool = False
     ) -> dict[str,
               dict[str,
                    'ROOT.TH1' | dict[str,
                                      float]]]:
    '''Build event flow histograms and cutflow data aggregated by process.

    Accumulates cut yields and errors across all samples for each physics process.
    Creates ROOT TH1D histograms with bin contents set to cumulative event counts and
    bin errors from error propagation (sqrt of summed squared errors). Optionally saves
    results to JSON file (excluding ROOT objects). Modifies procs[0] to match naming
    convention (e.g., 'ZeeH' or 'ZH' if tot=True).

    Args:
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data
               indexed as events[sample_name][selection_name] containing 'cut' and 'err' dicts.
        procs (list[str]): List of process names (modified in-place to set procs[0]).
        processes (dict[str, str]): Mapping of process names to list of sample names.
        cuts (dict[str, dict[str, str]]): Dictionary of selections with cut name lists.
        cat (str): Category label for naming (e.g., 'ee', 'mumu').
        sel (str): Selection name key (e.g., 'pre-selection', 'final-selection').
        tot (bool, optional): If True, set procs[0]='ZH'; otherwise 'Z{cat}H'. Defaults to False.
        json_file (bool, optional): If True, save results to JSON file. Defaults to False.
        loc (str, optional): Output directory for JSON files. Defaults to ''.
        outName (str, optional): Base name for JSON output file. Defaults to 'flow'.
        suffix (str, optional): Suffix appended to JSON filename. Defaults to ''.

    Returns:
        dict[str, dict]: Dictionary with keys for each process containing:
            - 'hist': list with one ROOT.TH1D histogram per process
            - 'cut': dict mapping cut names to cumulative event counts
            - 'err': dict mapping cut names to error values (sqrt of summed squared errors)
    '''
    import ROOT

    # Rename first process to follow naming convention: ZeeH, ZmumuH, etc.
    procs[0] = f'Z{cat}H' if not tot else 'ZH'
    _cat, _sel, _tot = f'_{cat}', f'_{sel}', '_tot' if tot else ''

    flow = {}
    for proc in procs:
        flow[proc] = {}
        # Initialize containers: 'hist' (list of ROOT histograms), 'cut' and 'err' (dicts by cut name)
        flow[proc]['hist'], flow[proc]['cut'], flow[proc]['err']  = [], {}, {}
        # Create ROOT histogram with number of bins equal to number of cuts
        hist = ROOT.TH1D(proc+_cat+_sel+_tot, proc+_cat+_sel+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            # Initialize count and error for this cut
            flow[proc]['cut'][cut], flow[proc]['err'][cut]  = 0, 0
            # Sum yields from all samples for this process (skip 'Hinv' samples)
            for sample in processes[proc]:
                if 'Hinv' in sample: continue
                # Accumulate yields and errors from all samples
                flow[proc]['cut'][cut] += events[sample][sel]['cut'][cut]
                flow[proc]['err'][cut] += events[sample][sel]['err'][cut]**2

            # Fill histogram bin with accumulated count and propagated error (sqrt of sum of squares)
            hist.SetBinContent(i+1, flow[proc]['cut'][cut])
            hist.SetBinError(i+1,   flow[proc]['err'][cut]**0.5)
        flow[proc]['hist'].append(hist)

    if json_file:
        mkdir(loc)
        dump_json(flow, loc, outName+_sel+suffix, hist=True, procs=procs)

    if save_hist:
        mkdir(loc)
        with ROOT.TFile(f"{loc}/{outName}{_sel}{suffix}.root", 'RECREATE'):
            for proc in procs:
                h = flow[proc]['hist'][0]
                h.Write()
    return flow

# _______________________________________________________
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
     ) -> 'dict[str, dict[str, ROOT.TH1 | dict[str, float]]]':
    '''Build event flow histograms aggregated by Higgs decay channel.

    Accumulates cut yields for each Higgs decay mode (bb, tautau, etc.) across Z decay
    channels specified in z_decays list. Constructs signal sample names from decay mode
    combinations (e.g., 'wzp6_ee_eeH_Hbb_ecm240'). Creates ROOT TH1D histograms with
    error propagation. Optionally saves results to JSON file.

    Args:
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data
               indexed as events[sample_name][selection_name] containing 'cut' and 'err' dicts.
        z_decays (list[str]): List of Z decay modes (e.g., ['ee', 'mumu', 'tautau']).
        h_decays (list[str]): List of Higgs decay modes (e.g., ['bb', 'tautau', 'cc']).
        cuts (dict[str, dict[str, str]]): Dictionary of selections with cut name lists.
        cat (str): Category label for naming (typically single Z decay mode).
        sel (str): Selection name key (e.g., 'pre-selection', 'final-selection').
        outName (str, optional): Base name for JSON output file. Defaults to 'flow_decay'.
        ecm (int, optional): Center-of-mass energy in GeV (used in sample naming). Defaults to 240.
        json_file (bool, optional): If True, save results to JSON file. Defaults to False.
        loc (str, optional): Output directory for JSON files. Defaults to ''.
        suffix (str, optional): Suffix appended to JSON filename. Defaults to ''.
        tot (bool, optional): If True, accumulate across all z_decays; otherwise use [cat]. Defaults to False.

    Returns:
        dict[str, dict]: Dictionary with keys for each Higgs decay mode containing:
            - 'hist': list with one ROOT.TH1D histogram per Higgs decay
            - 'cut': dict mapping cut names to cumulative event counts
            - 'err': dict mapping cut names to error values (sqrt of summed squared errors)
    '''
    import ROOT

    # Determine which Z decay modes to include in flow
    cats = z_decays if tot else [cat]
    # Build signal sample names from decay mode combinations
    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in cats] for y in h_decays]
    _cat, _sel, _tot = f'_{cat}', f'_{sel}', '_tot' if tot else ''

    flow = {}
    for h, sig in zip(h_decays, sigs):
        flow[h] = {}
        # Initialize containers: 'hist' (list of ROOT histograms), 'cut' and 'err' (dicts by cut name)
        flow[h]['hist'], flow[h]['cut'], flow[h]['err'] = [], {}, {}
        # Create ROOT histogram with number of bins equal to number of cuts
        hist = ROOT.TH1D('H'+h+_cat+_sel+_tot, 'H'+h+_cat+_sel+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            # Initialize count and error for this Higgs decay mode
            flow[h]['cut'][cut], flow[h]['err'][cut] = 0, 0
            # Accumulate yields from all signal samples for this specific Higgs decay
            for s in sig:
                flow[h]['cut'][cut] += events[s][sel]['cut'][cut]
                flow[h]['err'][cut] += events[s][sel]['err'][cut]**2
            # Fill histogram bin with accumulated count and propagated error (sqrt of sum of squares)
            hist.SetBinContent(i+1, flow[h]['cut'][cut])
            hist.SetBinError(i+1,   flow[h]['err'][cut]**0.5)
        flow[h]['hist'].append(hist)

    if json_file:
        dump_json(flow, loc, outName+_sel+suffix, hist=True, procs=h_decays)
    return flow

# _______________________________________________________
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
    '''Generate both process-level and Higgs-decay-level event flows.

    Convenience wrapper that computes two complementary views of event flow data:
    1. Flows aggregated by physics process (using get_flow())
    2. Flows aggregated by Higgs decay channel (using get_flow_decay())

    Both are created with identical selection and naming parameters to ensure consistency.
    Optionally saves both flows to JSON files with '_tot' suffix if tot=True.

    Args:
        procs (list[str]): List of process names (modified in-place by get_flow).
        processes (dict[str, list[str]]): Mapping of process names to lists of sample names.
        cuts (dict[str, dict[str, str]]): Dictionary of selections with cut name lists.
        events (dict[str, dict[str, float | int | dict[str, float | str]]]): Event data
               indexed as events[sample_name][selection_name] containing 'cut' and 'err' dicts.
        cat (str): Category label for naming (e.g., 'ee', 'mumu').
        sel (str): Selection name key (e.g., 'pre-selection', 'final-selection').
        z_decays (list[str]): List of Z decay modes for decay-level flow.
        h_decays (list[str]): List of Higgs decay modes for decay-level flow.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        json_file (bool, optional): If True, save both flows to JSON files. Defaults to False.
        tot (bool, optional): If True, combines categories with '_tot' suffix. Defaults to False.
        loc_json (str, optional): Output directory for JSON files. Defaults to ''.

    Returns:
        tuple: (flow_by_process, flow_by_decay) where each is a dict with process/decay
               names as keys and dicts containing 'hist', 'cut', 'err' keys.
    '''
    suffix = '_tot' if tot else ''
    # Generate process-level flow (aggregated across samples per physics process)
    flow = get_flow(
        events, procs,
        processes, cuts,
        cat, sel, tot=tot,
        json_file=json_file,
        loc=loc_json,
        suffix=suffix
    )

    # Generate decay-level flow (aggregated across Z decays per Higgs decay channel)
    flow_decay = get_flow_decay(
        events, z_decays, h_decays,
        cuts, cat, sel, ecm=ecm,
        json_file=json_file,
        loc=loc_json,
        suffix=suffix, tot=tot
    )
    return flow, flow_decay

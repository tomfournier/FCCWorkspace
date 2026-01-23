'''Process management and histogram operations.

Provides:
- Process dictionary retrieval with caching: `_get_procDict()`, `getMetaInfo()`.
- Histogram I/O from ROOT files: `get_hist()`, `getHist()`.
- Histogram combination and post-processing: `concat()`, `proc_scale()`.

Conventions:
- Process dictionaries are cached in `PROCDICT_CACHE` to avoid repeated reads.
- WW cross-section corrections: For `p8_ee_WW_ecm*` processes, leptonic decays
  (ee, mumu) are optionally removed when `remove=True` or automatically during
  histogram retrieval via `get_hist()` and `getHist()`.
- Histogram I/O: Files are `{proc}{suffix}.root` under `inDir`; histograms are
  detached from TFile with `SetDirectory(0)` to prevent cleanup issues.
- Lazy mode (`lazy=True`): Missing files/histograms are skipped silently; when
  `False`, warnings are raised via `config.warning()`.
- `concat()` unrolls multiple 1D histograms into a single concatenated histogram.
- `proc_scale()` applies process-specific scaling from `proc_scales` dict using
  keys matched in `processes`.

Lazy Imports:
- ROOT is lazy-loaded only when histograms are accessed
- numpy and tqdm are lazy-loaded in functions where needed
'''

import os
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import ROOT


from .utils import get_procDict
from ..config import warning


# Cache for process dictionaries to avoid repeated file I/O
PROCDICT_CACHE = {}

# Cache for cross-section/meta info (includes rmww variations)
# Key: (proc, info, rmww, procFile, fcc)
XSEC_CACHE = {}

# Cache for WW rescaling factors (xsec_new/xsec_old)
# Key: proc name string
WW_SCALE_CACHE = {}

# Cache for histograms to avoid repeated file opening
# Structure: {(proc, suffix, inDir): {hName: ROOT.TH1}}
HIST_CACHE = {}

def _get_procDict(procFile: str = 'FCCee_procDict_winter2023_IDEA.json',
                  fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts') -> dict:
    '''
    Retrieve and cache process dictionary from file.
    
    Args:
        procFile (str, optional): Name of the process dictionary file. Defaults to 'FCCee_procDict_winter2023_IDEA.json'.
        fcc (str, optional): Path to the FCC directory containing the dictionary. Defaults to '/cvmfs/fcc.cern.ch/FCCDicts'.
    
    Returns:
        dict: Process dictionary with metadata (cross-sections, decay channels, etc.).
    '''
    cache_key = (procFile, fcc)
    if cache_key not in PROCDICT_CACHE:
        PROCDICT_CACHE[cache_key] = get_procDict(procFile, fcc=fcc)
    return PROCDICT_CACHE[cache_key]


def preload_histograms(procs: list[str],
                       inDir: str,
                       suffix: str = '',
                       hNames: list[str] = None,
                       rebin: int = 1,
                       rmww: bool = True) -> None:
    '''
    Preload all histograms from files into memory cache.
    
    Opens each ROOT file once and caches all histograms (or specified ones)
    to avoid repeated file I/O during plotting loops.
    
    Args:
        procs (list[str]): List of process names to preload.
        inDir (str): Input directory path.
        suffix (str, optional): File suffix. Defaults to ''.
        hist_names (list[str], optional): List of histogram names to load. If None, loads all. Defaults to None.
        rebin (int, optional): Rebinning factor to apply. Defaults to 1.
        rmww (bool, optional): Apply WW cross-section correction. Defaults to True.
    '''
    import ROOT
    from tqdm import tqdm
    
    print('\n----->[Info] Preloading histograms into cache...')

    # Pre-cache WW scales (use global cache to persist across calls)
    ww_scales_cache = WW_SCALE_CACHE
    
    for proc in tqdm(procs):
        fInName = os.path.join(inDir, f'{proc}{suffix}.root')
        cache_key = (proc, suffix, inDir)
        
        # Skip if already cached
        if cache_key in HIST_CACHE:
            continue
            
        # Check file existence
        if not os.path.exists(fInName):
            continue
        
        # Open ROOT file
        f = ROOT.TFile.Open(fInName)
        if not f or f.IsZombie():
            if f:
                f.Close()
            continue
        
        # Initialize cache for this file
        HIST_CACHE[cache_key] = {}
        
        # Compute WW scale once per process
        scale = 1.0
        if rmww and 'p8_ee_WW_ecm' in proc:
            if proc not in ww_scales_cache:
                xsec_old = getMetaInfo(proc, rmww=False)
                if xsec_old != 0:
                    xsec_new = getMetaInfo(proc, rmww=True)
                    ww_scales_cache[proc] = xsec_new / xsec_old
                else:
                    ww_scales_cache[proc] = 1.0
            scale = ww_scales_cache[proc]
        
        # Load histograms
        if hNames is None:
            # Load all histograms in file
            key_list = f.GetListOfKeys()
            hist_names_to_load = [
                key.GetName() for key in key_list 
                if key.GetClassName().startswith('TH')
            ]
        else:
            hist_names_to_load = hNames
        
        for hName in hist_names_to_load:
            h = f.Get(hName)
            if h is None:
                continue
            
            # Clone and detach from file
            h_clone = h.Clone(f'{proc}_{hName}')
            h_clone.SetDirectory(0)
            
            # Apply rebinning
            if rebin != 1:
                h_clone.Rebin(rebin)
            
            # Apply WW scale
            if scale != 1.0:
                h_clone.Scale(scale)
            
            HIST_CACHE[cache_key][hName] = h_clone
        
        f.Close()
    
    print(f'----->[Info] Preloading complete. Cached {len(HIST_CACHE)} files.\n')


def clear_histogram_cache() -> None:
    '''
    Clear the histogram cache to free memory.
    Also clears WW scale cache to ensure consistent re-scaling on next run.
    Retrocompatible: signature and primary behavior unchanged.
    '''
    global HIST_CACHE, WW_SCALE_CACHE
    HIST_CACHE.clear()
    WW_SCALE_CACHE.clear()
    print('----->[Info] Histogram cache cleared.')


#____________________________________________________________________
def getMetaInfo(proc: str, 
                info: str = 'crossSection', 
                rmww: bool = False,
                fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts', 
                procFile: str = 'FCCee_procDict_winter2023_IDEA.json'
                ) -> float:
    '''
    Retrieve metadata information for a process from the process dictionary.
    
    Args:
        proc (str): Process name identifier.
        info (str, optional): Type of metadata to retrieve. Defaults to 'crossSection'.
        rmww (bool, optional): If True, remove leptonic WW decays from WW cross-section. Defaults to False.
        fcc (str, optional): Path to the FCC directory. Defaults to '/cvmfs/fcc.cern.ch/FCCDicts'.
        procFile (str, optional): Process dictionary file name. Defaults to 'FCCee_procDict_winter2023_IDEA.json'.
    
    Returns:
        float: Requested metadata value, or None if process not found.
    '''
    # Check cache first (includes rmww flag and procFile/fcc context)
    cache_key = (proc, info, rmww, procFile, fcc)
    if cache_key in XSEC_CACHE:
        return XSEC_CACHE[cache_key]

    procDict = _get_procDict(procFile, fcc=fcc)

    if proc not in procDict:
        return None

    val = procDict[proc][info]
    # Remove ee and mumu decay channels for WW processes
    if rmww and 'p8_ee_WW_ecm' in proc and info == 'crossSection':
        xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'), info=info, rmww=False, fcc=fcc, procFile=procFile)
        xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'), info=info, rmww=False, fcc=fcc, procFile=procFile)
        xsec_tot  = getMetaInfo(proc, info=info, rmww=False, fcc=fcc, procFile=procFile)
        val       = xsec_tot - xsec_ee - xsec_mumu

    XSEC_CACHE[cache_key] = val
    return val

#______________________________________________
def get_hist(hName: str, 
             proc: str, 
             processes: dict[str, list[str]], 
             inDir: str, 
             suffix: str = '',
             rebin: int = 1,
             proc_scales: dict[str, float] = {}
             ) -> 'ROOT.TH1':
    '''
    Retrieve a histogram for a single process from ROOT file.
    
    Applies cross-section corrections for WW processes and process-specific scales.
    
    Args:
        hName (str): Name of the histogram to retrieve.
        proc (str): Process name identifier.
        processes (dict[str, list[str]]): Process configuration mapping process names to process lists.
        inDir (str): Input directory path.
        suffix (str, optional): File suffix to append. Defaults to ''.
        proc_scales (dict[str, float], optional): Process scaling factors. Defaults to {}.
    
    Returns:
        ROOT.TH1: Scaled histogram, or None if retrieval fails.
    '''
    import ROOT

    fpath = os.path.join(inDir, f'{proc}{suffix}.root')
    print(f'----->[Info] Getting {hName} from \n\t {fpath}')
    
    # Verify input file exists
    if not os.path.exists(fpath):
        warning(f'Input file not found: {fpath}')
        return None
    
    # Open ROOT file and check validity
    f = ROOT.TFile.Open(fpath)
    if not f or f.IsZombie():
        if f:
            f.Close()
        return None
    
    # Retrieve histogram and check if found
    h = f.Get(hName)
    if h is None:
        warning(f'Histogram {hName} not found in {fpath}')
        f.Close()
        return None
    
    # Detach histogram from file to prevent cleanup issues
    h.SetDirectory(0)

    # Apply WW cross-section correction if needed
    scale, xsec = 1.0, getMetaInfo(proc, rmww=False)
    if xsec!=0 and 'p8_ee_WW_ecm' in proc:
        print(f'----->[Info] Rescaling {proc} sample to account for leptonic decay removing')
        xsec_new = getMetaInfo(proc, rmww=True)
        scale = xsec_new / xsec

    if rebin!=1:
        h.Rebin(rebin)

    if scale!=1.0:
        h.Scale(scale)

    f.Close()

    h = proc_scale(h, proc, processes, proc_scales=proc_scales)
    return h

#__________________________________
def getHist(hName: str, 
            procs: str, 
            inDir: str, 
            suffix: str = '', 
            rebin: int = 1, 
            lazy: bool = True, 
            proc_scale: float = 1.,
            rmww: bool = True,
            use_cache: bool = True,
            ) -> 'ROOT.TH1':
    '''
    Retrieve and sum histograms across multiple processes.
    
    Handles missing files gracefully based on lazy mode and applies cross-section
    corrections, rebinning, and scaling. Checks cache first if use_cache=True.
    
    Args:
        hName (str): Name of the histogram to retrieve.
        procs (str): List of process names to combine.
        inDir (str): Input directory path.
        suffix (str, optional): File suffix. Defaults to ''.
        rebin (int, optional): Rebinning factor. Defaults to 1.
        lazy (bool, optional): If True, skip missing files; if False, raise warnings. Defaults to True.
        proc_scale (float, optional): Global scaling factor for the histogram. Defaults to 1.0.
        rmww (bool, optional): Apply WW cross-section correction. Defaults to True.
        use_cache (bool, optional): Check cache before opening files. Defaults to True.
    
    Returns:
        ROOT.TH1: Combined and scaled histogram, or None if no histograms found.
    '''
    import ROOT 

    hist = None
    names, where = [], []
    
    # Pre-cache WW metadata for all processes to avoid repeated lookups
    ww_scales_cache = {}

    for proc in procs:
        cache_key = (proc, suffix, inDir)
        
        # Check cache first if enabled
        if use_cache and cache_key in HIST_CACHE:
            if hName in HIST_CACHE[cache_key]:
                # Clone from cache to avoid modifying cached histogram
                h = HIST_CACHE[cache_key][hName].Clone(f'{proc}_{hName}_copy')
                h.SetDirectory(0)
                
                # Accumulate
                if hist is None:
                    hist = h
                else:
                    hist.Add(h)
                continue
            elif not lazy:
                warning(f'Histogram {hName} not found in cache for {proc}')
                continue
        fInName = os.path.join(inDir, f'{proc}{suffix}.root')

        # Check file existence
        if not os.path.exists(fInName):
            if lazy:
                names.append(fInName)
                where.append('File existence')
                continue
            else:
                warning(f'ERROR: cannot open input file {fInName}')

        # Open ROOT file
        f = ROOT.TFile.Open(fInName)
        if not f or f.IsZombie():
            if f:
                f.Close()
            if lazy:
                names.append(fInName)
                where.append('File opening')
                continue
            else:
                warning(f'ERROR: cannot open input file {fInName}')
                exit(1)

        # Retrieve histogram
        h = f.Get(hName)
        if h is None:
            f.Close()
            if lazy:
                names.append(fInName)
                where.append('Histogram retrieval')
                continue
            else:
                warning(f'Histogram {hName} not found in {fInName}')
                return None
        
        # Detach histogram from file
        h.SetDirectory(0)

        # Apply WW cross-section correction with cached lookups
        scale = 1.0
        if rmww and 'p8_ee_WW_ecm' in proc:
            # Use cached scale if available, otherwise compute and cache
            if proc not in ww_scales_cache:
                xsec_old = getMetaInfo(proc, rmww=False)
                if xsec_old != 0:
                    xsec_new = getMetaInfo(proc, rmww=True)
                    ww_scales_cache[proc] = xsec_new / xsec_old
                else:
                    ww_scales_cache[proc] = 1.0
            scale = ww_scales_cache[proc]

        if scale != 1.0:
            h.Scale(scale)

        # Accumulate histograms
        if hist is None: 
            hist = h
        else: 
            hist.Add(h)

        f.Close()

    if hist is None:
        msgs = [n+' | at step '+w for n, w in zip(names, where)]
        print("----->[WARNING] Couldn't find histograms")
        print("----->[WARNING] For processes\n\t"+'\n\t'.join(msgs))
        print('----->[WARNING] Returning None')
        return None

    # Apply post-processing: rebinning and scaling
    if rebin!=1:
        hist.Rebin(rebin)
    
    if proc_scale!=1:
        hist.Scale(proc_scale)
    return hist

#_________________________________
def concat(h_list: list['ROOT.TH1'], 
           hName: str, 
           outName: str = ''
           ) -> 'ROOT.TH1':
    '''
    Concatenate multiple histograms into a single unrolled 1D histogram.
    
    Args:
        h_list (list[ROOT.TH1]): List of ROOT histograms to concatenate.
        hName (str): Name for logging purposes.
        outName (str, optional): Name for the output histogram. Defaults to '' (uses hName if empty).
    
    Returns:
        ROOT.TH1: Unrolled 1D histogram combining all input histograms.
    '''
    import ROOT

    print(f'----->[Info] Concatenating {hName}')

    # Calculate total number of bins
    tot_bins = sum([h.GetNbinsX() for h in h_list])

    # Create unrolled 1D histogram
    h_concat = ROOT.TH1D('h1', '1D Unrolled Histogram', 
                         tot_bins, 0.5, tot_bins + 0.5)
    h_concat.SetDirectory(0)

    # Copy bin contents and errors from input histograms
    bin_offset = 0
    for hist in h_list:
        nbins = hist.GetNbinsX()

        for bin_idx in range(1, nbins + 1):
            h_concat.SetBinContent(bin_offset+bin_idx, hist.GetBinContent(bin_idx))
            h_concat.SetBinError(bin_offset+bin_idx, hist.GetBinError(bin_idx))

        bin_offset += nbins

    h_concat.SetName(outName if outName else hName)
            
    return h_concat

#________________________________________________
def proc_scale(hist: 'ROOT.TH1', 
               proc: str, 
               processes: dict[str, list[str]], 
               proc_scales: dict[str, float] = {}
               ) -> 'ROOT.TH1':
    '''
    Apply process-specific scaling factor to a histogram.
    
    Looks up the process in the configuration and applies the corresponding scale.
    
    Args:
        hist (ROOT.TH1): ROOT histogram to scale.
        proc (str): Process name identifier.
        processes (dict[str, list[str]]): Process configuration mapping process names to process lists.
        proc_scales (dict[str, float], optional): Dictionary of scaling factors by process name. Defaults to {}.
    
    Returns:
        ROOT.TH1: Scaled histogram.
    '''
    import ROOT

    if not proc_scales:
        return hist
    
    # Find process category and apply corresponding scale
    for proc_name, proc_list in processes.items():
        if proc in proc_list and proc_name!='Rare':
            scale = proc_scales.get(proc_name)
            if scale is not None:
                hist.Scale(scale)
                print(f'----->[Info] Scaled histogram '
                    f'to ILC scale by a factor of {scale:.3f}')
            break
    return hist

#__________________________________
def get_stack(hists: list['ROOT.TH1']
              ) -> 'ROOT.TH1':
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
def get_xrange(hist: 'ROOT.TH1', 
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
    import numpy as np
    nbins = hist.GetNbinsX()
    
    # Direct numpy array from ROOT (excludes under/overflow bins)
    contents = np.array(hist, dtype=np.float64)[1:nbins+1]
    
    # Get axis object once
    xaxis = hist.GetXaxis()
    
    # Check if variable bin width
    if xaxis.IsVariableBinSize():
        # Variable bins: extract edges individually but efficiently
        edges = np.array([xaxis.GetBinLowEdge(i+1) for i in range(nbins+1)], dtype=np.float64)
    else:
        # Fixed bins: use linspace for maximum speed
        edges = np.linspace(xaxis.GetXmin(), xaxis.GetXmax(), nbins + 1, dtype=np.float64)
    
    # Vectorized filtering
    mask = np.ones(nbins, dtype=bool)
    if strict:
        mask &= (contents != 0)
    if xmin is not None:
        mask &= (edges[:-1] > xmin)
    if xmax is not None:
        mask &= (edges[1:] < xmax)
    
    if not mask.any():
        # Fallback to full range if nothing matched
        return float(edges[0]), float(edges[-1])
    
    # Find min/max from masked edges
    valid_bins = np.where(mask)[0]
    return float(edges[valid_bins[0]]), float(edges[valid_bins[-1] + 1])

#___________________________________________________
def get_yrange(hist: 'ROOT.TH1', 
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
    import numpy as np
    nbins = hist.GetNbinsX()
    
    # Direct numpy array extraction (excludes under/overflow bins)
    contents = np.array(hist, dtype=np.float64)[1:nbins+1]

    if logY:
        # Handle log scale: exclude zero and negative values
        nonzero = contents[contents > 0]
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
def get_range(h_sigs: list['ROOT.TH1'], 
              h_bkgs: list['ROOT.TH1'],
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
    import numpy as np
    # Stack all signal and background histograms
    h_stack = get_stack(h_sigs + h_bkgs)

    # Determine x-range from stacked histogram
    xMin, xMax = get_xrange(
        h_stack, strict=strict, 
        xmin=xmin, xmax=xmax
    )
    h_bkg = [get_stack(h_bkgs)]
    
    # Get y-min values from all histograms
    yMin = np.array([
        get_yrange(
            h, logY=logY,
            ymin=ymin, ymax=ymax,
            scale_min=scale_min,
            scale_max=scale_max)[0] 
            for h in h_sigs + h_bkgs
    ])
    
    # Get y-max values based on stacking option
    if stack:
        yMax = get_yrange(
            h_stack, logY=logY, 
            ymin=ymin, ymax=ymax, 
            scale_min=scale_min,
            scale_max=scale_max
        )[1]
    else:
        yMax = np.array([
            get_yrange(
                h, logY=logY, 
                ymin=ymin, ymax=ymax,
                scale_min=scale_min,
                scale_max=scale_max)[1] 
                for h in h_sigs + h_bkg
        ])
        
    yMin = yMin.min()
    yMax = yMax.max()
    return xMin, xMax, yMin, yMax

#________________________________________________________
def get_range_decay(h_sigs: list['ROOT.TH1'], 
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
    import numpy as np
    # Stack signal histograms
    h_sig = get_stack(h_sigs)
    
    # Get x-range from stacked histogram
    xMin, xMax = get_xrange(
        h_sig, strict=strict, 
        xmin=xmin, xmax=xmax
        )

    # Get y-range values from all signal histograms
    y_ranges = np.array([get_yrange(
        h, logY=logY, ymin=ymin, ymax=ymax
        ) for h in h_sigs])
    yMin = y_ranges[:,0].min()*scale_min
    yMax = y_ranges[:,1].max()*scale_max

    return xMin, xMax, yMin, yMax

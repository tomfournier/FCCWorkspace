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
'''

import os, ROOT

from ..config import warning
from .utils import get_procDict

# Cache for process dictionaries to avoid repeated file I/O
PROCDICT_CACHE = {}

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


#____________________________________________________________________
def getMetaInfo(proc: str, 
                info: str = 'crossSection', 
                remove: bool = False,
                fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts', 
                procFile: str = 'FCCee_procDict_winter2023_IDEA.json'
                ) -> float:
    '''
    Retrieve metadata information for a process from the process dictionary.
    
    Args:
        proc (str): Process name identifier.
        info (str, optional): Type of metadata to retrieve. Defaults to 'crossSection'.
        remove (bool, optional): If True, remove leptonic WW decays from WW cross-section. Defaults to False.
        fcc (str, optional): Path to the FCC directory. Defaults to '/cvmfs/fcc.cern.ch/FCCDicts'.
        procFile (str, optional): Process dictionary file name. Defaults to 'FCCee_procDict_winter2023_IDEA.json'.
    
    Returns:
        float: Requested metadata value, or None if process not found.
    '''
    procDict = _get_procDict(procFile, fcc=fcc)

    if proc not in procDict:
        return None

    xsec = procDict[proc][info]
    # Remove ee and mumu decay channels for WW processes
    if remove and 'p8_ee_WW_ecm' in proc:
        xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'))
        xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'))
        xsec      = getMetaInfo(proc) - xsec_ee - xsec_mumu
    return xsec

#______________________________________________
def get_hist(hName: str, 
             proc: str, 
             processes: dict[str, list[str]], 
             inDir: str, 
             suffix: str = '',
             proc_scales: dict[str, float] = {}
             ) -> ROOT.TH1:
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
    fpath = os.path.join(inDir, f'{proc}{suffix}.root')
    print(f'----->[Info] Getting histogram from \n\t {fpath}')
    
    # Verify input file exists
    if not os.path.exists(fpath):
        msg = f'Input file not found: {fpath}'
        warning(msg, lenght=len(msg)+2)
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
        msg = f'Histogram {hName} not found in {fpath}'
        warning(msg, lenght=len(msg)+2)
        f.Close()
        return None
    
    # Detach histogram from file to prevent cleanup issues
    h.SetDirectory(0)

    # Apply WW cross-section correction if needed
    scale, xsec = 1.0, getMetaInfo(proc, remove=False)
    if xsec!=0 and 'p8_ee_WW_ecm' in proc:
        xsec_new = getMetaInfo(proc, remove=True)
        scale = xsec_new / xsec

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
            proc_scale: float = 1.
            ) -> ROOT.TH1:
    '''
    Retrieve and sum histograms across multiple processes.
    
    Handles missing files gracefully based on lazy mode and applies cross-section
    corrections, rebinning, and scaling.
    
    Args:
        hName (str): Name of the histogram to retrieve.
        procs (str): List of process names to combine.
        inDir (str): Input directory path.
        suffix (str, optional): File suffix. Defaults to ''.
        rebin (int, optional): Rebinning factor. Defaults to 1.
        lazy (bool, optional): If True, skip missing files; if False, raise warnings. Defaults to True.
        proc_scale (float, optional): Global scaling factor for the histogram. Defaults to 1.0.
    
    Returns:
        ROOT.TH1: Combined and scaled histogram, or None if no histograms found.
    '''
    hist = None
    names, where = [], []

    for proc in procs:
        fInName = os.path.join(inDir, f'{proc}{suffix}.root')

        # Check file existence
        if not os.path.exists(fInName):
            if lazy:
                names.append(fInName)
                where.append('File existence')
                continue
            else:
                msg = f'ERROR: cannot open input file {fInName}'
                warning(msg, lenght=len(msg)+2)

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
                msg = f'ERROR: cannot open input file {fInName}'
                warning(msg, lenght=len(msg)+2)

        # Retrieve histogram
        h = f.Get(hName)
        if h is None:
            f.Close()
            if lazy:
                names.append(names)
                where.append('Histogram retrieval')
                continue
            else:
                msg = f'Histogram {hName} not found in {fInName}'
                warning(msg, lenght=len(msg)+2)
                return None
        
        # Detach histogram from file
        h.SetDirectory(0)

        # Apply WW cross-section correction
        scale, xsec_old = 1.0, getMetaInfo(proc)
        if xsec_old!=0 and 'p8_ee_WW_ecm' in proc:
            xsec_new = getMetaInfo(proc, remove=True)
            scale = xsec_new / xsec_old

        if scale!=1:
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
def concat(h_list: list[ROOT.TH1], 
           hName: str, 
           outName: str = ''
           ) -> ROOT.TH1:
    '''
    Concatenate multiple histograms into a single unrolled 1D histogram.
    
    Args:
        h_list (list[ROOT.TH1]): List of ROOT histograms to concatenate.
        hName (str): Name for logging purposes.
        outName (str, optional): Name for the output histogram. Defaults to '' (uses hName if empty).
    
    Returns:
        ROOT.TH1: Unrolled 1D histogram combining all input histograms.
    '''
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
def proc_scale(hist: ROOT.TH1, 
               proc: str, 
               processes: dict[str, list[str]], 
               proc_scales: dict[str, float] = {}
               ) -> ROOT.TH1:
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

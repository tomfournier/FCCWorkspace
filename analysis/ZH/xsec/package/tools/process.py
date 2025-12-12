import os, ROOT

from ..config import warning
from .utils import get_procDict

PROCDICT_CACHE = {}

def _get_procDict(procFile: str = 'FCCee_procDict_winter2023_IDEA.json',
                  fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts') -> dict:
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

    procDict = _get_procDict(procFile, fcc=fcc)

    if proc not in procDict:
        return None

    xsec = procDict[proc][info]
    if remove:
        if 'p8_ee_WW_ecm' in proc:
            xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'))
            xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'))
            xsec      = getMetaInfo(proc) - xsec_ee - xsec_mumu
    return xsec

#______________________________________________
def get_hist(hName: str, 
             proc: str, 
             procs_cfg: dict[str, list[str]], 
             inDir: str, 
             suffix: str = '',
             proc_scales: dict[str, float] = {}
             ) -> ROOT.TH1:
    
    fpath = os.path.join(inDir, f'{proc}{suffix}.root')
    print(f'----->[Info] Getting histogram from \n\t {fpath}')
    
    if not os.path.exists(fpath):
        msg = f'Input file not found: {fpath}'
        warning(msg, lenght=len(msg)+2)
        return None
    
    f = ROOT.TFile.Open(fpath)
    if not f or f.IsZombie():
        if f:
            f.Close()
        return None
    
    h = f.Get(hName)
    if h is None:
        msg = f'Histogram {hName} not found in {fpath}'
        warning(msg, lenght=len(msg)+2)
        f.Close()
        return None
    
    h.SetDirectory(0)

    scale, xsec = 1.0, getMetaInfo(proc, remove=False)
    if xsec!=0 and 'p8_ee_WW_ecm' in proc:
        xsec_new = getMetaInfo(proc, remove=True)
        scale = xsec_new / xsec

    if scale!=1.0:
        h.Scale(scale)

    f.Close()

    h = proc_scale(h, proc, procs_cfg, proc_scales=proc_scales)
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
    hist = None

    for proc in procs:
        fInName = os.path.join(inDir, f'{proc}{suffix}.root')

        if not os.path.exists(fInName):
            if lazy:
                continue
            else:
                msg = f'ERROR: cannot open input file {fInName}'
                warning(msg, lenght=len(msg)+2)

        f = ROOT.TFile.Open(fInName)
        if not f or f.IsZombie():
            if f:
                f.Close()
            if lazy:
                continue
            else:
                msg = f'ERROR: cannot open input file {fInName}'
                warning(msg, lenght=len(msg)+2)

        h = f.Get(hName)
        if h is None:
            f.Close()
            if lazy: 
                continue
            else:
                msg = f'Histogram {hName} not found in {fInName}'
                warning(msg, lenght=len(msg)+2)
                return None
        
        h.SetDirectory(0)

        scale, xsec_old = 1.0, getMetaInfo(proc)
        if xsec_old!=0 and 'p8_ee_WW_ecm' in proc:
            xsec_new = getMetaInfo(proc, remove=True)
            scale = xsec_new / xsec_old

        if scale!=1:
            h.Scale(scale)

        if hist is None: 
            hist = h
        else: 
            hist.Add(h)

        f.Close()

    if hist is None:
        return None

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
    print(f'----->[Info] Concatenating {hName}')

    tot_bins = sum([h.GetNbinsX() for h in h_list])

    h_concat = ROOT.TH1D('h1', '1D Unrolled Histogram', 
                         tot_bins, 0.5, tot_bins + 0.5)
    h_concat.SetDirectory(0)

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
               procs_cfg: dict[str, list[str]], 
               proc_scales: dict[str, float] = {}
               ) -> ROOT.TH1:
    if not proc_scales:
        return hist
    
    for proc_name, proc_list in procs_cfg.items():
        if proc in proc_list and proc_name!='Rare':
            scale = proc_scales.get(proc_name)
            if scale is not None:
                hist.Scale(scale)
                print(f'----->[Info] Scaled histogram '
                    f'to ILC scale by a factor of {scale:.3f}')
            break
    return hist

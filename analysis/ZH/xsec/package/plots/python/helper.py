import os, json, copy, uproot, ROOT

import numpy as np
import pandas as pd

from functools import lru_cache
from glob import glob

from ...tools.utils import mkdir



@lru_cache(maxsize=None)
def _key_from_file(filepath: str, 
                   key: str
                   ) -> int:
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
def _cut_from_file(filepath: str, 
                   branch: str
                   ) -> int:
    try:
        return int(uproot.open(filepath)['events'][branch].array(entry_stop=1)[0])
    except Exception:
        return 0
    
@lru_cache(maxsize=None)
def _col_from_file(filepath: str
                   ) -> set[str]:
    try:
        f = uproot.open(filepath)
        if 'events' in f:
            return set(map(str, f['events'].keys()))
    except Exception:
        pass
    return set()



#____________________________________
def find_sample_files(inDir: str, 
                      sample: str
                      ) -> list[str]:
    '''Find input files for the specified sample name.'''
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
        
#_____________________________________________
def get_processed(files: list[str] | str, 
                  arg: str = 'eventsProcessed'
                  ) -> int:
    if isinstance(files, str):
        files = [files]
    return int(sum(_key_from_file(os.fspath(f), arg) 
                   for f in files))

#____________________________
def get_cut(files: list[str], 
            cut: str) -> int:
    if isinstance(files, str):
        files = [files]
    return int(sum(_cut_from_file(os.fspath(f), cut) 
                   for f in files))

#_________________________________________
def getcut(df: pd.DataFrame, 
           filter: str,
           mask: np.ndarray | None = None,
           ) -> tuple[int, np.ndarray]:
    
    if df is None or df.empty:
        return 0, mask
    sel_mask = np.asarray(df.eval(filter), 
                          dtype=bool)
    if mask is None:
        mask = sel_mask
    else:
        mask &= sel_mask
    return int(mask.sum()), mask

#________________________________________________
def get_count(df: pd.DataFrame,
              df_mask: np.ndarray | None,
              file_list: list[str],
              cut_name: str,
              filter_expr: str,
              columns: set | None = None
              ) -> tuple[int, np.ndarray | None]:

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

#__________________________________
def is_there_events(proc: str, 
                    path: str = '', 
                    end='.root'
                    ) -> bool:
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

#________________________________________________
def dump_json(flow: dict[str, 
                         dict[str, 
                              dict[str, float]]], 
              outDir: str, 
              outName: str, 
              hist: bool = False, 
              procs: list = []) -> None:

    dictio = copy.deepcopy(flow)
    if hist:
        for proc in procs:
            del dictio[proc]['hist']
    
    mkdir(outDir)
    with open(f'{outDir}/{outName}.json', 'w') as fOut:
        json.dump(dictio, fOut, indent=4)

#________________________________________________________________
def get_flow(events: dict[str, 
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
                            ROOT.TH1 | dict[str, 
                                            float]]]:
    
    procs[0] = f'Z{cat}H' if not tot else 'ZH'
    _cat, _sel, _tot = f'_{cat}', f'_{sel}', '_tot' if tot else ''

    flow = {}
    for proc in procs:
        flow[proc] = {}
        flow[proc]['hist'], flow[proc]['cut'], flow[proc]['err']  = [], {}, {}
        hist = ROOT.TH1D(proc+_cat+_sel+_tot, proc+_cat+_sel+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            flow[proc]['cut'][cut], flow[proc]['err'][cut]  = 0, 0
            for sample in processes[proc]:
                if 'Hinv' in sample: continue
                flow[proc]['cut'][cut] += events[sample][sel]['cut'][cut]
                flow[proc]['err'][cut] += events[sample][sel]['err'][cut]**2
            
            hist.SetBinContent(i+1, flow[proc]['cut'][cut])
            hist.SetBinError(i+1,   flow[proc]['err'][cut]**0.5)
        flow[proc]['hist'].append(hist)
    
    if json_file:
        dump_json(flow, loc, outName+_sel+suffix, hist=True, procs=procs)
    return flow

#______________________________________________________________________
def get_flow_decay(events: dict[str,
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
                                  ROOT.TH1 | dict[str,
                                                  float]]]:
    
    cats = z_decays if tot else [cat]
    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in cats] for y in h_decays]
    _cat, _sel, _tot = f'_{cat}', f'_{sel}', '_tot' if tot else ''

    flow = {}
    for h, sig in zip(h_decays, sigs):
        flow[h] = {}
        flow[h]['hist'], flow[h]['cut'], flow[h]['err'] = [], {}, {}
        hist = ROOT.TH1D('H'+h+_cat+_sel+_tot, 'H'+h+_cat+_sel+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            flow[h]['cut'][cut], flow[h]['err'][cut] = 0, 0
            for s in sig:
                flow[h]['cut'][cut] += events[s][sel]['cut'][cut]
                flow[h]['err'][cut] += events[s][sel]['err'][cut]**2
            hist.SetBinContent(i+1, flow[h]['cut'][cut])
            hist.SetBinError(i+1,   flow[h]['err'][cut]**0.5)
        flow[h]['hist'].append(hist)

    if json_file:
        dump_json(flow, loc, outName+_sel+suffix, hist=True, procs=h_decays)
    return flow

#_________________________________________________________________
def get_flows(procs: list[str], 
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
    
    suffix = '_tot' if tot else ''
    flow = get_flow(events, procs, 
                    processes, cuts, 
                    cat, sel, tot=tot,
                    json_file=json_file, 
                    loc=loc_json, 
                    suffix=suffix)
    
    flow_decay = get_flow_decay(events, 
                                z_decays, h_decays, 
                                cuts, cat, sel, ecm=ecm, 
                                json_file=json_file, 
                                loc=loc_json, 
                                suffix=suffix, 
                                tot=tot)
    return flow, flow_decay

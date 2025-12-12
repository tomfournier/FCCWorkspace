import os, json, ROOT
import numpy as np
import pandas as pd

from typing import Callable, Union

#_____________________________
def get_paths(mode: str, 
              path: str, 
              modes: str, 
              suffix: str = ''
              ) -> list:
    from glob import glob
    fpath = os.path.join(path, modes[mode]+suffix)
    return glob(f'{fpath}.root')


#_____________________________
def get_df(filename: str, 
           branches: list[str] = []
           ) -> pd.DataFrame:
    from uproot import open
    with open(filename) as file:
        tree = file['events']
        if tree.num_entries==0:
            return pd.DataFrame()
        if branches:
            return tree.arrays(branches, library='pd')
        return tree.arrays(library='pd')

#___________________
def mkdir(mydir: str
          ) -> None:
    os.makedirs(mydir, exist_ok=True)


#________________________________________________________
def get_procDict(procFile: str, 
                 fcc: str = '/cvmfs/fcc.cern.ch/FCCDicts'
                 ) -> dict[str, 
                           dict[str, 
                                float]]:
    env = os.getenv('FCCDICTSDIR')
    base_dir = env.split(':')[0] if env else fcc
    proc_path = os.path.join(base_dir, procFile)

    if not os.path.isfile(proc_path):
        raise FileNotFoundError(f'----> No procDict found: ==={proc_path}===')

    with open(proc_path, 'r') as f:
        procDict = json.load(f)
    return procDict


#________________________________________________
def update_keys(procDict: dict, 
                modes: list
                ) -> dict[str, 
                          dict[str, 
                               float]]:

    reversed_mode_names = {v: k for k, v in modes.items()}
    
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
    if training:
        procFile = 'FCCee_procDict_winter2023_training_IDEA.json'
    else:
        procFile = 'FCCee_procDict_winter2023_IDEA.json'
    
    proc_dict = get_procDict(procFile)
    procDict  = update_keys(proc_dict, modes)

    xsec = {}
    for key, value in procDict.items(): 
        if key in modes: xsec[key] = value['crossSection']
    return xsec


#___________________________________________
def load_data(inDir: str, 
              filename: str = 'preprocessed'
              ) -> pd.DataFrame:
    fpath = os.path.join(inDir, filename+'.pkl')
    df = pd.read_pickle(fpath)
    return df


#________________________________________
def to_pkl(df: pd.DataFrame, 
           path: str, 
           filename: str = 'preprocessed'
           ) -> None:
    mkdir(path)
    fpath = os.path.join(path, filename+'.pkl')
    df.to_pickle(fpath)
    print(f'--->Preprocessed saved {fpath}')

#____________________________
def dump_json(arg: dict, 
              file: str, 
              indent: int = 4
              ) -> None:
    with open(file, mode='w', 
              encoding='utf-8') as fOut:
        json.dump(arg, fOut, indent=indent)

#_______________________
def load_json(file: str
              ) -> dict:
    with open(file, mode='r', 
              encoding='utf-8') as fIn:
        arg = json.load(fIn)
    return arg


#_________________
def Z0(S: float, 
       B: float
       ) -> float:
    if B<=0:
        return np.nan
    return np.sqrt( 2*( (S + B)*np.log(1 + S/B) - S ) )


#__________________
def Zmu(S: float, 
        B: float
        ) -> float:
    if B<=0:
        return np.nan
    return np.sqrt( 2*( S - B*np.log(1 + S/B) ) )


#________________
def Z(S: float, 
      B: float
      ) -> float:
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
                 score_range: tuple = (0, 1), 
                 nbins: int = 50) -> pd.DataFrame:      
    # prepare arrays and weighted sums
    s_vals, s_w = df_s[column].values, df_s[weight].values
    b_vals, b_w = df_b[column].values, df_b[weight].values

    S0, B0 = s_w.sum(), b_w.sum()
    print('initial: S0 = {:.2f}, B0 = {:.2f}'.format(S0, B0))
    print('inclusive Z: {:.2f}'.format(func(S0, B0)))

    # build weighted histograms once
    edges     = np.linspace(*score_range, nbins + 1)
    hist_s, _ = np.histogram(s_vals, bins=edges, weights=s_w)
    hist_b, _ = np.histogram(b_vals, bins=edges, weights=b_w)

    # cumulative sums from high score -> low score
    S_cum = np.cumsum(hist_s[::-1])[::-1]
    B_cum = np.cumsum(hist_b[::-1])[::-1]

    Z_vals = np.array([(Si, Bi, func(Si, Bi)) for Si, Bi in zip(S_cum, B_cum)])
    return pd.DataFrame(data=Z_vals, index=edges[:-1], columns=['S', 'B', 'Z'])

#__________________________________
def get_stack(hists: list[ROOT.TH1]
              ) -> ROOT.TH1:
    if not hists:
        raise ValueError('get_stack requires at least one histogram')
    hist = hists[0].Clone()
    hist.SetDirectory(0)
    hist.SetName(hist.GetName() + '_stack')
    for h in hists[1:]: hist.Add(h)
    return hist

#___________________________________________________
def get_xrange(hist: ROOT.TH1, 
               strict: bool = True, 
               xmin: Union[float, int, None] = None,
               xmax: Union[float, int, None] = None
               ) -> tuple[Union[float, int], 
                          Union[float, int]]:
    nbins = hist.GetNbinsX()
    bin_data = [(hist.GetBinLowEdge(i+1), 
                 hist.GetBinLowEdge(i+2), 
                 hist.GetBinContent(i+1)) for i in range(nbins)]

    mask = [(le, he) for le, he, c in bin_data 
            if (xmin is None or le > xmin) 
            and (xmax is None or he < xmax)
            and (not strict or c != 0)]

    if not mask:
        # fallback to full range if nothing matched
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
    nbins = hist.GetNbinsX()
    contents = np.array([hist.GetBinContent(i+1) \
                         for i in range(nbins)], dtype=float)

    if logY:
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
    
    h_stack = get_stack(h_sigs + h_bkgs)

    xMin, xMax = get_xrange(h_stack, 
                            strict=strict, 
                            xmin=xmin, 
                            xmax=xmax)
    h_bkg = [get_stack(h_bkgs)]
    yMin = np.array([
        get_yrange(h, 
                    logY=logY,
                    ymin=ymin,
                    ymax=ymax,
                    scale_min=scale_min,
                    scale_max=scale_max)[0] 
                    for h in h_sigs + h_bkgs
    ])
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
    
    h_sig = get_stack(h_sigs)
    xMin, xMax = get_xrange(h_sig, 
                            strict=strict, 
                            xmin=xmin, 
                            xmax=xmax)

    y_ranges = np.array([get_yrange(h, 
                                    logY=logY, 
                                    ymin=ymin, 
                                    ymax=ymax) 
                                    for h in h_sigs])
    yMin = y_ranges[:,0].min()*scale_min
    yMax = y_ranges[:,1].max()*scale_max

    return xMin, xMax, yMin, yMax
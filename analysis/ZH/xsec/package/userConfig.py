'''User-facing configuration for the ZH cross-section analysis.

Provides:
- Global parameters: `plot_file`, `frac`, `nb`, `ww`, `ecm`, `lumi`.
- Storage toggle and repo discovery via `eos` and `repo`.
- Centralized filesystem layout under `loc` with template paths that use
    placeholders: `cat` (category like 'ee' or 'mumu'), `ecm` (GeV),
    `sel` (selection name, e.g. 'Baseline').
- Path helpers via `PathTemplate`: `loc.COMBINE.get(cat, ecm, sel)` to
    expand placeholders; `loc.COMBINE.astype(Path)` to convert to
    `pathlib.Path`.
- Small utilities to work with the layout and imports.

Conventions:
- `lumi` is in ab^-1 (10.8 at 240 GeV; 3.1 at 365 GeV).
- Path templates in `loc` are expanded with `.get()` or `.astype()`.
- Event files are ROOT files containing a TTree named 'events'.

Helpers:
- `get_loc(path, cat, ecm, sel)`: Expand a `loc` template into a concrete path.
- `event(procs, path='', end='.root')`: Keep processes whose files/directories
    all contain an 'events' TTree (supports single-file and multi-file layouts).
- `add_package_path(names)`: Append one or more directories to `sys.path`.

Notes:
- When `eos=True`, `repo` is resolved from the current working directory; a
    warning is printed if 'xsec' is not in the path.

Lazy Imports:
- uproot, glob, and json are lazy-loaded only when needed in functions
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os
from pathlib import Path
from typing import overload, Type, Union



######################
##### PARAMETERS #####
######################

# Output plot file formats
plot_file = ['png']

# Data processing parameters: 
# fraction of data to use, number of chunks
frac, nb  = 1, 10

# WW process flag
ww = True

# Center-of-mass energy in GeV
cat, ecm = 'mumu', 240

# Integrated luminosity in ab-1 (10.8 for 240 GeV, 3.1 for 365 GeV)
lumi = 10.8 if ecm==240 else (3.1 if ecm==365 else -1)



#############################
##### LOCATION OF FILES #####
#############################

# Toggle between EOS and AFS storage
eos = True
if eos: repo = os.path.abspath('.')
# example: '/eos/user/t/tofourni/FCC/FCCWorkspace/analysis/ZH/xsec'
else: repo = os.getenv('PWD')
# example: '/afs/cern.ch/user/t/tofourni/eos/FCC/FCCWorkspace/analysis/ZH/xsec'

if not 'xsec' in repo:
    print('WARNING: You are not executing the script from the good directory')

class loc : pass
# Location of the root folder
loc.ROOT               = repo

# Location of primary folders
loc.PACKAGE            = f'{loc.ROOT}/package'
loc.OUT                = f'{loc.ROOT}/output'
loc.PLOTS              = f'{loc.OUT}/plots'
loc.DATA               = f'{loc.OUT}/data'
loc.TMP                = f'{loc.OUT}/tmp'

# Location of files needed for configuration
loc.JSON               = f'{loc.TMP}/config_json'
loc.RUN                = f'{loc.JSON}/run'

# Location of events
loc.EVENTS             = f'{loc.DATA}/events/ecm/cat/full/analysis'
loc.EVENTS_TRAINING    = f'{loc.DATA}/events/ecm/cat/full/training'
loc.EVENTS_TEST        = f'{loc.DATA}/events/ecm/cat/test'

# Location of MVA related files
loc.MVA                = f'{loc.DATA}/MVA'
loc.MVA_INPUTS         = f'{loc.MVA}/ecm/cat/sel/MVAInputs'
loc.BDT                = f'{loc.MVA}/ecm/cat/sel/BDT'

# Location of histograms
loc.HIST               = f'{loc.DATA}/histograms'

loc.HIST_MVA           = f'{loc.HIST}/MVAInputs/ecm/cat/'
loc.HIST_PREPROCESSED  = f'{loc.HIST}/preprocessed/ecm/cat'
loc.HIST_PROCESSED     = f'{loc.HIST}/processed/ecm/cat/sel'

# Location of plots
loc.PLOTS_MVA          = f'{loc.PLOTS}/MVAInputs/ecm/cat'
loc.PLOTS_BDT          = f'{loc.PLOTS}/evaluation/ecm/cat/sel'
loc.PLOTS_MEASUREMENT  = f'{loc.PLOTS}/measurement/ecm/cat'

# Location of combine files
loc.COMBINE            = f'{loc.DATA}/combine/sel/ecm/cat'

loc.COMBINE_NOMINAL    = f'{loc.COMBINE}/nominal'
loc.COMBINE_BIAS       = f'{loc.COMBINE}/bias'

# Location of combine files when doing nominal fit
loc.NOMINAL_LOG        = f'{loc.COMBINE_NOMINAL}/log' 
loc.NOMINAL_RESULT     = f'{loc.COMBINE_NOMINAL}/results'
loc.NOMINAL_DATACARD   = f'{loc.COMBINE_NOMINAL}/datacard'
loc.NOMINAL_WS         = f'{loc.COMBINE_NOMINAL}/WS'

# Location of combine files when doing bias test
loc.BIAS_LOG           = f'{loc.COMBINE_BIAS}/log' 
loc.BIAS_FIT_RESULT    = f'{loc.COMBINE_BIAS}/results/fit'
loc.BIAS_RESULT        = f'{loc.COMBINE_BIAS}/results/bias'
loc.BIAS_DATACARD      = f'{loc.COMBINE_BIAS}/datacard'
loc.BIAS_WS            = f'{loc.COMBINE_BIAS}/WS'



#################
### FUNCTIONS ###
#################

#_____________________
def get_loc(path: str, 
            cat: str, 
            ecm: int, 
            sel: str,
            ) -> str:
    '''Backwards-compatible wrapper around `loc.get()`.

    Args:
        path: Path template containing 'cat', 'ecm', 'sel' placeholders
        cat: Category name (e.g., 'ee', 'mumu')
        ecm: Center-of-mass energy in GeV
        sel: Selection name (e.g., 'Baseline')

    Returns:
        str: Path with substituted values
    '''
    return path.replace('cat', cat).replace('ecm', str(ecm)).replace('sel', sel)

#___________________________
def event(procs: list[str], 
          path: str = '', 
          end: str = '.root'
          ) -> list[str]:
    '''Filter processes that contain valid ROOT event trees.
    
    Args:
        procs: List of process names to validate
        path: Base path where process files are located
        end: File extension (default: '.root')
    
    Returns:
       List[str]: List of valid process names with 'events' TTree
    '''
    import uproot
    from glob import glob

    newprocs = []
    for proc in procs:
        file = os.path.join(path, proc)
        # Check for single file or directory with multiple files
        filenames = [f'{file}{end}'] \
            if os.path.exists(f'{file}{end}') \
            else glob(f'{file}/*')
        
        # Verify all files contain 'events' TTree
        isTTree = [i for i, filename in enumerate(filenames) \
                   if 'events' in uproot.open(filename)]
        if len(isTTree)==len(filenames):
            newprocs.append(proc)
    return newprocs

@overload
def get_params(
    env: os._Environ, 
    cfg_json: str, 
    is_final: bool = False, 
    is_presel3: bool = False
    ) -> tuple[str, int]:
    ...

@overload
def get_params(
    env: os._Environ, 
    cfg_json: str, 
    is_final: bool = True, 
    is_presel3: bool = False
    ) -> tuple[str, int, float]:
    ...

@overload
def get_params(
    env: os._Environ, 
    cfg_json: str, 
    is_final: bool = True, 
    is_presel3: bool = True
    ) -> tuple[str, int, bool]:
    ...

@overload
def get_params(
    env: os._Environ, 
    cfg_json: str, 
    is_final: bool = False, 
    is_presel3: bool = True
    ) -> tuple[str, int, bool]:
    ...

#__________________________________________
def get_params(
        env: os._Environ, 
        cfg_json: str,
        is_final: bool = False,
        is_presel3: bool = False,
        ) -> Union[tuple[str, int], 
                   tuple[str, int, float]]:

    import json
    from pathlib import Path
    
    if env.get('RUN'):
        cfg_file = Path(loc.RUN) / cfg_json
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            cat, ecm, lumi = cfg['cat'], cfg['ecm'], cfg['lumi']
            if is_presel3: ww = cfg['ww']
        else:
            raise FileNotFoundError(f"Couldn't find {cfg_file} file")
    else:
        cat = input('Select channel [ee, mumu]: ')
        ecm = int(input('Select center-of-mass energy [240, 365]: '))
        lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)
        if is_presel3:
            ww = bool(input(f'Do only for p8_ee_WW_ecm{ecm}? [True, False]: '))

    if is_final and is_presel3:
        is_final = False

    if is_final:
        return cat, ecm, lumi
    elif is_presel3:
        return cat, ecm, ww
    
    return cat, ecm

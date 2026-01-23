'''Core configuration for path templates and helpers.

Provides:
- Path templates via `loc` with placeholders: `cat`, `ecm`, `sel`.
- Type-flexible expansion with `LocPath.get()` and `loc.get(...)`.
- Bidirectional type conversion via `astype(str)` and `astype(Path)`.
- Global parameters: `plot_file`, `frac`, `nb`, `ww`, `cat`, `ecm`, `lumi`.
- Utilities: `get_loc()`, `event()`, `get_params()`.

Conventions:
- `lumi` is in ab^-1 (10.8 at 240 GeV; 3.12 at 365 GeV).
- Templates are expanded with `loc.get()` or `LocPath.get()`.
- Expanded paths can be str or pathlib.Path; both support `astype()`.

Usage:
- path = loc.EVENTS
- path1 = loc.EVENTS.get(cat='ee', ecm=240, sel='Baseline')
- path2 = loc.get('EVENTS', cat='ee', ecm=240, sel='Baseline', type=Path)
- path3 = path1.astype(str)
- path4 = path3.astype(Path)
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os
from pathlib import Path
from typing import overload, Type, Union



##################
### PARAMETERS ###
##################

# Output plot file formats
plot_file = ['png']

# Data processing parameters: 
# fraction of data to use, number of chunks
frac, nb  = 1, 10

# WW process flag
ww = True

# Center-of-mass energy in GeV
cat, ecm = 'mumu', 240

# Integrated luminosity in ab-1 (10.8 for 240 GeV, 3.12 for 365 GeV)
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)



#############################
##### LOCATION OF FILES #####
#############################

############################
### Root file definition ###
############################

# Toggle between EOS and AFS storage
eos = True
if eos: repo = os.path.abspath('.')
# example: '/eos/home-t/u/username/FCC/FCCWorkspace/analysis/ZH/xsec'
else: repo = os.getenv('PWD')
# example: '/afs/cern.ch/user/u/username/eos/FCC/FCCWorkspace/analysis/ZH/xsec'

if not 'xsec' in repo:
    print('WARNING: You are not executing the script from the good directory')

###############################
### Custom class definition ###
###############################

# str-like class with get and astype methods
class LocPath(str):
    """String subclass for templates and expanded paths.
    
    Adds:
    - `get(...)`: expand placeholders in a template
    - `astype(type)`: convert between LocPath (str) and PathObj (Path)
    """
    
    def astype(self, 
               type: Type[Union[str, Path]]
               ) -> Union['LocPath', 
                          'PathObj']:
        """Convert to str (LocPath) or Path (PathObj).
        
        Args:
            type: Either str or Path class
            
        Returns:
            LocPath or PathObj with an `astype()` method for roundtrip conversion
        """
        # str → LocPath; Path → PathObj; keep interface symmetric
        if type is str:
            return LocPath(self)
        elif type is Path:
            return PathObj(Path(str(self)))
        raise TypeError("Only 'str' or 'Path' supported")
    
    def get(self, 
            name: Union[str, None] = None, 
            cat: Union[str, None] = None,
            ecm: Union[int, None] = None, 
            sel: Union[str, None] = None,
            type: Type[Union[str, Path]] = str
            ) -> Union['LocPath', 
                       'PathObj']:
        """Expand placeholders in this template or a named template.
        
        Args:
            name: Template name ('EVENTS', etc.). If None, self is the template.
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) or Path
            
        Returns:
            Expanded LocPath or PathObj (both with `astype()`)
        """
        # If name is None, expand self; otherwise fetch `loc.<name>`
        template = self if name is None else getattr(loc, name, name)
        expanded = loc.expand(template, cat=cat, ecm=ecm, sel=sel)
        return LocPath(expanded).astype(type)


# pathlib.Path-like class with get and astype methods
class PathObj:
    """Thin wrapper over `pathlib.Path` mirroring `LocPath` interface.
    
    Adds:
    - `get(...)`: expand a named template from `loc`
    - `astype(type)`: convert between PathObj (Path) and LocPath (str)
    """
    
    def __init__(self, path: Union[Path, str]):
        self.path = path if isinstance(path, Path) else Path(path)
    
    def astype(self, 
               type: Type[Union[str, Path]]
               ) -> Union[LocPath, 
                          'PathObj']:
        """Convert to str (LocPath) or Path (PathObj).
        
        Args:
            type: Either str or Path class
            
        Returns:
            LocPath or PathObj with an `astype()` method for roundtrip conversion
        """
        # Path → PathObj; str → LocPath; keep interface symmetric
        if type is Path:
            return PathObj(self.path)
        elif type is str:
            return LocPath(str(self.path))
        raise TypeError("Only 'str' or 'Path' supported")
    
    def get(self, name: str, 
            cat: Union[str, None] = None,
            ecm: Union[int, None] = None, 
            sel: Union[str, None] = None,
            type: Type[Union[str, Path]] = str
            ) -> Union[LocPath, 
                       'PathObj']:
        """Fetch named template from `loc`, expand, and return.
        
        Args:
            name: Template name ('EVENTS', etc.)
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) or Path
            
        Returns:
            Expanded LocPath or PathObj (both with `astype()`)
        """
        template = getattr(loc, name, name)
        expanded = loc.expand(template, cat=cat, ecm=ecm, sel=sel)
        return LocPath(expanded).astype(type)
    
    def __str__(self) -> str:
        return str(self.path)
    
    def __repr__(self) -> str:
        return repr(self.path)
    
    def __truediv__(self, other) -> 'PathObj':
        return PathObj(self.path / other)
    
    def __getattr__(self, name):
        """Forward unknown attributes to underlying Path object."""
        return getattr(self.path, name)

######################
### Path defintion ###
######################

class loc:
    """Path templates with placeholders ('cat', 'ecm', 'sel') and expansion methods."""

    # Templates as LocPath strings with placeholders
    ROOT               = LocPath(repo)                   # Repo root
    PACKAGE            = LocPath(f"{repo}/package")      # Python package code
    OUT                = LocPath(f"{repo}/output")       # Output root
    PLOTS              = LocPath(f"{repo}/output/plots") # Plots root
    DATA               = LocPath(f"{repo}/output/data")  # Data artifacts
    TMP                = LocPath(f"{repo}/output/tmp")   # Scratch state
    
    JSON               = LocPath(f"{repo}/output/tmp/config_json")     # JSON configs
    RUN                = LocPath(f"{repo}/output/tmp/config_json/run") # Per-run configs
    
    EVENTS             = LocPath(f"{repo}/output/data/events/ecm/cat/full/analysis") # Full ntuples
    EVENTS_TRAINING    = LocPath(f"{repo}/output/data/events/ecm/cat/full/training") # Training split
    EVENTS_TEST        = LocPath(f"{repo}/output/data/events/ecm/cat/test")          # Test split
    
    MVA                = LocPath(f"{repo}/output/data/MVA")                       # MVA root
    MVA_INPUTS         = LocPath(f"{repo}/output/data/MVA/ecm/cat/sel/MVAInputs") # Engineered features
    BDT                = LocPath(f"{repo}/output/data/MVA/ecm/cat/sel/BDT")       # Trained BDT
    
    HIST               = LocPath(f"{repo}/output/data/histograms")                       # Histograms root
    HIST_MVA           = LocPath(f"{repo}/output/data/histograms/MVAInputs/ecm/cat")     # MVA input hists
    HIST_PREPROCESSED  = LocPath(f"{repo}/output/data/histograms/preprocessed/ecm/cat")  # After preprocess
    HIST_PROCESSED     = LocPath(f"{repo}/output/data/histograms/processed/ecm/cat/sel") # After processing
    
    PLOTS_MVA          = LocPath(f"{repo}/output/plots/MVAInputs/ecm/cat")      # Input var plots
    PLOTS_BDT          = LocPath(f"{repo}/output/plots/evaluation/ecm/cat/sel") # BDT perf plots
    PLOTS_MEASUREMENT  = LocPath(f"{repo}/output/plots/measurement/ecm/cat")    # XS measurement plots
    
    COMBINE            = LocPath(f"{repo}/output/data/combine/sel/ecm/cat")         # Combine root
    COMBINE_NOMINAL    = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal") # Nominal outputs
    COMBINE_BIAS       = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias")    # Bias outputs
    
    NOMINAL_LOG        = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/log")      # Logs (nominal)
    NOMINAL_RESULT     = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/results")  # Results (nominal)
    NOMINAL_DATACARD   = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/datacard") # Datacards (nominal)
    NOMINAL_WS         = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/WS")       # Workspaces (nominal)
    
    BIAS_LOG           = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/log")          # Logs (bias)
    BIAS_FIT_RESULT    = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/results/fit")  # Fit outputs (bias)
    BIAS_RESULT        = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/results/bias") # Bias summaries
    BIAS_DATACARD      = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/datacard")     # Datacards (bias)
    BIAS_WS            = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/WS")           # Workspaces (bias)

    @staticmethod
    def expand(
        template: Union[str, Path], 
        cat: Union[str, None] = None,
        ecm: Union[int, None] = None, 
        sel: Union[str, None] = None
        ) -> str:
        """Replace placeholders in a template string.
        
        Args:
            template: Template string containing 'cat', 'ecm', 'sel' placeholders
            cat, ecm, sel: Values to substitute. Raises ValueError if required but None.
            
        Returns:
            str: Template with placeholders replaced
        """
        tpl = str(template)
        needs_cat = 'cat' in tpl
        needs_ecm = 'ecm' in tpl
        needs_sel = 'sel' in tpl
        
        if needs_cat and cat is None:
            raise ValueError("'cat' is required to expand this path")
        if needs_ecm and ecm is None:
            raise ValueError("'ecm' is required to expand this path")
        if needs_sel and sel is None:
            raise ValueError("'sel' is required to expand this path")
        
        tpl = tpl.replace('cat', cat or '')
        tpl = tpl.replace('ecm', str(ecm) if ecm is not None else '')
        tpl = tpl.replace('sel', sel or '')
        return tpl

    @classmethod
    def get(cls, 
            name: str, 
            cat: Union[str, None] = None,
            ecm: Union[int, None] = None, 
            sel: Union[str, None] = None,
            type: Type[Union[str, Path]] = str
            ) -> Union[LocPath, 
                       PathObj]:
        """Fetch template by name, expand placeholders, and return as LocPath or PathObj.
        
        Args:
            name: Template name (e.g., 'EVENTS')
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) returns LocPath, Path returns PathObj
            
        Returns:
            LocPath or PathObj (both with astype() and get() methods)
        """
        template = getattr(cls, name)
        expanded = cls.expand(template, cat=cat, ecm=ecm, sel=sel)
        return LocPath(expanded).astype(type)



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
    return loc.get(path, cat=cat, ecm=ecm, sel=sel)

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
def get_params(env: os._Environ, 
               cfg_json: str, 
               is_final: bool = False, 
               is_presel3: bool = False
               ) -> tuple[str, int]:
    ...

@overload
def get_params(env: os._Environ, 
               cfg_json: str, 
               is_final: bool = True, 
               is_presel3: bool = False
               ) -> tuple[str, int, float]:
    ...

@overload
def get_params(env: os._Environ, 
               cfg_json: str, 
               is_final: bool = True, 
               is_presel3: bool = True
               ) -> tuple[str, int, bool]:
    ...

@overload
def get_params(env: os._Environ, 
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
            if is_presel3:
                ww = cfg['ww']
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

'''Core configuration for path templates and helpers.

Provides:
- Path templates via `loc` with placeholders: `cat`, `ecm`, `sel`.
- Type-flexible expansion with `LocPath.get()` and `loc.get(...)`.
- Bidirectional type conversion via `astype(str)` and `astype(Path)`.
- Global parameters: `plot_file`, `frac`, `nb`, `ww`, `cat`, `ecm`, `lumi`.
- Utilities: `event()`, `get_params()`.

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
from pathlib import Path, PosixPath, WindowsPath
from typing import overload, Type, Union

# Select the appropriate Path base class for the OS
_PathBase = WindowsPath if os.name == 'nt' else PosixPath

from .logger import get_logger

LOGGER = get_logger(__name__)



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

###############################
### Custom class definition ###
###############################

class LocPath(str):
    """String subclass for templates and expanded paths.

    Extends str with methods to:
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
            return PathObj(str(self))
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


class PathObj(_PathBase):
    """Path subclass with get and astype methods, inheriting all Path functionality.

    Extends pathlib.Path with methods to:
    - `get(...)`: expand a named template from `loc`
    - `astype(type)`: convert between PathObj (Path) and LocPath (str)
    """

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
            return PathObj(str(self))
        elif type is str:
            return LocPath(str(self))
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



######################
### Path defintion ###
######################

class locMeta(type):
    """Metaclass for loc to handle type conversion based on default type setting."""

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._default_type = str  # Default returns LocPath (str type)

    def __getattribute__(cls, name):
        """Intercept attribute access to convert templates based on default type."""
        obj = super().__getattribute__(name)

        # Convert templates (LocPath) based on default type
        # Only convert uppercase attributes (templates), not methods or private attrs
        if (isinstance(obj, LocPath) and not callable(obj) and
                name.isupper() and not name.startswith('_')):
            default_type = super().__getattribute__('_default_type')
            # Convert to PathObj if default type is Path or PathObj
            if default_type is Path or default_type is PathObj:
                return PathObj(str(obj))

        return obj

    def set_default_type(cls, type_: Union[Type[Union[str, Path]], str]) -> None:
        """Set the default type for loc.ATTRIBUTE access and loc.get() calls.

        Args:
            type_: Either a type object (str, LocPath, Path, PathObj) or a string
                   name ('str', 'LocPath', 'Path', 'PathObj')

        Raises:
            ValueError: If type_ is not one of the supported types or names
        """
        # Map of string names to type objects
        type_map = {
            'str':  str,  'LocPath': LocPath,
            'Path': Path, 'PathObj': PathObj,
        }

        # If type_ is a string, convert it to the actual type
        if isinstance(type_, str):
            if type_ not in type_map:
                raise ValueError(f"type_ must be one of {list(type_map.keys())}")
            type_ = type_map[type_]

        # Validate that the resulting type is one of the supported types
        if type_ not in (str, LocPath, Path, PathObj):
            raise ValueError("type_ must be str, LocPath, Path, or PathObj")

        cls._default_type = type_


class loc(metaclass=locMeta):
    """Path template registry with placeholders for category, energy, and selection.

    Templates use placeholders: 'cat' (channel), 'ecm' (energy), 'sel' (selection).
    All templates are LocPath instances that can be expanded via get() or astype().
    """

    repo = str(Path(__file__).parent.parent.resolve())

    # Templates as LocPath strings with placeholders: cat, ecm, sel
    ROOT                = LocPath(repo)                    # Repository root
    PACKAGE             = LocPath(f"{repo}/package")       # Python package directory
    OUT                 = LocPath(f"{repo}/output")        # Output root directory
    PLOTS               = LocPath(f"{repo}/output/plots")  # All plots output
    DATA                = LocPath(f"{repo}/output/data")   # All data artifacts
    TMP                 = LocPath(f"{repo}/output/tmp")    # Temporary/scratch data

    JSON                = LocPath(f"{repo}/output/tmp/config_json")      # JSON configuration directory
    RUN                 = LocPath(f"{repo}/output/tmp/config_json/run")  # Per-run configuration directory

    # Event directories: templates use {ecm}, {cat} placeholders
    EVENTS              = LocPath(f"{repo}/output/data/events/ecm/cat/full/analysis")  # Full event samples for analysis
    EVENTS_TEST         = LocPath(f"{repo}/output/data/events/ecm/cat/test/analysis")  # Test event samples for analysis
    EVENTS_TRAINING     = LocPath(f"{repo}/output/data/events/ecm/cat/full/training")  # Full event samples for BDT training
    EVENTS_TRAIN_TEST   = LocPath(f"{repo}/output/data/events/ecm/cat/test/training")  # Test event samples for BDT training

    # Optimisation directories: templates use {ecm}, {cat} placeholders
    OPTIMISATION        = LocPath(f"{repo}/output/data/optimisation/Inputs/ecm/cat/full")  # Full optimisation samples
    OPTIMISATION_TEST   = LocPath(f"{repo}/output/data/optimisation/Inputs/ecm/cat/test")  # Test optimisation samples
    OPTIMISATION_RES    = LocPath(f"{repo}/output/data/optimisation/results/ecm/cat")      # Optimisation results

    # FSR (Final State Radiation) directories: templates use {ecm}, {cat} placeholders
    FSR_TREE            = LocPath(f"{repo}/output/data/FSR/Inputs/ecm/cat/full")  # Full FSR samples
    FSR_TEST            = LocPath(f"{repo}/output/data/FSR/Inputs/ecm/cat/test")  # Test FSR samples
    FSR_RES             = LocPath(f"{repo}/output/data/FSR/results/ecm/cat")      # FSR analysis results

    # Multivariate analysis (BDT): templates use {ecm}, {cat}, {sel} placeholders
    MVA                 = LocPath(f"{repo}/output/data/MVA")                        # MVA root directory
    MVA_INPUTS          = LocPath(f"{repo}/output/data/MVA/ecm/cat/sel/MVAInputs")  # BDT input variables
    BDT                 = LocPath(f"{repo}/output/data/MVA/ecm/cat/sel/BDT")        # Trained BDT models

    # Histograms: templates use {ecm}, {cat}, {sel} placeholders
    HIST                = LocPath(f"{repo}/output/data/histograms")                           # Histograms root directory
    HIST_MVA            = LocPath(f"{repo}/output/data/histograms/MVAInputs/ecm/cat/")        # MVA input variable histograms
    HIST_PREPROCESSED   = LocPath(f"{repo}/output/data/histograms/preprocessed/ecm/cat")      # After final selection
    HIST_PROCESSED      = LocPath(f"{repo}/output/data/histograms/processed/ecm/cat/sel")     # After histogram processing
    HIST_OPTIMISATION   = LocPath(f"{repo}/output/data/histograms/optimisation/ecm/cat/sel")  # Optimisation analysis histograms

    # Plots: templates use {ecm}, {cat}, {sel} placeholders
    PLOTS_MVA           = LocPath(f"{repo}/output/plots/MVAInputs/ecm/cat")       # Input variable distributions
    PLOTS_BDT           = LocPath(f"{repo}/output/plots/evaluation/ecm/cat/sel")  # BDT performance and scores
    PLOTS_MEASUREMENT   = LocPath(f"{repo}/output/plots/measurement/ecm/cat")     # Analysis measurement plots
    PLOTS_OPTIMISATION  = LocPath(f"{repo}/output/plots/optimisation/ecm/cat")    # Selection optimisation plots
    PLOTS_FSR           = LocPath(f"{repo}/output/plots/fsr/ecm/cat")             # FSR analysis plots
    PLOTS_FIT_SCAN      = LocPath(f"{repo}/output/plots/fit/scans")                # Likelyhood scan comparison plots

    # Statistical fit: templates use {sel}, {ecm}, {cat} placeholders
    COMBINE             = LocPath(f"{repo}/output/data/combine/sel/ecm/cat")          # Combine root directory
    COMBINE_NOMINAL     = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal")  # Nominal analysis
    COMBINE_BIAS        = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias")     # Bias test results

    # Nominal fit outputs: templates use {sel}, {ecm}, {cat} placeholders
    NOMINAL_LOG         = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/log")       # Combine job logs
    NOMINAL_RESULT      = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/results")   # Fit results and plots
    NOMINAL_DATACARD    = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/datacard")  # Combine datacards
    NOMINAL_WS          = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/nominal/WS")        # Combine workspaces

    # Bias test outputs: templates use {sel}, {ecm}, {cat} placeholders
    BIAS_LOG            = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/log")           # Combine job logs
    BIAS_FIT_RESULT     = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/results/fit")   # Individual toy fit results
    BIAS_RESULT         = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/results/bias")  # Bias summaries and statistics
    BIAS_DATACARD       = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/datacard")      # Combine datacards
    BIAS_WS             = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/bias/WS")            # Combine workspaces


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
            type: Union[Type[Union[str, Path]], None] = None
            ) -> Union[LocPath,
                       PathObj]:
        """Fetch template by name, expand placeholders, and return as LocPath or PathObj.

        Args:
            name: Template name (e.g., 'EVENTS')
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) returns LocPath, Path returns PathObj.
                  If None, uses the default type set via set_default_type()

        Returns:
            LocPath or PathObj (both with astype() and get() methods)
        """
        template = getattr(cls, name)
        expanded = cls.expand(template, cat=cat, ecm=ecm, sel=sel)

        # Use default type if not explicitly provided
        if type is None:
            type = cls._default_type

        return LocPath(expanded).astype(type)



#################
### FUNCTIONS ###
#################

def event(procs: list[str],
          path: str = '',
          end: str = '.root'
          ) -> list[str]:
    """Filter processes that contain valid ROOT event trees.

    Validates that all files for each process contain the 'events' TTree.
    Supports both single files and directories with multiple files.

    Args:
        procs: List of process names to validate
        path: Base path where process files are located
        end: File extension to search for (default: '.root')

    Returns:
        List of process names where all associated files contain 'events' TTree
    """
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
        isTTree = [i for i, filename in enumerate(filenames)
                   if 'events' in uproot.open(filename)]
        if len(isTTree)==len(filenames):
            newprocs.append(proc)
    return newprocs

@overload
def get_params(
    env: os._Environ,
    cfg_json: str,
    is_final: bool = False,
    qq_allowed: bool = False,
     ) -> tuple[str, int]:
    ...

@overload
def get_params(
    env: os._Environ,
    cfg_json: str,
    is_final: bool = True,
    qq_allowed: bool = False,
     ) -> tuple[str, int, float]:
    ...

def get_params(
        env: os._Environ,
        cfg_json: str,
        is_final: bool = False,
        qq_allowed: bool = False,
     ) -> Union[tuple[str, int],
                tuple[str, int, float]]:
    """Retrieve analysis parameters from configuration or interactive input.

    In automated mode (RUN environment or HTCondor), reads from JSON config file.
    Otherwise, prompts user for channel, energy, and kinematic cut settings.

    Args:
        env: Environment variables (typically os.environ)
        cfg_json: JSON config filename in loc.RUN directory
        is_final: If True, also returns luminosity value; returns (cat, ecm, lumi, test)
                 If False, returns (cat, ecm, test)
        qq_allowed: If True, allows 'qq' channel; default channels are ['ee', 'mumu']

    Returns:
        If is_final=False: Tuple of (category, ecm_energy, test_flag)
            - category: 'ee', 'mumu', or 'qq' (if allowed)
            - ecm_energy: 240 or 365 (GeV)
            - test_flag: bool, True = kinematic cuts disabled, False = normal selection
        If is_final=True: Tuple of (category, ecm_energy, luminosity, test_flag)
            - luminosity: Integrated luminosity in ab^-1 (10.8 for 240 GeV, 3.12 for 365 GeV)

    Raises:
        FileNotFoundError: If config file not found in loc.RUN during automated mode
    """

    import json

    # Check if running in automated mode (RUN set) or on HTCondor
    is_batch = '_CONDOR_SCRATCH_DIR' in env
    is_automated = env.get('RUN') or is_batch

    cat_allowed = ['ee', 'mumu']
    if qq_allowed: cat_allowed.append('qq')

    if is_automated:
        # Local run: use loc.RUN
        cfg_file = loc.RUN.astype(Path) / cfg_json
        LOGGER.info(f'Getting config file from {cfg_file}')
        if cfg_file.exists():
            cfg: dict = json.loads(cfg_file.read_text())
            cat, ecm, lumi, test = cfg['cat'], cfg['ecm'], cfg.get('lumi', -1), cfg.get('test')
        else:
            raise FileNotFoundError(f"Couldn't find config file at {cfg_file}")
    else:
        cat = input(f'Select channel [{", ".join(cat_allowed)}]: ')
        while cat not in cat_allowed:
            cat = input(f'Wrong input selected, choose between [{", ".join(cat_allowed)}]: ')
        ecm = int(input('Select center-of-mass energy [240, 365]: '))
        while (ecm!=240) and (ecm!=365):
            ecm = int(input('Wrong input selected, choose between 240 and 365: '))
        lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

        test = input('Do selection with kinematic cuts? [Yes/No, True/False]: ')
        while test.lower() not in ['yes', 'y', 'no', 'n', 't', 'true', 'false', 'f']:
            test = input('Wrong input selected, choose between Yes/No or True/False: ')
        test = test.lower() in ['yes', 'y', 'true', 't']

    if is_final:
        return cat, ecm, lumi, test
    return cat, ecm, test

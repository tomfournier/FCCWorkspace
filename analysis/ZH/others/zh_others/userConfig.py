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
from pathlib import Path

from .logger import get_logger
from path import loc, LocPath

LOGGER = get_logger(__name__)



#############################
##### LOCATION OF FILES #####
#############################



repo = str(Path(__file__).parent.parent.resolve())

# Templates as LocPath strings with placeholders: cat, ecm, sel
loc.ROOT                = LocPath(repo)                    # Repository root
loc.PACKAGE             = LocPath(f"{repo}/zh_others")     # Python package directory
loc.OUT                 = LocPath(f"{repo}/output")        # Output root directory
loc.DATA                = LocPath(f"{repo}/output/data")   # All data artifacts
loc.PLOTS               = LocPath(f"{repo}/output/plots")  # All plots output
loc.TMP                 = LocPath(f"{repo}/output/tmp")    # Temporary/scratch data

# Optimisation directories: templates use {ecm}, {cat} placeholders
loc.OPTIMISATION        = LocPath(f"{repo}/output/data/optimisation/Inputs/ecm/cat/full")  # Full optimisation samples
loc.OPTIMISATION_TEST   = LocPath(f"{repo}/output/data/optimisation/Inputs/ecm/cat/test")  # Test optimisation samples
loc.OPTIMISATION_RES    = LocPath(f"{repo}/output/data/optimisation/results/ecm/cat")      # Optimisation results

# FSR (Final State Radiation) directories: templates use {ecm}, {cat} placeholders
loc.FSR_TREE            = LocPath(f"{repo}/output/data/FSR/Inputs/ecm/cat/full")  # Full FSR samples
loc.FSR_TEST            = LocPath(f"{repo}/output/data/FSR/Inputs/ecm/cat/test")  # Test FSR samples
loc.FSR_RES             = LocPath(f"{repo}/output/data/FSR/results/ecm/cat")      # FSR analysis results

# Histograms: templates use {ecm}, {cat}, {sel} placeholders
loc.HIST                = LocPath(f"{repo}/output/data/histograms")                           # Histograms root directory
loc.HIST_OPTIMISATION   = LocPath(f"{repo}/output/data/histograms/optimisation/ecm/cat/sel")  # Optimisation analysis histograms

# Plots: templates use {ecm}, {cat}, {sel} placeholders
loc.PLOTS_OPTIMISATION  = LocPath(f"{repo}/output/plots/optimisation/ecm/cat")    # Selection optimisation plots
loc.PLOTS_FSR           = LocPath(f"{repo}/output/plots/fsr/ecm/cat")             # FSR analysis plots



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

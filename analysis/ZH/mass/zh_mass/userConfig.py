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

from path import loc, LocPath
from .logger import get_logger

LOGGER = get_logger(__name__)



#############################
##### LOCATION OF FILES #####
#############################

repo = str(Path(__file__).parent.parent.resolve())

# Templates as LocPath strings with placeholders
loc.ROOT                = LocPath(repo)                    # Repo root
loc.PACKAGE             = LocPath(f"{repo}/package")       # Python package code
loc.OUT                 = LocPath(f"{repo}/output")        # Output root
loc.PLOTS               = LocPath(f"{repo}/output/plots")  # Plots root
loc.DATA                = LocPath(f"{repo}/output/data")   # Data artifacts
loc.TMP                 = LocPath(f"{repo}/output/tmp")    # Scratch state

loc.JSON                = LocPath(f"{repo}/output/tmp/config_json")      # JSON configs
loc.RUN                 = LocPath(f"{repo}/output/tmp/config_json/run")  # Per-run configs

loc.EVENTS              = LocPath(f"{repo}/output/data/events/ecm/cat/full/analysis")  # Analysis samples
loc.EVENTS_TEST         = LocPath(f"{repo}/output/data/events/ecm/cat/test/analysis")  # Test samples for analysis

loc.HIST                = LocPath(f"{repo}/output/data/histograms")                           # Histograms root
loc.HIST_MEASUREMENT    = LocPath(f"{repo}/output/data/histograms/measurement/ecm/cat")       # Measurement histograms

loc.PLOTS_MEASUREMENT   = LocPath(f"{repo}/output/plots/measurement/ecm/cat")   # Analysis plots
loc.PLOTS_FIT           = LocPath(f"{repo}/output/plots/fit/ecm/cat/sel")       # Fit plots

loc.COMBINE             = LocPath(f"{repo}/output/data/combine/sel/ecm/cat")           # Combine root
loc.COMBINE_LOG         = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/log")       # Logs (nominal)
loc.COMBINE_RESULT      = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/results")   # Results (nominal)
loc.COMBINE_DATACARD    = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/datacard")  # Datacards (nominal)
loc.COMBINE_WS          = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/WS")        # Workspaces (nominal)



#################
### FUNCTIONS ###
#################

# __________________________
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
        isTTree = [i for i, filename in enumerate(filenames)
                   if 'events' in uproot.open(filename)]
        if len(isTTree)==len(filenames):
            newprocs.append(proc)
    return newprocs

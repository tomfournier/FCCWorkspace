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

from path import loc, LocPath, PathObj
from logger import get_logger

LOGGER = get_logger(__name__)



#############################
##### LOCATION OF FILES #####
#############################

repo = str(PathObj(__file__).parent.parent.resolve())

# Templates as LocPath strings with placeholders
loc.ROOT        = LocPath(repo)                    # Repo root
loc.PACKAGE     = LocPath(f"{repo}/package")       # Python package code
loc.OUT         = LocPath(f"{repo}/output")        # Output root
loc.PLOTS       = LocPath(f"{repo}/output/plots")  # Plots root
loc.DATA        = LocPath(f"{repo}/output/data")   # Data artifacts
loc.TMP         = LocPath(f"{repo}/output/tmp")    # For temporary files

loc.EVENTS_DIR  = LocPath(f"{repo}/output/data/1-events")      # 1. Event samples
loc.HIST_DIR    = LocPath(f"{repo}/output/data/2-histograms")  # 2. Histograms root directory
loc.FIT_DIR     = LocPath(f"{repo}/output/data/3-combine")     # 3. Statistical fits

loc.EVENTS      = LocPath(f"{repo}/output/data/events/ecm/cat/full")  # Analysis samples
loc.EVENTS_TEST = LocPath(f"{repo}/output/data/events/ecm/cat/test")  # Test samples for analysis

loc.HIST_RAW    = LocPath(f"{repo}/output/data/histograms/1-raw-hists/ecm/cat")      # Raw histograms
loc.HIST_PROC   = LocPath(f"{repo}/output/data/histograms/2-processed/ecm/cat/sel")  # Processed histograms

loc.PLOTS_SEL   = LocPath(f"{repo}/output/plots/1-selection/ecm/cat")  # Analysis plots
loc.PLOTS_FIT   = LocPath(f"{repo}/output/plots/2-fit")                # Fit plots

loc.FIT         = LocPath(f"{repo}/output/data/combine/sel/ecm/cat")           # Combine root
loc.FIT_LOG     = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/log")       # Logs
loc.FIT_RESULT  = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/results")   # Results
loc.FIT_CARD    = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/datacard")  # Datacards
loc.FIT_WS      = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/WS")        # Workspaces

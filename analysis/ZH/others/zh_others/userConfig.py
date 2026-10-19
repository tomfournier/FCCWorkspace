'''Core configuration for path templates and helpers.

Provides:
- Path templates via `loc` with placeholders: `cat`, `ecm`, `sel`.
- Type-flexible expansion with `LocPath.get()` and `loc.get(...)`.
- Bidirectional type conversion via `astype(str)` and `astype(Path)`.

Conventions:
- Templates are expanded with `loc.get()` or `LocPath.get()`.
- Expanded paths can be str or pathlib.Path; both support `astype()`.

Usage:
- path  = loc.EVENTS
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

# Templates as LocPath strings with placeholders: cat, ecm, sel
loc.ROOT     = LocPath(repo)                    # Repository root
loc.PACKAGE  = LocPath(f"{repo}/zh_others")     # Python package directory
loc.OUT      = LocPath(f"{repo}/output")        # Output root directory
loc.DATA     = LocPath(f"{repo}/output/data")   # All data artifacts
loc.PLOTS    = LocPath(f"{repo}/output/plots")  # All plots output
loc.TMP      = LocPath(f"{repo}/output/tmp")    # Temporary/scratch data

# Events directories: templates use {ecm}, {cat} placeholders
loc.EVENTS_CHI2  = LocPath(f"{repo}/output/data/1-events/chi2/ecm/cat")   # Chi2 optimization samples
loc.EVENTS_FSR   = LocPath(f"{repo}/output/data/1-events/FSR/ecm/cat")    # FSR recovery samples

# Histogram directoriess: templates use {ecm}, {cat}, {sel} placeholders
loc.HIST      = LocPath(f"{repo}/output/data/2-histograms")                    # Histograms root directory
loc.HIST_CHI2 = LocPath(f"{repo}/output/data/2-histograms/chi2/ecm/cat/sel")   # Chi2 analysis histograms
loc.HIST_FSR  = LocPath(f"{repo}/output/data/2-histograms/fsr/ecm/cat/sel")    # FSR analysis histograms

# Results directories: templates use {ecm}, {cat} placeholders
loc.CHI2_RES  = LocPath(f"{repo}/output/data/3-results/chi2/ecm/cat")   # Chi2 analysis results
loc.FSR_RES   = LocPath(f"{repo}/output/data/3-results/fsr/ecm/cat")    # FSR analysis results


# Plots directories: templates use {ecm}, {cat} placeholders
loc.PLOTS       = LocPath(f"{repo}/output/plots")                # Plots root directories
loc.PLOTS_CHI2  = LocPath(f"{repo}/output/plots/chi2/ecm/cat")   # Chi2 analysis plots
loc.PLOTS_FSR   = LocPath(f"{repo}/output/plots/fsr/ecm/cat")    # FSR analysis plots

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



##################
### PARAMETERS ###
##################

# Tuple of kinematic variables used as input features for BDT training
input_vars_ll = (
    'leading_p',    'leading_theta',
    'subleading_p', 'subleading_theta',
    'acolinearity', 'acoplanarity',
    'zll_m', 'zll_p', 'zll_theta'
)

input_vars_qq = (
    'leading_p',    'leading_costheta',
    'subleading_p', 'subleading_costheta',
    'acolinearity', 'acoplanarity',
    'zqq_p',        'zqq_costheta',
    'W1_m', 'W1_p', 'W1_costheta',
    'W2_m', 'W2_p', 'W2_costheta',
    'thrust',
    # 'thrust_costheta',
    # 'delta_mWW4'
)



#########################
### LOCATION OF FILES ###
#########################

repo = str(PathObj(__file__).parent.parent.resolve())

# Base locations. Each numbered root is a complete stage of the workflow.
loc.ROOT      = LocPath(repo)                       # Repository root
loc.PACKAGE   = LocPath(f'{repo}/zh_xsec')          # Shared Python package
loc.OUT       = LocPath(f'{repo}/output')           # All generated output
loc.DATA      = LocPath(f'{repo}/output/data')      # Generated data files
loc.PLOTS     = LocPath(f'{repo}/output/plots')     # Generated plots
loc.TMP       = LocPath(f'{repo}/output/tmp')       # Temporary files
loc.JSON      = LocPath(f'{repo}/output/tmp/json')  # json files directory

# Data output's main directories
loc.EVENTS_DIR = LocPath(f'{repo}/output/data/1-events')       # 1. Event samples
loc.HIST_DIR   = LocPath(f'{repo}/output/data/2-histograms')   # 2. Histogram root
loc.MVA_DIR    = LocPath(f'{repo}/output/data/3-MVA')          # 3. MVA inputs and models
loc.FIT_DIR    = LocPath(f'{repo}/output/data/4-combine')      # 4. Statistical fits
loc.NLO_DIR    = LocPath(f'{repo}/output/data/5-nlo')          # 5. Self-coupling fits

# Events: ecm and cat are path placeholders.
loc.EVENTS_TRAIN      = LocPath(f'{repo}/output/data/1-events/ecm/cat/training/full')   # Full  training events
loc.EVENTS            = LocPath(f'{repo}/output/data/1-events/ecm/cat/analysis/full')   # Full  analysis events
loc.EVENTS_TRAIN_TEST = LocPath(f'{repo}/output/data/1-events/ecm/cat/training/test')   # Test  training events
loc.EVENTS_TEST       = LocPath(f'{repo}/output/data/1-events/ecm/cat/analysis/test')   # Test  analysis events
loc.EVENTS_TRAIN_JAN  = LocPath(f'{repo}/output/data/1-events/ecm/cat/training/jan')    # Jan's training events
loc.EVENTS_JAN        = LocPath(f'{repo}/output/data/1-events/ecm/cat/analysis/jan')    # Jan's analysis events

# Histograms: ecm, cat and sel are path placeholders.
loc.HIST_INPUT = LocPath(f'{repo}/output/data/2-histograms/1-inputs/ecm/cat')          # Histogram inputs
loc.HIST_RAW   = LocPath(f'{repo}/output/data/2-histograms/2-raw-hists/ecm/cat')       # Raw histograms
loc.HIST_PROC  = LocPath(f'{repo}/output/data/2-histograms/3-processed/ecm/cat/sel')   # Processed histograms

# MVA: ecm, cat, and sel are path placeholders.
loc.MVA_INPUT  = LocPath(f'{repo}/output/data/3-MVA/1-inputs/ecm/cat/sel')   # MVA input variables
loc.MVA_MODEL  = LocPath(f'{repo}/output/data/3-MVA/2-bdt/ecm/cat/sel')      # Trained BDT models

# Plots: keep the same numbered order as the analysis stages.
loc.PLOT_INPUT = LocPath(f'{repo}/output/plots/1-MVAInputs/ecm/cat')     # Input distributions
loc.PLOT_BDT   = LocPath(f'{repo}/output/plots/2-BDT/ecm/cat/sel')       # BDT evaluation plots
loc.PLOT_MEAS  = LocPath(f'{repo}/output/plots/3-measurement/ecm/cat')   # Measurement plots
loc.PLOT_SCAN  = LocPath(f'{repo}/output/plots/4-fit')                   # Fit scan plots
loc.PLOT_NLO   = LocPath(f'{repo}/output/plots/5-nlo')                   # Self-coupling plots

# Combine outputs: sel, ecm, and cat are path placeholders.
loc.FIT         = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat')             # Combine output root
loc.FIT_NOMINAL = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/1-nominal')   # Nominal fit
loc.FIT_BIAS    = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias')      # Bias tests

loc.FIT_LOG     = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/1-nominal/log')        # Fit logs
loc.FIT_RESULT  = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/1-nominal/results')    # Fit results
loc.FIT_CARD    = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/1-nominal/datacard')   # Nominal datacards
loc.FIT_SCAN    = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/1-nominal/fastscan')   # Fast scans
loc.FIT_WS      = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/1-nominal/WS')         # Workspaces

loc.BIAS_LOG    = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias/log')              # Bias logs
loc.BIAS_RESULT = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias/results/1-bias')   # Bias test results
loc.BIAS_FIT    = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias/results/2-fit')    # Bias fit results
loc.BIAS_CARD   = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias/datacard')         # Bias datacards
loc.BIAS_SCAN   = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias/fastscan')         # Bias fast scans
loc.BIAS_WS     = LocPath(f'{repo}/output/data/4-combine/sel/ecm/cat/2-bias/WS')               # Bias workspaces

# Self-coupling outputs: sel and cat are path placeholders.
loc.SELF        = LocPath(f'{repo}/output/data/5-nlo/sel/cat')            # Self-coupling output root
loc.SELF_LOG    = LocPath(f'{repo}/output/data/5-nlo/sel/cat/log')        # Self-coupling logs
loc.SELF_RESULT = LocPath(f'{repo}/output/data/5-nlo/sel/cat/results')    # Self-coupling results
loc.SELF_CARD   = LocPath(f'{repo}/output/data/5-nlo/sel/cat/datacard')   # Self-coupling datacards
loc.SELF_SCAN   = LocPath(f'{repo}/output/data/5-nlo/sel/cat/fastscan')   # Self-coupling scans
loc.SELF_WS     = LocPath(f'{repo}/output/data/5-nlo/sel/cat/WS')         # Self-coupling workspaces

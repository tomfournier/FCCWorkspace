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
- path  = loc.EVENTS
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
from logger import get_logger

LOGGER = get_logger(__name__)


repo = str(Path(__file__).parent.parent.resolve())

# Templates as LocPath strings with placeholders: cat, ecm, sel
loc.ROOT                = LocPath(repo)                    # Repository root
loc.PACKAGE             = LocPath(f'{repo}/zh_xsec')       # Python package directory
loc.OUT                 = LocPath(f'{repo}/output')        # Output root directory
loc.PLOTS               = LocPath(f'{repo}/output/plots')  # All plots output
loc.DATA                = LocPath(f'{repo}/output/data')   # All data artifacts
loc.TMP                 = LocPath(f'{repo}/output/tmp')    # Temporary/scratch data

loc.JSON                = LocPath(f'{repo}/output/tmp/config_json')      # JSON configuration directory
loc.RUN                 = LocPath(f'{repo}/output/tmp/config_json/run')  # Per-run configuration directory

# Event directories: templates use {ecm}, {cat} placeholders
loc.EVENTS_TRAINING     = LocPath(f'{repo}/output/data/events/ecm/cat/full/training')  # Full event samples for BDT training
loc.EVENTS              = LocPath(f'{repo}/output/data/events/ecm/cat/full/analysis')  # Full event samples for analysis
loc.EVENTS_TRAIN_TEST   = LocPath(f'{repo}/output/data/events/ecm/cat/test/training')  # Test event samples for BDT training
loc.EVENTS_TEST         = LocPath(f'{repo}/output/data/events/ecm/cat/test/analysis')  # Test event samples for analysis
loc.EVENTS_TRAIN_JAN    = LocPath(f'{repo}/output/data/events/ecm/cat/jan/training')   # Test event samples for BDT training with Jan's defintion
loc.EVENTS_JAN          = LocPath(f'{repo}/output/data/events/ecm/cat/jan/analysis')   # Test event samples for analysis with Jan's definition

# Multivariate analysis (BDT): templates use {ecm}, {cat}, {sel} placeholders
loc.MVA                 = LocPath(f'{repo}/output/data/MVA')                        # MVA root directory
loc.MVA_INPUTS          = LocPath(f'{repo}/output/data/MVA/ecm/cat/sel/MVAInputs')  # BDT input variables
loc.BDT                 = LocPath(f'{repo}/output/data/MVA/ecm/cat/sel/BDT')        # Trained BDT models

# Histograms: templates use {ecm}, {cat}, {sel} placeholders
loc.HIST                = LocPath(f'{repo}/output/data/histograms')                           # Histograms root directory
loc.HIST_MVA            = LocPath(f'{repo}/output/data/histograms/MVAInputs/ecm/cat/')        # MVA input variable histograms
loc.HIST_PREPROCESSED   = LocPath(f'{repo}/output/data/histograms/preprocessed/ecm/cat')      # After final selection
loc.HIST_PROCESSED      = LocPath(f'{repo}/output/data/histograms/processed/ecm/cat/sel')     # After histogram processing
loc.HIST_OPTIMISATION   = LocPath(f'{repo}/output/data/histograms/optimisation/ecm/cat/sel')  # Optimisation analysis histograms

# Plots: templates use {ecm}, {cat}, {sel} placeholders
loc.PLOTS_MVA           = LocPath(f'{repo}/output/plots/1-MVAInputs/ecm/cat')       # Input variable distributions
loc.PLOTS_BDT           = LocPath(f'{repo}/output/plots/2-evaluation/ecm/cat/sel')  # BDT performance and scores
loc.PLOTS_MEASUREMENT   = LocPath(f'{repo}/output/plots/3-measurement/ecm/cat')     # Analysis measurement plots
loc.PLOTS_FIT_SCAN      = LocPath(f'{repo}/output/plots/4-fit/scans')               # Likelyhood scan comparison plots
loc.PLOTS_FIT_NLO       = LocPath(f'{repo}/output/plots/4-fit/nlo')                 # NLO scan comparison plots

# Statistical fit: templates use {sel}, {ecm}, {cat} placeholders
loc.COMBINE             = LocPath(f'{repo}/output/data/combine/sel/ecm/cat')           # Combine root directory
loc.COMBINE_NOMINAL     = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/nominal')   # Nominal analysis
loc.COMBINE_BIAS        = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias')      # Bias test results

# Nominal fit outputs: templates use {sel}, {ecm}, {cat} placeholders
loc.NOMINAL_LOG         = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/nominal/log')       # Combine job logs
loc.NOMINAL_RESULT      = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/nominal/results')   # Fit results and plots
loc.NOMINAL_DATACARD    = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/nominal/datacard')  # Combine datacards
loc.NOMINAL_FASTSCAN    = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/nominal/fastscan')  # To check the nll of the fit
loc.NOMINAL_WS          = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/nominal/WS')        # Combine workspaces

# Bias test outputs: templates use {sel}, {ecm}, {cat} placeholders
loc.BIAS_LOG            = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias/log')           # Combine job logs
loc.BIAS_FIT_RESULT     = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias/results/fit')   # Individual toy fit results
loc.BIAS_RESULT         = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias/results/bias')  # Bias summaries and statistics
loc.BIAS_DATACARD       = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias/datacard')      # Combine datacards
loc.BIAS_FASTSCAN       = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias/fastscan')      # To check the nll of the fit
loc.BIAS_WS             = LocPath(f'{repo}/output/data/combine/sel/ecm/cat/bias/WS')            # Combine workspaces

# Self-coupling outputs: templates use {sel}, {ecm}, {cat} placeholders
loc.NLO                = LocPath(f'{repo}/output/data/nlo/sel/cat')               # Self-coupling root directory
loc.NLO_LOG            = LocPath(f'{repo}/output/data/nlo/sel/cat/log')           # Combine job logs
loc.NLO_RESULT         = LocPath(f'{repo}/output/data/nlo/sel/cat/results')       # Fit results and plots
loc.NLO_DATACARD       = LocPath(f'{repo}/output/data/nlo/sel/cat/datacard')      # Combine datacards
loc.NLO_FASTSCAN       = LocPath(f'{repo}/output/data/nlo/sel/cat/fastscan')      # To check the nll of the fit
loc.NLO_WS             = LocPath(f'{repo}/output/data/nlo/sel/cat/WS')            # Combine workspaces



#################
### FUNCTIONS ###
#################

def event(procs: list[str],
          path: str = '',
          end: str = '.root'
          ) -> list[str]:
    '''Filter processes that contain valid ROOT event trees.

    Validates that all files for each process contain the 'events' TTree.
    Supports both single files and directories with multiple files.

    Args:
        procs: List of process names to validate
        path: Base path where process files are located
        end: File extension to search for (default: '.root')

    Returns:
        List of process names where all associated files contain 'events' TTree
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
